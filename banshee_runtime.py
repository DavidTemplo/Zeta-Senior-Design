#!/usr/bin/env python3
import time
import math
import csv
import argparse
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone

from smbus2 import SMBus
import gpsd

# ============================================================
# CONFIG (to be edited during tests, but not every run)
# ============================================================

DEFAULT_COMPASS_ADDR = 0x2C

ROT_CFG_DEFAULT = {
    # Fast trigger: total degrees and how "fast" is defined
    "fast_total_deg": 720.0,          # two spins
    "fast_rate_min_dps": 120.0,       # deg/sec lower bound considered "fast"
    "fast_rate_max_dps": 900.0,       # deg/sec sanity upper bound
    "fast_timeout_s": 4.0,            # must complete 720 within this window

    # Slow stop: total degrees and how "fast" is defined
    "slow_total_deg": 360.0,          # one slow spin
    "slow_rate_min_dps": 10.0,        # deg/sec lower bound considered "slow"
    "slow_rate_max_dps": 80.0,        # deg/sec upper bound considered "slow"
    "slow_timeout_s": 20.0,           # must complete 360 within this window

    # Noise gating
    "min_step_deg": 1.0,              # ignore tiny heading steps (noise)
    "max_step_deg": 90.0,             # ignore unrealistic single-step jumps
}

# ============================================================
# STATE MACHINE 
#  0: CALIBRATE (rotate sensor, compute offsets)
#  1: READY (wait for FAST_DOUBLE trigger, optionally require GPS fix)
#  2: LOGGING (write NAV samples, wait for SLOW_SINGLE to stop)
# ============================================================

class State(Enum):
    CALIBRATE = 0
    READY = 1
    LOGGING = 2

# ============================================================
# COMPASS + GPS UTILITIES
# ============================================================

def s16(lo, hi):
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v

def trigger(bus, addr):
    bus.write_byte_data(addr, 0x0A, 0x01)
    time.sleep(0.01)

def read_xyz(bus, addr):
    trigger(bus, addr)
    d = bus.read_i2c_block_data(addr, 0x00, 6)
    x = s16(d[0], d[1])
    y = s16(d[2], d[3])
    z = s16(d[4], d[5])
    return x, y, z

def median3(a, b, c):
    return sorted([a, b, c])[1]

def read_xyz_med(bus, addr):
    x1, y1, z1 = read_xyz(bus, addr)
    x2, y2, z2 = read_xyz(bus, addr)
    x3, y3, z3 = read_xyz(bus, addr)
    return median3(x1, x2, x3), median3(y1, y2, y3), median3(z1, z2, z3)

def heading_deg(x, y, declination_deg=0.0):
    h = math.degrees(math.atan2(y, x))
    h = (h + 360.0) % 360.0
    return (h + declination_deg) % 360.0

def wrap_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0

def ema_angle(prev, new, alpha):
    if prev is None:
        return new
    d = wrap_diff(new, prev)
    return (prev + alpha * d) % 360.0

def calibrate_offsets(bus, addr, seconds, interval):
    print(f"[CAL] Rotate sensor for {seconds:.0f}s...")
    t0 = time.time()
    xmin = ymin =  10**9
    xmax = ymax = -10**9
    while time.time() - t0 < seconds:
        x, y, _ = read_xyz_med(bus, addr)
        xmin = min(xmin, x); xmax = max(xmax, x)
        ymin = min(ymin, y); ymax = max(ymax, y)
        time.sleep(interval)
    x_off = (xmax + xmin) / 2.0
    y_off = (ymax + ymin) / 2.0
    print(f"[CAL] x_off={x_off:.1f}  y_off={y_off:.1f}")
    return x_off, y_off

def iso_utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

def safe_num(v):
    return v if isinstance(v, (int, float)) else None

def fmt(v, places=6):
    if v is None:
        return "n/a"
    return f"{v:.{places}f}"

# ============================================================
# RATE-AWARE ROTATION DETECTOR
# ============================================================

@dataclass
class RotationEvent:
    kind: str                 # "FAST_DOUBLE" or "SLOW_SINGLE"
    total_deg: float
    elapsed_s: float
    avg_rate_dps: float

class RateAwareRotationDetector:
    """
    Integrates absolute heading change only when angular rate falls inside the
    configured window. Maintains independent accumulators for FAST and SLOW.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.prev_heading = None
        self.prev_t = None

        self.fast_accum = 0.0
        self.fast_t0 = None

        self.slow_accum = 0.0
        self.slow_t0 = None

    def reset_fast(self):
        self.fast_accum = 0.0
        self.fast_t0 = None

    def reset_slow(self):
        self.slow_accum = 0.0
        self.slow_t0 = None

    def update(self, heading_deg_now: float, t_now: float) -> RotationEvent | None:
        if self.prev_heading is None:
            self.prev_heading = heading_deg_now
            self.prev_t = t_now
            return None

        dt = t_now - (self.prev_t or t_now)
        if dt <= 0:
            self.prev_heading = heading_deg_now
            self.prev_t = t_now
            return None

        step = wrap_diff(heading_deg_now, self.prev_heading)
        step_abs = abs(step)
        rate = step_abs / dt

        self.prev_heading = heading_deg_now
        self.prev_t = t_now

        if step_abs < self.cfg["min_step_deg"]:
            return None
        if step_abs > self.cfg["max_step_deg"]:
            return None

        # FAST
        if self.cfg["fast_rate_min_dps"] <= rate <= self.cfg["fast_rate_max_dps"]:
            if self.fast_t0 is None:
                self.fast_t0 = t_now
            self.fast_accum += step_abs

            elapsed = t_now - self.fast_t0
            if elapsed > self.cfg["fast_timeout_s"]:
                self.reset_fast()
            elif self.fast_accum >= self.cfg["fast_total_deg"]:
                avg_rate = self.fast_accum / max(elapsed, 1e-6)
                ev = RotationEvent("FAST_DOUBLE", self.fast_accum, elapsed, avg_rate)
                self.reset_fast()
                return ev
        else:
            if self.fast_t0 is not None and (t_now - self.fast_t0) > self.cfg["fast_timeout_s"]:
                self.reset_fast()

        # SLOW
        if self.cfg["slow_rate_min_dps"] <= rate <= self.cfg["slow_rate_max_dps"]:
            if self.slow_t0 is None:
                self.slow_t0 = t_now
            self.slow_accum += step_abs

            elapsed = t_now - self.slow_t0
            if elapsed > self.cfg["slow_timeout_s"]:
                self.reset_slow()
            elif self.slow_accum >= self.cfg["slow_total_deg"]:
                avg_rate = self.slow_accum / max(elapsed, 1e-6)
                ev = RotationEvent("SLOW_SINGLE", self.slow_accum, elapsed, avg_rate)
                self.reset_slow()
                return ev
        else:
            if self.slow_t0 is not None and (t_now - self.slow_t0) > self.cfg["slow_timeout_s"]:
                self.reset_slow()

        return None

# ============================================================
# MESHTASTIC (minimal send-only integration)
# ============================================================

_MESH = {"iface": None}

def meshtastic_init(port: str | None):
    if not port:
        _MESH["iface"] = None
        return

    try:
        from meshtastic.serial_interface import SerialInterface
        iface = SerialInterface(port)
        _MESH["iface"] = iface
        print(f"[MESH] connected on {port}", flush=True)

        # IMPORTANT: wait until the radio has sent config/nodedb
        # otherwise sendText can appear "stuck" until program exit
        try:
            ok = iface.waitForConfig()   # blocks until config received
            print(f"[MESH] waitForConfig={ok}", flush=True)
        except Exception as e:
            print(f"[MESH] waitForConfig failed: {type(e).__name__}: {e}", flush=True)

    except Exception as e:
        _MESH["iface"] = None
        print(f"[MESH] init failed: {type(e).__name__}: {e}", flush=True)


def meshtastic_send(msg: str):
    iface = _MESH.get("iface")
    print(f"[MESHDBG] iface={'OK' if iface else 'NONE'} msg={msg}", flush=True)
    if iface is None:
        print(f"[MESH] {msg}", flush=True)
        return
    try:
        iface.sendText(msg)
    except Exception as e:
        print(f"[MESH] send failed: {type(e).__name__}: {e} | msg={msg}", flush=True)

def meshtastic_close():
    iface = _MESH.get("iface")
    if iface is None:
        return
    try:
        iface.close()
    except Exception:
        pass
    _MESH["iface"] = None

# ============================================================
# SDR HOOKS (stubs so you don't crash on trigger)
# ============================================================

def sdr_start(run_id: str, rf_csv: str):
    # TODO: replace with IQ capture hook / subprocess spawn
    print(f"[SDR] START run_id={run_id} rf_csv={rf_csv}", flush=True)

def sdr_stop(run_id: str):
    print(f"[SDR] STOP run_id={run_id}", flush=True)

# ============================================================
# NAV SAMPLE READER
# ============================================================

def read_nav_sample(bus, addr, x_off, y_off, smooth, declination, alpha):
    # --- GPS ---
    try:
        pkt = gpsd.get_current()
    except Exception:
        pkt = None

    lat = safe_num(getattr(pkt, "lat", None)) if pkt else None
    lon = safe_num(getattr(pkt, "lon", None)) if pkt else None
    alt = safe_num(getattr(pkt, "alt", None)) if pkt else None
    mode = getattr(pkt, "mode", 0) if pkt else 0
    sats_used = getattr(pkt, "sats", None) if pkt else None

    # --- Compass ---
    x, y, z = read_xyz_med(bus, addr)
    xc = x - x_off
    yc = y - y_off
    hdg_raw = heading_deg(xc, yc, declination)
    smooth = ema_angle(smooth, hdg_raw, alpha)

    return {
        "lat": lat,
        "lon": lon,
        "alt_m": alt,
        "mode": mode,
        "sats_used": sats_used,
        "heading_raw": hdg_raw,
        "heading_smooth": smooth,
        "x_raw": x,
        "y_raw": y,
        "z_raw": z,
        "smooth_state": smooth,
    }

# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="BANSHEE runtime state-machine logger (GPS+Compass with rate-aware rotation triggers).")
    ap.add_argument("--i2c-bus", type=int, default=1)
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=DEFAULT_COMPASS_ADDR)
    ap.add_argument("--declination", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=0.2)

    ap.add_argument("--cal-seconds", type=float, default=30.0)
    ap.add_argument("--cal-interval", type=float, default=0.05)

    ap.add_argument("--hz", type=float, default=1.0)
    ap.add_argument("--print-every", type=int, default=1)

    ap.add_argument("--nav-csv", type=str, default=None, help="NAV CSV path (default nav_<runid>.csv)")
    ap.add_argument("--rf-csv", type=str, default=None, help="RF CSV path (default rf_<runid>.csv)")

    ap.add_argument("--require-3d-fix", action="store_true", default=True,
                    help="If set (default), only allow trigger when GPS mode >= 3.")
    ap.add_argument("--allow-no-fix", action="store_true",
                    help="Override: allow trigger even without GPS 3D fix.")

    # Rotation tuning overrides (optional)
    ap.add_argument("--fast-rate-min", type=float, default=ROT_CFG_DEFAULT["fast_rate_min_dps"])
    ap.add_argument("--fast-rate-max", type=float, default=ROT_CFG_DEFAULT["fast_rate_max_dps"])
    ap.add_argument("--fast-timeout", type=float, default=ROT_CFG_DEFAULT["fast_timeout_s"])
    ap.add_argument("--slow-rate-min", type=float, default=ROT_CFG_DEFAULT["slow_rate_min_dps"])
    ap.add_argument("--slow-rate-max", type=float, default=ROT_CFG_DEFAULT["slow_rate_max_dps"])
    ap.add_argument("--slow-timeout", type=float, default=ROT_CFG_DEFAULT["slow_timeout_s"])

    ap.add_argument("--meshtastic-port", type=str, default=None, help="Meshtastic serial port (/dev/serial/by-id/...)")

    args = ap.parse_args()

    # require-3d-fix behavior
    require_fix = (not args.allow_no_fix)

    # build runtime cfg (easy to tweak)
    rot_cfg = dict(ROT_CFG_DEFAULT)
    rot_cfg["fast_rate_min_dps"] = args.fast_rate_min
    rot_cfg["fast_rate_max_dps"] = args.fast_rate_max
    rot_cfg["fast_timeout_s"] = args.fast_timeout
    rot_cfg["slow_rate_min_dps"] = args.slow_rate_min
    rot_cfg["slow_rate_max_dps"] = args.slow_rate_max
    rot_cfg["slow_timeout_s"] = args.slow_timeout

    detector = RateAwareRotationDetector(rot_cfg)
    period = 1.0 / max(args.hz, 0.1)

    # connect gpsd
    gpsd.connect()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    nav_csv = args.nav_csv or f"nav_{run_id}.csv"
    rf_csv = args.rf_csv or f"rf_{run_id}.csv"

    nav_headers = [
        "run_id",
        "sample_index",
        "timestamp_utc",
        "t_monotonic",
        "dt",
        "mode",
        "sats_used",
        "lat",
        "lon",
        "alt_m",
        "heading_raw_deg",
        "heading_smooth_deg",
    ]

    x_off = 0.0
    y_off = 0.0

    state = State.CALIBRATE
    smooth = None
    sample_index = 0
    row_count = 0
    t_prev = time.monotonic()

    with SMBus(args.i2c_bus) as bus, open(nav_csv, "a", newline="") as nav_f:
        meshtastic_init(args.meshtastic_port)

        nav_w = csv.DictWriter(nav_f, fieldnames=nav_headers)
        if nav_f.tell() == 0:
            nav_w.writeheader()
            nav_f.flush()

        meshtastic_send(f"BOOT {run_id}")

        try:
            while True:
                t_loop = time.monotonic()
                dt = t_loop - t_prev
                t_prev = t_loop

                # Light compass read each loop
                x, y, _ = read_xyz_med(bus, args.addr)

                # ----------------------------
                # CALIBRATE
                # ----------------------------
                if state == State.CALIBRATE:
                    x_off, y_off = calibrate_offsets(bus, args.addr, args.cal_seconds, args.cal_interval)
                    meshtastic_send("READY")
                    state = State.READY
                    detector = RateAwareRotationDetector(rot_cfg)
                    smooth = None

                # ----------------------------
                # READY: wait for FAST_DOUBLE (optionally require GPS fix)
                # ----------------------------
                elif state == State.READY:
                    xc = x - x_off
                    yc = y - y_off
                    hdg_raw = heading_deg(xc, yc, args.declination)
                    smooth = ema_angle(smooth, hdg_raw, args.alpha)

                    # Check GPS mode for gating
                    pkt = None
                    try:
                        pkt = gpsd.get_current()
                    except Exception:
                        pass
                    gps_mode = getattr(pkt, "mode", 0) if pkt else 0
                    sats_used = getattr(pkt, "sats", None) if pkt else None

                    if require_fix and gps_mode < 3:
                        # still show status but do not allow trigger
                        if row_count % args.print_every == 0:
                            print(f"[READY] mode={gps_mode} sats={sats_used} hdg={smooth:6.1f}° (waiting for 3D fix)", flush=True)
                        row_count += 1
                    else:
                        ev = detector.update(smooth, t_loop)
                        if ev and ev.kind == "FAST_DOUBLE":
                            meshtastic_send(f"TRIGGER_DETECTED fast total={ev.total_deg:.0f}deg avg={ev.avg_rate_dps:.0f}dps")
                            meshtastic_send(f"LOGGING_START {run_id}")
                            sdr_start(run_id, rf_csv)
                            sample_index = 0
                            state = State.LOGGING

                        if row_count % args.print_every == 0:
                            print(f"[READY] mode={gps_mode} sats={sats_used} hdg={smooth:6.1f}°", flush=True)
                        row_count += 1

                # ----------------------------
                # LOGGING: write NAV; stop on SLOW_SINGLE
                # ----------------------------
                elif state == State.LOGGING:
                    nav = read_nav_sample(bus, args.addr, x_off, y_off, smooth, args.declination, args.alpha)
                    smooth = nav["smooth_state"]

                    nav_row = {
                        "run_id": run_id,
                        "sample_index": sample_index,
                        "timestamp_utc": iso_utc_now(),
                        "t_monotonic": f"{t_loop:.6f}",
                        "dt": f"{dt:.6f}",
                        "mode": nav["mode"],
                        "sats_used": nav["sats_used"],
                        "lat": "" if nav["lat"] is None else f"{nav['lat']:.8f}",
                        "lon": "" if nav["lon"] is None else f"{nav['lon']:.8f}",
                        "alt_m": "" if nav["alt_m"] is None else f"{nav['alt_m']:.3f}",
                        "heading_raw_deg": f"{nav['heading_raw']:.2f}",
                        "heading_smooth_deg": f"{nav['heading_smooth']:.2f}",
                    }
                    nav_w.writerow(nav_row)
                    nav_f.flush()

                    sample_index += 1
                    row_count += 1

                    ev = detector.update(nav["heading_smooth"], t_loop)
                    if ev and ev.kind == "SLOW_SINGLE":
                        sdr_stop(run_id)
                        meshtastic_send(f"LOGGING_STOP {run_id} slow total={ev.total_deg:.0f}deg avg={ev.avg_rate_dps:.0f}dps")
                        meshtastic_send("READY")
                        state = State.READY
                        detector = RateAwareRotationDetector(rot_cfg)
                        smooth = None

                    if row_count % args.print_every == 0:
                        print(
                            f"[LOG] dt={dt:5.2f}s mode={nav['mode']} sats={nav['sats_used']} "
                            f"lat={fmt(nav['lat'])} lon={fmt(nav['lon'])} hdg={nav['heading_smooth']:6.1f}°",
                            flush=True
                        )

                # ----------------------------
                # Pace loop
                # ----------------------------
                sleep_for = max(0.0, period - (time.monotonic() - t_loop))
                time.sleep(sleep_for)

        except KeyboardInterrupt:
            print("\n[STOP] Ctrl+C", flush=True)
            if state == State.LOGGING:
                try:
                    sdr_stop(run_id)
                except Exception as e:
                    print(f"[STOP] sdr_stop error: {type(e).__name__}: {e}", flush=True)
                try:
                    meshtastic_send(f"LOGGING_STOP {run_id} (interrupt)")
                except Exception:
                    pass

        finally:
            meshtastic_close()

if __name__ == "__main__":
    main()
