#!/usr/bin/env python3
import time, math, csv, argparse
from datetime import datetime, timezone
from smbus2 import SMBus
import gpsd

ADDR_DEFAULT = 0x2C

# ---------- Compass ----------
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

# ---------- Helpers ----------
def iso_utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

def safe_num(v):
    return v if isinstance(v, (int, float)) else None

def fmt(v, places=6):
    if v is None:
        return "n/a"
    return f"{v:.{places}f}"

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="GPS+Compass CLI + CSV logger with dt/monotonic timing.")
    ap.add_argument("--i2c-bus", type=int, default=1)
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=ADDR_DEFAULT)
    ap.add_argument("--declination", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=0.2)

    ap.add_argument("--cal-seconds", type=float, default=30.0)
    ap.add_argument("--cal-interval", type=float, default=0.05)

    ap.add_argument("--hz", type=float, default=1.0)
    ap.add_argument("--csv", type=str, default=None, help="CSV output path. Default: navlog_YYYYMMDD_HHMMSSZ.csv")
    ap.add_argument("--print-every", type=int, default=1, help="Print every N rows (reduce SSH spam).")
    args = ap.parse_args()

    period = 1.0 / max(args.hz, 0.1)

    # gpsd-py3 connect
    gpsd.connect()

    # CSV file name
    if args.csv is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        csv_path = f"navlog_{ts}.csv"
    else:
        csv_path = args.csv

    headers = [
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
        "x_raw",
        "y_raw",
        "z_raw",
        "x_off",
        "y_off",
    ]

    smooth = None
    row_count = 0

    t_prev = time.monotonic()

    with SMBus(args.i2c_bus) as bus, open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)

        # write header if file empty
        if f.tell() == 0:
            w.writeheader()
            f.flush()

        x_off, y_off = calibrate_offsets(bus, args.addr, args.cal_seconds, args.cal_interval)

        print(f"[RUN] Logging to {csv_path}")
        print("[RUN] Ctrl+C to stop.\n")

        try:
            while True:
                t_now = time.monotonic()
                dt = t_now - t_prev
                t_prev = t_now

                # --- GPS ---
                pkt = gpsd.get_current()
                lat = safe_num(getattr(pkt, "lat", None))
                lon = safe_num(getattr(pkt, "lon", None))
                alt = safe_num(getattr(pkt, "alt", None))
                mode = getattr(pkt, "mode", 0)

                # gpsd-py3 sometimes exposes sats as "sats" (used) — keep as-is if present
                sats_used = getattr(pkt, "sats", None)

                # --- Compass ---
                x, y, z = read_xyz_med(bus, args.addr)
                xc = x - x_off
                yc = y - y_off
                hdg_raw = heading_deg(xc, yc, args.declination)
                smooth = ema_angle(smooth, hdg_raw, args.alpha)

                # --- Log ---
                row = {
                    "timestamp_utc": iso_utc_now(),
                    "t_monotonic": f"{t_now:.6f}",
                    "dt": f"{dt:.6f}",
                    "mode": mode,
                    "sats_used": sats_used,
                    "lat": "" if lat is None else f"{lat:.8f}",
                    "lon": "" if lon is None else f"{lon:.8f}",
                    "alt_m": "" if alt is None else f"{alt:.3f}",
                    "heading_raw_deg": f"{hdg_raw:.2f}",
                    "heading_smooth_deg": f"{smooth:.2f}",
                    "x_raw": x,
                    "y_raw": y,
                    "z_raw": z,
                    "x_off": f"{x_off:.1f}",
                    "y_off": f"{y_off:.1f}",
                }
                w.writerow(row)
                f.flush()

                row_count += 1
                if row_count % args.print_every == 0:
                    # Keep this line short to reduce SSH redraw lag
                    print(
                        f"dt={dt:5.2f}s mode={mode} sats={sats_used} "
                        f"lat={fmt(lat)} lon={fmt(lon)} hdg={smooth:6.1f}°",
                        flush=True
                    )

                # --- Pace the loop ---
                # (sleep the remainder; if dt already > period, this becomes ~0)
                sleep_for = max(0.0, period - (time.monotonic() - t_now))
                time.sleep(sleep_for)

        except KeyboardInterrupt:
            print("\n[STOP]")

if __name__ == "__main__":
    main()
