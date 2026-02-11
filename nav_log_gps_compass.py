#!/usr/bin/env python3
"""
nav_log_gps_compass.py

Logs:
- Meshtastic GPS (lat/lon/alt/sats/fix)
- Magnetometer heading (calibrated + smoothed)

Writes one row per interval to a CSV (good for pairing with SDR/IQ logs later).

Notes:
- This is written for the "0x2C MMC5883-family behavior" you discovered (measurement trigger needed).
- Requires meshtastic + pypubsub if using GPS.
- Requires smbus2 for I2C compass.

Run inside your meshtastic venv if using --meshtastic-port:
  source ~/meshtastic-venv/bin/activate
"""

import argparse
import csv
import os
import time
import math
from datetime import datetime, timezone

from smbus2 import SMBus

# -------- Compass (MMC5883-family behavior at 0x2C) --------

def s16(lo, hi):
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v

def trigger_measurement(bus: SMBus, addr: int):
    # Your working pattern: write 0x01 to 0x0A then wait briefly :contentReference[oaicite:2]{index=2}
    bus.write_byte_data(addr, 0x0A, 0x01)
    time.sleep(0.01)

def read_xyz_once(bus: SMBus, addr: int):
    trigger_measurement(bus, addr)
    data = bus.read_i2c_block_data(addr, 0x00, 6)
    x = s16(data[0], data[1])
    y = s16(data[2], data[3])
    z = s16(data[4], data[5])
    return x, y, z

def median3(a, b, c):
    return sorted([a, b, c])[1]

def read_xyz_median(bus: SMBus, addr: int):
    # Median-of-3 to kill occasional outliers/flips
    x1, y1, z1 = read_xyz_once(bus, addr)
    x2, y2, z2 = read_xyz_once(bus, addr)
    x3, y3, z3 = read_xyz_once(bus, addr)
    x = median3(x1, x2, x3)
    y = median3(y1, y2, y3)
    z = median3(z1, z2, z3)
    return x, y, z

def heading_deg(x, y, declination_deg=0.0):
    h = math.degrees(math.atan2(y, x))
    h = (h + 360.0) % 360.0
    return (h + declination_deg) % 360.0

def wrap_angle_diff(a, b):
    # smallest signed difference a-b in degrees
    return (a - b + 180.0) % 360.0 - 180.0

def ema_angle(prev, new, alpha=0.2):
    # exponential moving average on a circle
    if prev is None:
        return new
    d = wrap_angle_diff(new, prev)
    return (prev + alpha * d) % 360.0

def calibrate_offsets(bus: SMBus, addr: int, cal_seconds: float, interval: float):
    print(f"[CAL] Rotate sensor for {cal_seconds:.0f}s...")
    t0 = time.time()
    xmin = ymin = 10**9
    xmax = ymax = -10**9

    while time.time() - t0 < cal_seconds:
        x, y, z = read_xyz_median(bus, addr)
        xmin = min(xmin, x); xmax = max(xmax, x)
        ymin = min(ymin, y); ymax = max(ymax, y)
        time.sleep(interval)

    x_off = (xmax + xmin) / 2.0
    y_off = (ymax + ymin) / 2.0
    print(f"[CAL] x_offset={x_off:.1f}  y_offset={y_off:.1f}")
    return x_off, y_off

# -------- GPS (Meshtastic one-shot, from your iq_log_gps.py pattern) --------

def get_gps_once(meshtastic_port: str, timeout_s: float = 8.0):
    """
    Best-effort GPS snapshot via Meshtastic serial.
    Returns: (lat, lon, alt_m, fix, sats)
    """
    try:
        from meshtastic.serial_interface import SerialInterface
        from pubsub import pub
    except Exception:
        return (None, None, None, "NO_MESHTASTIC", None)

    lat = lon = alt = sats = None
    fix = "NO_FIX"
    got = {"ok": False}

    def on_receive(packet, interface):
        nonlocal lat, lon, alt, sats, fix
        decoded = packet.get("decoded", {})
        pos = decoded.get("position")
        if not isinstance(pos, dict):
            return
        lat = pos.get("latitude", lat)
        lon = pos.get("longitude", lon)
        alt = pos.get("altitude", alt)
        sats = pos.get("satsInView", sats) or pos.get("satsInUse", sats) or sats
        if lat is not None and lon is not None:
            fix = "FIX"
            got["ok"] = True

    iface = None
    try:
        iface = SerialInterface(meshtastic_port)
        pub.subscribe(on_receive, "meshtastic.receive")
        t0 = time.time()
        while time.time() - t0 < timeout_s and not got["ok"]:
            time.sleep(0.1)
    except Exception as e:
        return (None, None, None, f"ERR:{type(e).__name__}", None)
    finally:
        try:
            if iface is not None:
                iface.close()
        except Exception:
            pass

    return (lat, lon, alt, fix, sats)

# -------- CSV --------

def append_csv(path: str, row: dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

# -------- Main --------

def parse_args():
    ap = argparse.ArgumentParser(description="Log Meshtastic GPS + Compass heading to CSV")

    ap.add_argument("--out-csv", type=str, default="nav_log.csv",
                    help="CSV file to append rows to (default: nav_log.csv)")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="Seconds between log rows (default: 1.0)")

    # Compass
    ap.add_argument("--i2c-bus", type=int, default=1)
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=0x2C)
    ap.add_argument("--declination", type=float, default=0.0,
                    help="Mag declination degrees (optional)")
    ap.add_argument("--alpha", type=float, default=0.2,
                    help="Heading smoothing alpha (default: 0.2)")

    # Calibration: either provide offsets, or do a quick calibration at start
    ap.add_argument("--x-off", type=float, default=None, help="Hard-iron X offset")
    ap.add_argument("--y-off", type=float, default=None, help="Hard-iron Y offset")
    ap.add_argument("--cal-seconds", type=float, default=0.0,
                    help="If >0 and offsets not provided, calibrate for this many seconds at start")
    ap.add_argument("--cal-interval", type=float, default=0.05,
                    help="Calibration sample interval (default: 0.05s)")

    # GPS
    ap.add_argument("--meshtastic-port", type=str, default="",
                    help="Meshtastic serial port (/dev/serial/by-id/...)")
    ap.add_argument("--gps-timeout", type=float, default=8.0)
    ap.add_argument("--gps-interval", type=float, default=5.0,
                    help="How often to refresh GPS (seconds). default 5s. Use 0 to do one-shot only.")

    return ap.parse_args()

def main():
    args = parse_args()
    print("[NAV] Logging GPS + Compass to:", args.out_csv)

    last_gps_t = 0.0
    lat = lon = alt = sats = None
    fix = "SKIPPED"

    smoothed = None

    with SMBus(args.i2c_bus) as bus:
        # Offsets
        if args.x_off is None or args.y_off is None:
            if args.cal_seconds > 0:
                x_off, y_off = calibrate_offsets(bus, args.addr, args.cal_seconds, args.cal_interval)
            else:
                x_off, y_off = 0.0, 0.0
                print("[NAV] No offsets provided and no calibration requested; using 0,0 (works but less accurate).")
        else:
            x_off, y_off = float(args.x_off), float(args.y_off)

        print(f"[NAV] Using offsets x_off={x_off:.1f}, y_off={y_off:.1f}")

        # Main loop
        while True:
            now = time.time()
            ts_utc = datetime.now(timezone.utc).isoformat()

            # GPS refresh
            if args.meshtastic_port:
                if args.gps_interval == 0:
                    if fix == "SKIPPED":  # do it once
                        lat, lon, alt, fix, sats = get_gps_once(args.meshtastic_port, args.gps_timeout)
                        print(f"[GPS] fix={fix} lat={lat} lon={lon} alt={alt} sats={sats}")
                else:
                    if now - last_gps_t >= args.gps_interval:
                        lat, lon, alt, fix, sats = get_gps_once(args.meshtastic_port, args.gps_timeout)
                        last_gps_t = now
                        print(f"[GPS] fix={fix} lat={lat} lon={lon} alt={alt} sats={sats}")

            # Compass
            x, y, z = read_xyz_median(bus, args.addr)
            xc = x - x_off
            yc = y - y_off
            hdg = heading_deg(xc, yc, args.declination)
            smoothed = ema_angle(smoothed, hdg, args.alpha)

            row = {
                "timestamp_utc": ts_utc,
                "lat": lat,
                "lon": lon,
                "alt_m": alt,
                "sats": sats,
                "fix": fix,
                "heading_deg": hdg,
                "heading_smooth_deg": smoothed,
                "mx_raw": x,
                "my_raw": y,
                "mz_raw": z,
                "mx": xc,
                "my": yc,
            }
            append_csv(args.out_csv, row)

            print(f"[NAV] hdg={hdg:7.2f} smooth={smoothed:7.2f}  (x,y)=({x:6d},{y:6d})")
            time.sleep(args.interval)

if __name__ == "__main__":
    main()
