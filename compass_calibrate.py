#!/usr/bin/env python3
import time, math, argparse
from smbus2 import SMBus

ADDR = 0x2C

def s16(lo, hi):
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v

def trigger(bus):
    bus.write_byte_data(ADDR, 0x0A, 0x01)
    time.sleep(0.01)

def read_xyz(bus):
    trigger(bus)
    data = bus.read_i2c_block_data(ADDR, 0x00, 6)
    x = s16(data[0], data[1])
    y = s16(data[2], data[3])
    z = s16(data[4], data[5])
    return x, y, z

def heading_deg(x, y, decl=0.0):
    h = math.degrees(math.atan2(y, x))
    h = (h + 360.0) % 360.0
    return (h + decl) % 360.0

def wrap_angle_diff(a, b):
    # smallest signed difference a-b in degrees
    d = (a - b + 180.0) % 360.0 - 180.0
    return d

def ema_angle(prev, new, alpha=0.2):
    # exponential moving average on a circle
    if prev is None:
        return new
    d = wrap_angle_diff(new, prev)
    return (prev + alpha * d) % 360.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i2c-bus", type=int, default=1)
    ap.add_argument("--addr", type=lambda x: int(x,0), default=0x2C)
    ap.add_argument("--cal-seconds", type=float, default=30.0)
    ap.add_argument("--interval", type=float, default=0.05)
    ap.add_argument("--declination", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=0.2)
    args = ap.parse_args()

    global ADDR
    ADDR = args.addr

    with SMBus(args.i2c_bus) as bus:
        # --- CALIBRATE ---
        print(f"[CAL] Rotate the sensor in all directions for {args.cal_seconds:.0f}s...")
        t0 = time.time()
        xmin = ymin =  10**9
        xmax = ymax = -10**9

        while time.time() - t0 < args.cal_seconds:
            x, y, z = read_xyz(bus)
            xmin = min(xmin, x); xmax = max(xmax, x)
            ymin = min(ymin, y); ymax = max(ymax, y)
            time.sleep(args.interval)

        x_off = (xmax + xmin) / 2.0
        y_off = (ymax + ymin) / 2.0
        print(f"[CAL] xmin={xmin} xmax={xmax} -> x_offset={x_off:.1f}")
        print(f"[CAL] ymin={ymin} ymax={ymax} -> y_offset={y_off:.1f}")
        print("[CAL] Done. Now holding calibrated heading (Ctrl+C to stop).")

        # --- RUN ---
        smoothed = None
        while True:
            x, y, z = read_xyz(bus)
            xc = x - x_off
            yc = y - y_off
            hdg = heading_deg(xc, yc, args.declination)
            smoothed = ema_angle(smoothed, hdg, args.alpha)
            print(f"hdg={hdg:7.2f}  smooth={smoothed:7.2f}  x={x:6d} y={y:6d} z={z:6d}")
            time.sleep(0.25)

if __name__ == "__main__":
    main()
