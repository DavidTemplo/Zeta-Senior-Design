#!/usr/bin/env python3
import argparse, math, time, csv, os
from datetime import datetime, timezone

from smbus2 import SMBus

# QMC5883L register map (datasheet / common drivers)
REG_X_L = 0x00
REG_CTRL1 = 0x09
REG_CTRL2 = 0x0A
REG_RST_PERIOD = 0x0B

# CTRL1 bitfields: OSR[7:6] RNG[5:4] ODR[3:2] MODE[1:0]
# Example: OSR=512(0b11), RNG=8G(0b01), ODR=50Hz(0b10), MODE=Continuous(0b01) => 0xD9
CTRL1_CONTINUOUS_50HZ_8G_512OSR = 0xD9

def s16(lo, hi):
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v

def qmc_init(bus, addr):
    # reset
    bus.write_byte_data(addr, 0x0A, 0x80)
    time.sleep(0.01)

    # set continuous measurement
    bus.write_byte_data(addr, 0x09, 0x1D)  # continuous, 50Hz
    time.sleep(0.01)


def qmc_read_xyz(bus: SMBus, addr: int):
    # Read 6 bytes starting at 0x00: X_L, X_H, Y_L, Y_H, Z_L, Z_H :contentReference[oaicite:4]{index=4}
    data = bus.read_i2c_block_data(addr, REG_X_L, 6)
    x = s16(data[0], data[1])
    y = s16(data[2], data[3])
    z = s16(data[4], data[5])
    return x, y, z

def heading_deg(x, y, declination_deg=0.0):
    # heading = atan2(Y, X) (module orientation may require swapping/sign flips)
    h = math.degrees(math.atan2(y, x))
    h = (h + 360.0) % 360.0
    h = (h + declination_deg) % 360.0
    return h

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

def append_csv(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i2c-bus", type=int, default=1)
    ap.add_argument("--addr", type=lambda x: int(x, 0), default=0x0D)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--declination", type=float, default=0.0, help="Mag declination in degrees (optional)")
    ap.add_argument("--meshtastic-port", type=str, default="", help="/dev/serial/by-id/... for Heltec")
    ap.add_argument("--gps-timeout", type=float, default=8.0)
    ap.add_argument("--csv", type=str, default="", help="Optional: append results to CSV file")
    args = ap.parse_args()

    # GPS one-shot first (so you can take it outside and keep the compass streaming)
    ts = datetime.now(timezone.utc).isoformat()
    lat = lon = alt = sats = None
    fix = "SKIPPED"
    if args.meshtastic_port:
        lat, lon, alt, fix, sats = get_gps_once(args.meshtastic_port, args.gps_timeout)

    print(f"[GPS] fix={fix} lat={lat} lon={lon} alt_m={alt} sats={sats}")

    with SMBus(args.i2c_bus) as bus:
        qmc_init(bus, args.addr)
        print(f"[MAG] QMC5883L init OK on bus={args.i2c_bus} addr=0x{args.addr:02X}")

        for i in range(args.samples):
            x, y, z = qmc_read_xyz(bus, args.addr)
            hdg = heading_deg(x, y, args.declination)
            print(f"[MAG] {i+1:02d}/{args.samples} heading={hdg:7.2f}°  x={x:6d} y={y:6d} z={z:6d}")

            if args.csv:
                append_csv(args.csv, {
                    "timestamp_utc": ts,
                    "gps_fix": fix,
                    "lat": lat,
                    "lon": lon,
                    "alt_m": alt,
                    "sats": sats,
                    "heading_deg": hdg,
                    "mx": x,
                    "my": y,
                    "mz": z,
                })

            time.sleep(args.interval)

if __name__ == "__main__":
    main()
