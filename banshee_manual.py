#!/usr/bin/env python3
"""
banshee_manual.py

Manual capture utility that performs, in one run:
1) One-shot GPS snapshot from gpsd JSON (or optional serial NMEA GPS).
2) LimeSDR raw I/Q logging.
3) Magnetometer heading logging (MMC5883-style I2C reads).
"""

import argparse
import csv
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX

# GPSD mode requires: pip install gpsd-py3
# and gpsd daemon listening on localhost:2947 (e.g., via gpsd.socket).
try:
    import gpsd
except Exception:
    gpsd = None

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = None


def parse_args():
    p = argparse.ArgumentParser(description="Manual GPS + heading + IQ capture")
    p.add_argument("--config", type=str, default="",
                   help="Path to TOML config file")

    # SDR / IQ logging
    p.add_argument("--freq", type=float, default=915e6)
    p.add_argument("--rate", type=float, default=1e6)
    p.add_argument("--gain", type=float, default=40.0)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--chunk-samps", type=int, default=4096)
    p.add_argument("--iq-out", type=str, default="iq_dump.c64")

    # GPS snapshot source (default gpsd JSON)
    p.add_argument("--gps-source", choices=["gpsd", "serial"], default="gpsd",
                   help="GPS source: gpsd JSON (default) or serial NMEA")
    p.add_argument("--gpsd-host", type=str, default="127.0.0.1")
    p.add_argument("--gpsd-port", type=int, default=2947)
    # Used only when --gps-source serial
    p.add_argument("--gps-port", type=str, default="",
                   help="GPS serial port (e.g. /dev/ttyUSB0)")
    p.add_argument("--gps-baud", type=int, default=9600)
    p.add_argument("--gps-timeout", type=float, default=8.0)

    # Heading logging
    p.add_argument("--i2c-bus", type=int, default=1)
    p.add_argument("--mag-addr", type=lambda x: int(x, 0), default=0x2C)
    p.add_argument("--declination", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=0.2,
                   help="heading smoothing alpha")
    p.add_argument("--x-off", type=float, default=0.0)
    p.add_argument("--y-off", type=float, default=0.0)
    p.add_argument("--heading-interval", type=float, default=0.25,
                   help="seconds between heading rows")

    # Metadata/log files
    p.add_argument("--session-csv", type=str, default="banshee_manual_sessions.csv")
    p.add_argument("--heading-csv", type=str, default="banshee_manual_heading.csv")

    return p.parse_args()


def _load_toml_config(path: str):
    if not path:
        return {}
    if tomllib is None:
        raise RuntimeError("TOML config requires Python 3.11+ (tomllib)")

    cfg_path = Path(path)
    with cfg_path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level TOML value must be a table/object")
    return data


def _parse_hex_or_int(value, default: int):
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 0)
    return default


def apply_toml_config(args):
    cfg = _load_toml_config(args.config)
    if not cfg:
        return args

    general = cfg.get("general", {})
    compass = cfg.get("compass", {})
    sdr = cfg.get("sdr", {})

    # NOTE: We currently read gps/meshtastic tables for compatibility only.
    # Keys not used by this script are intentionally ignored.
    _ = cfg.get("gps", {})
    _ = cfg.get("meshtastic", {})

    # [general]
    output_dir = general.get("output_dir")
    if output_dir:
        iq_name = args.iq_out if os.path.dirname(args.iq_out) else os.path.basename(args.iq_out)
        session_name = args.session_csv if os.path.dirname(args.session_csv) else os.path.basename(args.session_csv)
        heading_name = args.heading_csv if os.path.dirname(args.heading_csv) else os.path.basename(args.heading_csv)

        args.iq_out = str(Path(output_dir) / iq_name)
        args.session_csv = str(Path(output_dir) / session_name)
        args.heading_csv = str(Path(output_dir) / heading_name)

    hz = general.get("hz")
    if hz:
        args.heading_interval = 1.0 / float(hz)

    # [compass]
    if "i2c_bus" in compass:
        args.i2c_bus = int(compass["i2c_bus"])
    args.mag_addr = _parse_hex_or_int(compass.get("addr"), args.mag_addr)
    if "declination" in compass:
        args.declination = float(compass["declination"])
    if "alpha" in compass:
        args.alpha = float(compass["alpha"])
    if "x_off" in compass:
        args.x_off = float(compass["x_off"])
    if "y_off" in compass:
        args.y_off = float(compass["y_off"])

    # [sdr]
    if "freq_hz" in sdr:
        args.freq = float(sdr["freq_hz"])
    if "rate_sps" in sdr:
        args.rate = float(sdr["rate_sps"])
    if "gain_db" in sdr:
        args.gain = float(sdr["gain_db"])
    if "chunk_samps" in sdr:
        args.chunk_samps = int(sdr["chunk_samps"])
    if "out" in sdr and sdr["out"]:
        args.iq_out = str(sdr["out"])

    return args


def append_csv(path: str, row: dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def _nmea_to_decimal(raw: str, hemi: str):
    if not raw or not hemi:
        return None
    if hemi in ("N", "S"):
        deg_len = 2
    elif hemi in ("E", "W"):
        deg_len = 3
    else:
        return None

    try:
        degrees = float(raw[:deg_len])
        minutes = float(raw[deg_len:])
    except ValueError:
        return None

    value = degrees + (minutes / 60.0)
    if hemi in ("S", "W"):
        value = -value
    return value


def _valid_lat_lon(lat, lon):
    return (
        isinstance(lat, (int, float))
        and isinstance(lon, (int, float))
        and not math.isnan(lat)
        and not math.isnan(lon)
    )


def get_gps_once_gpsd(host: str, port: int, timeout_s: float):
    try:
        if gpsd is None:
            raise ModuleNotFoundError("gpsd")
        gpsd.connect(host=host, port=port)
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            pkt = gpsd.get_current()
            mode = getattr(pkt, "mode", 0) or 0
            if mode >= 2:
                lat = getattr(pkt, "lat", None)
                lon = getattr(pkt, "lon", None)
                if _valid_lat_lon(lat, lon):
                    return (float(lat), float(lon), "FIX")
            time.sleep(0.2)
    except Exception as exc:
        return (None, None, f"ERR:{type(exc).__name__}")

    return (None, None, "NO_FIX")


def get_gps_once_serial(serial_port: str, baud: int, timeout_s: float):
    if not serial_port:
        return (None, None, "SKIPPED")

    try:
        import serial
    except Exception:
        return (None, None, "NO_PYSERIAL")

    deadline = time.time() + timeout_s
    try:
        with serial.Serial(serial_port, baudrate=baud, timeout=0.5) as ser:
            while time.time() < deadline:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("ascii", errors="ignore").strip()

                if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                    parts = line.split(",")
                    if len(parts) < 7:
                        continue
                    fix_q = parts[6]
                    lat = _nmea_to_decimal(parts[2], parts[3])
                    lon = _nmea_to_decimal(parts[4], parts[5])
                    if fix_q and fix_q != "0" and lat is not None and lon is not None:
                        return (lat, lon, "FIX")

                if line.startswith("$GPRMC") or line.startswith("$GNRMC"):
                    parts = line.split(",")
                    if len(parts) < 7:
                        continue
                    status = parts[2]
                    lat = _nmea_to_decimal(parts[3], parts[4])
                    lon = _nmea_to_decimal(parts[5], parts[6])
                    if status == "A" and lat is not None and lon is not None:
                        return (lat, lon, "FIX")
    except Exception as exc:
        return (None, None, f"ERR:{type(exc).__name__}")

    return (None, None, "NO_FIX")


def get_gps_once(args):
    if args.gps_source == "serial":
        return get_gps_once_serial(args.gps_port, args.gps_baud, args.gps_timeout)
    return get_gps_once_gpsd(args.gpsd_host, args.gpsd_port, args.gps_timeout)


def s16(lo, hi):
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v


def heading_deg(x, y, declination_deg=0.0):
    h = math.degrees(math.atan2(y, x))
    h = (h + 360.0) % 360.0
    return (h + declination_deg) % 360.0


def wrap_angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def ema_angle(prev, new, alpha=0.2):
    if prev is None:
        return new
    d = wrap_angle_diff(new, prev)
    return (prev + alpha * d) % 360.0


def read_heading_once(bus, addr: int, x_off: float, y_off: float, declination_deg: float):
    # MMC5883-style trigger + read from nav_log_gps_compass.py
    bus.write_byte_data(addr, 0x0A, 0x01)
    time.sleep(0.01)
    data = bus.read_i2c_block_data(addr, 0x00, 6)
    x = s16(data[0], data[1])
    y = s16(data[2], data[3])
    z = s16(data[4], data[5])
    xc = x - x_off
    yc = y - y_off
    hdg = heading_deg(xc, yc, declination_deg)
    return x, y, z, xc, yc, hdg


def main():
    args = parse_args()
    args = apply_toml_config(args)

    out_dir = os.path.dirname(args.iq_out) or os.path.dirname(args.session_csv) or os.path.dirname(args.heading_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("=== Banshee Manual Capture ===")
    print(f"Center freq: {args.freq/1e6:.3f} MHz")
    print(f"Sample rate: {args.rate/1e6:.3f} Msps")
    print(f"Gain:        {args.gain:.1f} dB")
    print(f"Duration:    {args.seconds:.2f} s")

    # 1) One-shot GPS snapshot
    lat, lon, gps_status = get_gps_once(args)
    print(f"[GPS] lat={lat} lon={lon} status={gps_status} source={args.gps_source}")

    ts_start = datetime.now(timezone.utc).isoformat()
    append_csv(args.session_csv, {
        "timestamp_utc": ts_start,
        "gps_lat": lat,
        "gps_lon": lon,
        "gps_status": gps_status,
        "iq_file": args.iq_out,
        "heading_file": args.heading_csv,
        "freq_hz": args.freq,
        "rate_sps": args.rate,
        "gain_db": args.gain,
        "seconds": args.seconds,
    })

    # 2) Setup heading logging bus
    bus = None
    heading_status = "OK"
    try:
        from smbus2 import SMBus
        bus = SMBus(args.i2c_bus)
        print(f"[MAG] i2c bus={args.i2c_bus} addr=0x{args.mag_addr:02X}")
    except Exception as exc:
        heading_status = f"ERR:{type(exc).__name__}"
        print(f"[MAG] heading disabled: {heading_status}")

    # 3) Setup SDR and capture IQ while periodically logging heading rows
    devs = SoapySDR.Device.enumerate()
    if not devs:
        raise RuntimeError("No SoapySDR devices found. Check Lime/USB.")
    sdr = SoapySDR.Device(devs[0])

    chan = 0
    sdr.setSampleRate(SOAPY_SDR_RX, chan, args.rate)
    sdr.setGain(SOAPY_SDR_RX, chan, args.gain)
    sdr.setFrequency(SOAPY_SDR_RX, chan, args.freq)
    try:
        sdr.setBandwidth(SOAPY_SDR_RX, chan, args.rate)
    except Exception:
        pass

    stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [chan])
    sdr.activateStream(stream)
    time.sleep(0.1)

    total_samps_target = int(args.rate * args.seconds)
    total_samps_logged = 0
    buf = np.empty(args.chunk_samps, dtype=np.complex64)
    smoothed = None
    next_heading_t = time.time()

    print("[LOG] Capturing IQ + heading...")
    try:
        with open(args.iq_out, "wb") as f:
            while total_samps_logged < total_samps_target:
                sr = sdr.readStream(stream, [buf], args.chunk_samps, timeoutUs=int(1e6))
                if sr.ret > 0:
                    samps = buf[:sr.ret]
                    samps.tofile(f)
                    total_samps_logged += sr.ret

                now = time.time()
                if bus is not None and now >= next_heading_t:
                    row_lat, row_lon, row_gps_status = lat, lon, gps_status
                    if args.gps_source == "gpsd":
                        row_lat, row_lon, row_gps_status = get_gps_once_gpsd(
                            args.gpsd_host, args.gpsd_port, timeout_s=0.5
                        )
                    try:
                        x, y, z, xc, yc, hdg = read_heading_once(
                            bus, args.mag_addr, args.x_off, args.y_off, args.declination
                        )
                        smoothed = ema_angle(smoothed, hdg, args.alpha)
                        append_csv(args.heading_csv, {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "gps_lat": row_lat,
                            "gps_lon": row_lon,
                            "gps_status": row_gps_status,
                            "heading_deg": hdg,
                            "heading_smooth_deg": smoothed,
                            "mx_raw": x,
                            "my_raw": y,
                            "mz_raw": z,
                            "mx": xc,
                            "my": yc,
                            "status": "OK",
                        })
                    except Exception as exc:
                        append_csv(args.heading_csv, {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "gps_lat": row_lat,
                            "gps_lon": row_lon,
                            "gps_status": row_gps_status,
                            "heading_deg": "",
                            "heading_smooth_deg": "",
                            "mx_raw": "",
                            "my_raw": "",
                            "mz_raw": "",
                            "mx": "",
                            "my": "",
                            "status": f"ERR:{type(exc).__name__}",
                        })
                    next_heading_t = now + args.heading_interval

        print(f"[LOG] Done. Logged {total_samps_logged} IQ samples.")
    except KeyboardInterrupt:
        print("\n[LOG] Interrupted by user.")
    finally:
        try:
            sdr.deactivateStream(stream)
            sdr.closeStream(stream)
        except Exception:
            pass
        if bus is not None:
            try:
                bus.close()
            except Exception:
                pass

    print(f"[DONE] Session CSV: {args.session_csv}")
    print(f"[DONE] Heading CSV: {args.heading_csv} ({heading_status})")


if __name__ == "__main__":
    main()
