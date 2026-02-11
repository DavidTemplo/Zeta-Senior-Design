#!/usr/bin/env python3
"""
iq_log_gps.py - Log raw I/Q samples from LimeSDR using SoapySDR, and append a
single GPS+timestamp row to a CSV per capture (via Meshtastic/Heltec).

This keeps the IQ file pure (complex64) and stores "where/when" metadata in a
sidecar session CSV.

Example:
  python3 iq_log_gps.py --freq 915e6 --rate 1e6 --gain 40 --seconds 10 \
      --out iq_915MHz_1Msps_10s.c64 \
      --meshtastic-port "/dev/serial/by-id/usb-...-if00" \
      --session-csv iq_sessions.csv --gps-timeout 8
"""

import argparse
import csv
import os
import time
from datetime import datetime, timezone

import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32


def parse_args():
    p = argparse.ArgumentParser(description="Raw I/Q logger for LimeSDR (+ one-shot GPS CSV)")
    p.add_argument("--freq", type=float, default=915e6,
                   help="Center frequency in Hz (default: 915e6)")
    p.add_argument("--rate", type=float, default=1e6,
                   help="Sample rate in samples/sec (default: 1e6)")
    p.add_argument("--gain", type=float, default=40.0,
                   help="RX gain in dB (default: 40)")
    p.add_argument("--seconds", type=float, default=5.0,
                   help="How many seconds of I/Q to log (default: 5.0)")
    p.add_argument("--chunk-samps", type=int, default=4096,
                   help="Samples per read chunk (default: 4096)")
    p.add_argument("--out", type=str, default="iq_dump.c64",
                   help="Output binary file (complex64, default: iq_dump.c64)")

    # One-shot session metadata
    p.add_argument("--meshtastic-port", type=str, default="",
                   help="Meshtastic serial port (e.g. /dev/serial/by-id/...) to query GPS once")
    p.add_argument("--session-csv", type=str, default="iq_sessions.csv",
                   help="CSV file to append one row per IQ capture (default: iq_sessions.csv)")
    p.add_argument("--gps-timeout", type=float, default=5.0,
                   help="Seconds to wait for a GPS fix/position packet (default: 5.0)")
    return p.parse_args()


def append_session_csv(csv_path: str, row: dict) -> None:
    """Append a single row to csv_path, writing headers if new."""
    exists = os.path.exists(csv_path)
    fieldnames = list(row.keys())
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


def get_gps_once(meshtastic_port: str, timeout_s: float = 5.0):
    """
    Best-effort GPS snapshot via Meshtastic serial.

    Returns: (lat, lon, alt_m, fix, sats)
      - lat/lon/alt_m may be None if no fix
      - fix is one of: FIX, NO_FIX, NO_MESHTASTIC, ERR:<ExceptionName>, SKIPPED
      - sats may be None (Meshtastic packets don't always include it)
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


def main():
    args = parse_args()

    print("=== LimeSDR I/Q Logger (+ one-shot GPS CSV) ===")
    print(f"Center freq: {args.freq/1e6:.3f} MHz")
    print(f"Sample rate: {args.rate/1e6:.3f} Msps")
    print(f"Gain:        {args.gain:.1f} dB")
    print(f"Duration:    {args.seconds:.2f} s")
    print(f"Chunk samps: {args.chunk_samps}")
    print(f"Output file: {args.out}")

    # One-shot session metadata (timestamp + GPS) per capture
    ts_utc = datetime.now(timezone.utc).isoformat()
    lat = lon = alt = sats = None
    fix = "SKIPPED"
    if args.meshtastic_port:
        lat, lon, alt, fix, sats = get_gps_once(args.meshtastic_port, args.gps_timeout)

    append_session_csv(args.session_csv, {
        "timestamp_utc": ts_utc,
        "lat": lat,
        "lon": lon,
        "alt_m": alt,
        "sats": sats,
        "fix": fix,
        "iq_file": args.out,
        "freq_hz": args.freq,
        "rate_sps": args.rate,
        "gain_db": args.gain,
        "seconds": args.seconds,
    })
    print(f"[META] Session logged to {args.session_csv} (fix={fix})")

    # 1) Open device
    print("[SDR] Enumerating devices...")
    devs = SoapySDR.Device.enumerate()
    if not devs:
        raise RuntimeError("No SoapySDR devices found. Check Lime/USB.")
    print(f"[SDR] Using device: {devs[0]}")
    sdr = SoapySDR.Device(devs[0])

    chan = 0
    print("[SDR] Configuring RX...")
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

    print("[LOG] Starting capture...")
    with open(args.out, "wb") as f:
        try:
            while total_samps_logged < total_samps_target:
                sr = sdr.readStream(
                    stream, [buf], args.chunk_samps, timeoutUs=int(1e6)
                )

                if sr.ret <= 0:
                    print(f"[SDR] readStream ret={sr.ret}, skipping...")
                    continue

                samps = buf[:sr.ret]

                # Write raw complex64 samples to file
                samps.tofile(f)

                total_samps_logged += sr.ret

            print(f"[LOG] Done. Logged {total_samps_logged} samples.")
        except KeyboardInterrupt:
            print("\n[LOG] Interrupted by user.")
        finally:
            sdr.deactivateStream(stream)
            sdr.closeStream(stream)
            print("[SDR] Stream closed.")


if __name__ == "__main__":
    main()
