#!/usr/bin/env python3
"""
sdr_iq_log.py - Log raw I/Q samples from LimeSDR using SoapySDR.

Example:
  # Log 10 seconds of I/Q at 1 Msps, center 915 MHz, gain 40
  python3 sdr_iq_log.py --freq 915e6 --rate 1e6 --gain 40 \
      --seconds 10 --out iq_915MHz_1Msps_10s.c64
"""

import argparse
import time

import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32


def parse_args():
    p = argparse.ArgumentParser(description="Raw I/Q logger for LimeSDR")
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
    return p.parse_args()


def main():
    args = parse_args()

    print("=== LimeSDR I/Q Logger ===")
    print(f"Center freq: {args.freq/1e6:.3f} MHz")
    print(f"Sample rate: {args.rate/1e6:.3f} Msps")
    print(f"Gain:        {args.gain:.1f} dB")
    print(f"Duration:    {args.seconds:.2f} s")
    print(f"Chunk samps: {args.chunk_samps}")
    print(f"Output file: {args.out}")

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
