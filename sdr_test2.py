#!/usr/bin/env python3
"""
sdr_test.py - Simple LimeSDR RX test using SoapySDR

- Connects to the first LimeSDR found (driver=lime)
- Tunes to a frequency
- Captures samples
- Plots the spectrum

Usage example:
  python3 sdr_test.py --freq 100e6 --rate 5e6 --gain 40 --nsamps 262144
"""

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32


def parse_args():
    p = argparse.ArgumentParser(description="LimeSDR RX test with SoapySDR")
    p.add_argument("--freq", type=float, default=100e6,
                   help="Center frequency in Hz (default: 100e6)")
    p.add_argument("--rate", type=float, default=5e6,
                   help="Sample rate in samples/sec (default: 5e6)")
    p.add_argument("--gain", type=float, default=40.0,
                   help="RX gain in dB (default: 40)")
    p.add_argument("--nsamps", type=int, default=262144,
                   help="Number of samples to capture (default: 262144)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=== LimeSDR RX Test (SoapySDR) ===")
    print(f"Requested center freq: {args.freq/1e6:.3f} MHz")
    print(f"Requested sample rate: {args.rate/1e6:.3f} Msps")
    print(f"Requested gain:        {args.gain:.1f} dB")
    print(f"Samples to capture:    {args.nsamps}")

    # ---------------------------------------------------------------------
    # 1. Create device
    # ---------------------------------------------------------------------
    print("\n[1] Enumerating SoapySDR devices...")
    devs = SoapySDR.Device.enumerate()
    print(f"    Devices found: {len(devs)}")
    for i, d in enumerate(devs):
        print(f"    [{i}] {d}")

    if not devs:
        raise RuntimeError("No SoapySDR devices found from Python. "
                        "Run SoapySDRUtil --find to compare.")

    print("\n[1] Opening first device from enumerate()...")
    sdr = SoapySDR.Device(devs[0])


    # ---------------------------------------------------------------------
    # 2. Configure RX channel 0
    # ---------------------------------------------------------------------
    chan = 0

    print("[2] Configuring device...")
    sdr.setSampleRate(SOAPY_SDR_RX, chan, args.rate)
    sdr.setFrequency(SOAPY_SDR_RX, chan, args.freq)
    sdr.setGain(SOAPY_SDR_RX, chan, args.gain)

    # Optional: set a reasonable bandwidth (can help with some front-ends)
    try:
        sdr.setBandwidth(SOAPY_SDR_RX, chan, args.rate)
    except Exception:
        pass  # not critical

    actual_rate = sdr.getSampleRate(SOAPY_SDR_RX, chan)
    actual_freq = sdr.getFrequency(SOAPY_SDR_RX, chan)
    print(f"    Actual center freq: {actual_freq/1e6:.3f} MHz")
    print(f"    Actual sample rate: {actual_rate/1e6:.3f} Msps")

    # ---------------------------------------------------------------------
    # 3. Setup RX stream
    # ---------------------------------------------------------------------
    print("[3] Setting up RX stream...")
    stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [chan])

    # Small delay to let things settle
    time.sleep(0.1)

    sdr.activateStream(stream)

    # ---------------------------------------------------------------------
    # 4. Capture samples
    # ---------------------------------------------------------------------
    print("[4] Capturing samples...")
    buff = np.empty(args.nsamps, dtype=np.complex64)

    # Read loop in case we don't get everything in one call
    num_captured = 0
    while num_captured < args.nsamps:
        to_read = min(4096, args.nsamps - num_captured)
        sr = sdr.readStream(stream, [buff[num_captured:num_captured+to_read]],
                            to_read, timeoutUs=int(1e6))  # 1s timeout

        if sr.ret > 0:
            num_captured += sr.ret
        else:
            print(f"WARNING: readStream returned {sr.ret}")
            break

    print(f"    Captured {num_captured} samples.")

    # Stop stream
    sdr.deactivateStream(stream)
    sdr.closeStream(stream)

    # ---------------------------------------------------------------------
    # 5. Process + plot FFT
    # ---------------------------------------------------------------------
    if num_captured == 0:
        print("ERROR: No samples captured, nothing to plot.")
        return

    print("[5] Computing FFT...")

    samples = buff[:num_captured]

    # Apply window to reduce spectral leakage
    window = np.hanning(len(samples))
    windowed = samples * window

    # FFT
    spectrum = np.fft.fftshift(np.fft.fft(windowed))
    power = 20 * np.log10(np.abs(spectrum) + 1e-12)  # in dB

    # Frequency axis in MHz
    freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), d=1.0 / actual_rate))
    freqs_mhz = (freqs + actual_freq) / 1e6

    print(f"    Power range: {power.min():.1f} dB to {power.max():.1f} dB")

    # ---------------------------------------------------------------------
    # 6. Plot
    # ---------------------------------------------------------------------
    print("[6] Plotting spectrum...")
    plt.figure(figsize=(10, 5))
    plt.plot(freqs_mhz, power)
    plt.title(f"LimeSDR Spectrum @ {actual_freq/1e6:.3f} MHz")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
