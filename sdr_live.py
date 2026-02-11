#!/usr/bin/env python3
"""
sdr_live.py - Live LimeSDR spectrum (with optional waterfall) using SoapySDR

Example:
  python3 sdr_live.py --freq 100e6 --rate 5e6 --gain 40 --fft 4096 --waterfall
"""

import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32


def parse_args():
    p = argparse.ArgumentParser(description="Live LimeSDR spectrum viewer")
    p.add_argument("--freq", type=float, default=100e6,
                   help="Center frequency in Hz (default: 100e6)")
    p.add_argument("--rate", type=float, default=5e6,
                   help="Sample rate in samples/sec (default: 5e6)")
    p.add_argument("--gain", type=float, default=40.0,
                   help="RX gain in dB (default: 40)")
    p.add_argument("--fft", type=int, default=4096,
                   help="FFT size / samples per frame (default: 4096)")
    p.add_argument("--waterfall", action="store_true",
                   help="Show scrolling waterfall under the spectrum")
    return p.parse_args()


def main():
    args = parse_args()

    print("=== LimeSDR Live Spectrum (SoapySDR) ===")
    print(f"Center freq: {args.freq/1e6:.3f} MHz")
    print(f"Sample rate: {args.rate/1e6:.3f} Msps")
    print(f"Gain:        {args.gain:.1f} dB")
    print(f"FFT size:    {args.fft}")

    # ---------------------------------------------------------------
    # 1. Open device via enumerate() so we don't depend on exact args
    # ---------------------------------------------------------------
    print("\n[1] Enumerating devices...")
    devs = SoapySDR.Device.enumerate()
    print(f"Devices found: {len(devs)}")
    for i, d in enumerate(devs):
        print(f"  [{i}] {d}")
    if not devs:
        raise RuntimeError("No SoapySDR devices found. "
                           "Check SoapySDRUtil --find.")

    print("[1] Opening device 0...")
    sdr = SoapySDR.Device(devs[0])

    # ---------------------------------------------------------------
    # 2. Configure RX
    # ---------------------------------------------------------------
    chan = 0
    print("[2] Configuring RX chain...")
    sdr.setSampleRate(SOAPY_SDR_RX, chan, args.rate)
    sdr.setFrequency(SOAPY_SDR_RX, chan, args.freq)
    sdr.setGain(SOAPY_SDR_RX, chan, args.gain)

    try:
        sdr.setBandwidth(SOAPY_SDR_RX, chan, args.rate)
    except Exception:
        pass

    actual_rate = sdr.getSampleRate(SOAPY_SDR_RX, chan)
    actual_freq = sdr.getFrequency(SOAPY_SDR_RX, chan)
    print(f"    Actual center freq: {actual_freq/1e6:.3f} MHz")
    print(f"    Actual sample rate: {actual_rate/1e6:.3f} Msps")

    # ---------------------------------------------------------------
    # 3. Setup stream
    # ---------------------------------------------------------------
    print("[3] Setting up RX stream...")
    stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [chan])
    sdr.activateStream(stream)
    time.sleep(0.1)

    # ---------------------------------------------------------------
    # 4. Prepare plotting
    # ---------------------------------------------------------------
    print("[4] Initializing plot...")
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 5))

    # Frequency axis (offset from center, in MHz)
    freqs = np.fft.fftshift(
        np.fft.fftfreq(args.fft, d=1.0 / actual_rate)
    ) / 1e6  # MHz offset

    line, = ax.plot(freqs, np.full(args.fft, -100.0))
    ax.set_xlabel("Offset from center (MHz)")
    ax.set_ylabel("Power (dB)")
    ax.set_ylim(-100, 80)
    ax.set_xlim(freqs[0], freqs[-1])
    ax.grid(True)
    ax.set_title(f"LimeSDR Live Spectrum @ {actual_freq/1e6:.3f} MHz")

    # Optional waterfall
    if args.waterfall:
        n_lines = 200
        wf_data = np.full((n_lines, args.fft), -120.0, dtype=np.float32)
        wf_ax = fig.add_axes([0.10, 0.07, 0.8, 0.25])  # [left, bottom, width, height]
        wf_im = wf_ax.imshow(
            wf_data,
            aspect="auto",
            extent=[freqs[0], freqs[-1], 0, n_lines],
            vmin=-120, vmax=0,
            origin="lower",
        )
        wf_ax.set_ylabel("Time →")
        wf_ax.set_xlabel("Offset (MHz)")

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # ---------------------------------------------------------------
    # 5. Live loop
    # ---------------------------------------------------------------
    print("[5] Entering live loop (Ctrl+C to quit)...")
    buf = np.empty(args.fft, dtype=np.complex64)
    fail_count = 0  # track consecutive read failures

    try:
        while True:
            sr = sdr.readStream(stream, [buf], args.fft, timeoutUs=int(1e6))

            # -----------------------------
            # Handle read failures clearly
            # -----------------------------
            if sr.ret <= 0:
                fail_count += 1
                print(f"readStream ret={sr.ret} (failure #{fail_count})")

                # Visually show "no data" instead of freezing old frame
                empty_line = np.full_like(freqs, -120.0, dtype=float)
                line.set_ydata(empty_line)

                if args.waterfall:
                    wf_data[:-1] = wf_data[1:]      # scroll up
                    wf_data[-1] = empty_line        # last row = no data
                    wf_im.set_data(wf_data)

                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)

                # After too many failures, try restarting the stream
                if fail_count >= 10:
                    print("[WARN] Too many read failures, restarting stream...")
                    try:
                        sdr.deactivateStream(stream)
                    except Exception:
                        pass
                    try:
                        sdr.closeStream(stream)
                    except Exception:
                        pass

                    stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [chan])
                    sdr.activateStream(stream)
                    fail_count = 0

                continue

            # If we got here, we have good data
            fail_count = 0
            samples = buf[:sr.ret]

            # Optional: DC removal so center spur is smaller
            # samples = samples - np.mean(samples)

            # Window + FFT
            window = np.hanning(len(samples))
            spec = np.fft.fftshift(np.fft.fft(samples * window))
            power = 20 * np.log10(np.abs(spec) + 1e-12)

            # Update main spectrum line
            line.set_ydata(power)

            # Update waterfall
            if args.waterfall:
                wf_data[:-1] = wf_data[1:]  # scroll up
                wf_data[-1] = power
                wf_im.set_data(wf_data)

            fig.canvas.draw()
            fig.canvas.flush_events()
            # small pause to keep UI responsive
            plt.pause(0.001)

    except KeyboardInterrupt:
        print("\n[!] Stopping...")

    # ---------------------------------------------------------------
    # 6. Cleanup
    # ---------------------------------------------------------------
    sdr.deactivateStream(stream)
    sdr.closeStream(stream)
    print("Done.")


if __name__ == "__main__":
    main()
