#!/usr/bin/env python3
"""
sdr_live.py - Live LimeSDR spectrum (with optional waterfall) using SoapySDR

Example:
  python3 sdr_live.py --freq 100e6 --rate 5e6 --gain 40 --fft 4096 --waterfall
"""

import argparse
import math
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
    p.add_argument("--meshtastic-port", type=str, default="",
                   help="Optional Meshtastic serial port for one-shot GPS capture")
    p.add_argument("--gps-timeout", type=float, default=8.0,
                   help="GPS timeout in seconds for one-shot Meshtastic read")
    p.add_argument("--i2c-bus", type=int, default=1,
                   help="I2C bus used by QMC5883L magnetometer (default: 1)")
    p.add_argument("--mag-addr", type=lambda x: int(x, 0), default=0x0D,
                   help="QMC5883L I2C address (default: 0x0D)")
    p.add_argument("--declination", type=float, default=0.0,
                   help="Magnetic declination in degrees (default: 0.0)")
    return p.parse_args()


def get_gps_once(meshtastic_port: str, timeout_s: float = 8.0):
    """Best-effort one-shot GPS read from Meshtastic."""
    if not meshtastic_port:
        return (None, None, "SKIPPED")

    try:
        from meshtastic.serial_interface import SerialInterface
        from pubsub import pub
    except Exception:
        return (None, None, "NO_MESHTASTIC")

    lat = lon = None
    status = "NO_FIX"
    got = {"ok": False}

    def on_receive(packet, interface):
        nonlocal lat, lon, status
        decoded = packet.get("decoded", {})
        pos = decoded.get("position")
        if not isinstance(pos, dict):
            return
        lat = pos.get("latitude", lat)
        lon = pos.get("longitude", lon)
        if lat is not None and lon is not None:
            status = "FIX"
            got["ok"] = True

    iface = None
    try:
        iface = SerialInterface(meshtastic_port)
        pub.subscribe(on_receive, "meshtastic.receive")
        t0 = time.time()
        while time.time() - t0 < timeout_s and not got["ok"]:
            time.sleep(0.1)
    except Exception as exc:
        return (None, None, f"ERR:{type(exc).__name__}")
    finally:
        try:
            if iface is not None:
                iface.close()
        except Exception:
            pass

    return (lat, lon, status)


def get_heading_once(i2c_bus: int, addr: int, declination_deg: float = 0.0):
    """Best-effort one-shot heading from QMC5883L magnetometer."""
    try:
        from smbus2 import SMBus
    except Exception:
        return (None, "NO_SMBUS2")

    def s16(lo, hi):
        val = (hi << 8) | lo
        return val - 65536 if val & 0x8000 else val

    try:
        with SMBus(i2c_bus) as bus:
            bus.write_byte_data(addr, 0x0A, 0x80)  # reset
            time.sleep(0.01)
            bus.write_byte_data(addr, 0x09, 0x1D)  # continuous, 50 Hz
            time.sleep(0.01)
            data = bus.read_i2c_block_data(addr, 0x00, 6)
    except Exception as exc:
        return (None, f"ERR:{type(exc).__name__}")

    x = s16(data[0], data[1])
    y = s16(data[2], data[3])
    heading = math.degrees(math.atan2(y, x))
    heading = (heading + 360.0 + declination_deg) % 360.0
    return (heading, "OK")


def main():
    args = parse_args()

    print("=== LimeSDR Live Spectrum (SoapySDR) ===")
    print(f"Center freq: {args.freq/1e6:.3f} MHz")
    print(f"Sample rate: {args.rate/1e6:.3f} Msps")
    print(f"Gain:        {args.gain:.1f} dB")
    print(f"FFT size:    {args.fft}")

    lat, lon, gps_status = get_gps_once(args.meshtastic_port, args.gps_timeout)
    heading, heading_status = get_heading_once(args.i2c_bus, args.mag_addr, args.declination)
    heading_txt = "N/A" if heading is None else f"{heading:.2f}°"
    print(f"GPS snapshot: lat={lat} lon={lon} status={gps_status}")
    print(f"Heading:      {heading_txt} status={heading_status}")

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
