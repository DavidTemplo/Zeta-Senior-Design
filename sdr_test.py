import SoapySDR
from SoapySDR import * # For SOAPY_SDR_RX
import numpy as np

# --- Configuration ---
CENTER_FREQ_HZ = 433.0e6  # LoRa beacon frequency (e.g., 433 MHz)
SAMPLE_RATE_HZ = 10.0e6   # 10 MS/s (Good for initial wideband check)
GAIN_DB = 50              # Set a high gain for initial testing
NUM_SAMPLES = 4096        # Number of samples to capture

def run_rx_test():
    """
    Initializes LimeSDR, captures samples, and prints a summary.
    """
    sdr = None
    rxStream = None
    try:
        # 1. Find and open the device
        print("Finding LimeSDR device...")
        # Use a dictionary to specify the 'lime' driver
        args = dict(driver="lime")
        sdr = SoapySDR.Device(args)
        print(f"Successfully opened device: {sdr.getHardwareInfo()}")

        # 2. Set necessary parameters
        print(f"Setting sample rate to {SAMPLE_RATE_HZ/1e6} MS/s")
        sdr.setSampleRate(SOAPY_SDR_RX, 0, SAMPLE_RATE_HZ)
        
        print(f"Setting center frequency to {CENTER_FREQ_HZ/1e6} MHz")
        sdr.setFrequency(SOAPY_SDR_RX, 0, CENTER_FREQ_HZ)
        
        print(f"Setting gain to {GAIN_DB} dB")
        # Assuming channel 0 and gain element 'LNA' (may vary)
        sdr.setGain(SOAPY_SDR_RX, 0, GAIN_DB) 

        # 3. Stream setup
        # CF32 is Complex Float 32-bit (I & Q samples)
        print("Setting up receive stream...")
        rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        sdr.activateStream(rxStream)

        # 4. Read samples
        buff = np.zeros(NUM_SAMPLES, dtype=np.complex64)
        
        # Read a block of data
        status = sdr.readStream(rxStream, [buff], NUM_SAMPLES)
        
        print("\n--- Capture Summary ---")
        print(f"ReadStream Status: {status.ret}")
        print(f"Captured {NUM_SAMPLES} samples.")
        print(f"First 5 samples (I/Q): {buff[:5]}")

        # Basic check to see if the data is non-zero (i.e., we received something)
        abs_values = np.abs(buff)
        if np.mean(abs_values) > 1e-6:
            print(f"✅ SUCCESS: Average signal magnitude is {np.mean(abs_values):.4f}. Data was successfully captured.")
        else:
            print("⚠️ WARNING: Average signal magnitude is very close to zero. Check antenna, power, and drivers.")

    except Exception as e:
        print("\n--- ❌ ERROR ---")
        print(f"An error occurred: {e}")
        print("Check LimeSDR power supply, USB connection, and LimeSuite/SoapySDR installation.")
        
    finally:
        # Clean up
        if rxStream:
            print("Deactivating and closing stream...")
            sdr.deactivateStream(rxStream)
            sdr.closeStream(rxStream)
        if sdr:
            print("Closing SDR device.")
            sdr = None # Device release

if __name__ == "__main__":
    run_rx_test()