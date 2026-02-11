#!/usr/bin/env python3
"""
gps_monitor.py
Continuously prints GPS position packets received from a Meshtastic node.
"""

import argparse
import time
from meshtastic.serial_interface import SerialInterface
from pubsub import pub

def on_receive(packet, interface):
    decoded = packet.get("decoded", {})
    pos = decoded.get("position")

    if not isinstance(pos, dict):
        return

    lat = pos.get("latitude")
    lon = pos.get("longitude")
    alt = pos.get("altitude")
    sats = pos.get("satsInView") or pos.get("satsInUse")

    print(
        f"GPS: lat={lat} lon={lon} alt={alt}m sats={sats}"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True,
                        help="/dev/serial/by-id/... Heltec device")
    args = parser.parse_args()

    iface = SerialInterface(args.port)
    pub.subscribe(on_receive, "meshtastic.receive")

    print("Listening for GPS packets...")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
