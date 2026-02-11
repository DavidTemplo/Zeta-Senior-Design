import time
import subprocess
from meshtastic.serial_interface import SerialInterface
from pubsub import pub

PORT = "/dev/serial/by-id/PUT_YOUR_PAYLOAD_PORT_HERE"

def send_msg(iface, msg):
    iface.sendText(f"PAYLOAD: {msg}")

def run_cmd(command: str):
    command = command.strip().upper()

    if command == "PING":
        return "PONG ✅"

    if command == "STATUS":
        # Example: uptime + disk
        up = subprocess.check_output("uptime -p", shell=True).decode().strip()
        return f"STATUS ✅ {up}"

    if command == "START_LOG":
        # Example placeholder: start a systemd service
        # subprocess.run(["sudo", "systemctl", "start", "banshee_log"], check=False)
        return "START_LOG received ✅"

    if command == "STOP_LOG":
        # subprocess.run(["sudo", "systemctl", "stop", "banshee_log"], check=False)
        return "STOP_LOG received ✅"

    return f"Unknown cmd: {command}"

def on_receive(packet, interface):
    decoded = packet.get("decoded", {})
    if decoded.get("portnum") != "TEXT_MESSAGE_APP":
        return

    text = decoded.get("payload", b"").decode("utf-8", errors="ignore").strip()

    print(f"RX: {text}")
    resp = run_cmd(text)
    send_msg(interface, resp)

def main():
    iface = SerialInterface(PORT)
    pub.subscribe(on_receive, "meshtastic.receive")

    send_msg(iface, "Command listener online ✅")

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
