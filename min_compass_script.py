from smbus2 import SMBus
import time, math

ADDR = 0x2C
X_OFF = 0.0
Y_OFF = 127.0


def s16(lo, hi):
    v = (hi << 8) | lo
    return v - 65536 if v & 0x8000 else v

def mmc_trigger(bus):
    # Trigger a measurement
    bus.write_byte_data(ADDR, 0x0A, 0x01)
    time.sleep(0.01)

def mmc_read_xyz(bus):
    """
    Many MMC5883-family devices output 6 bytes starting at 0x00.
    We'll read 6 bytes and interpret as little-endian 16-bit.
    """
    mmc_trigger(bus)
    data = bus.read_i2c_block_data(ADDR, 0x00, 6)

    x = s16(data[0], data[1])
    y = s16(data[2], data[3])
    z = s16(data[4], data[5])
    return x, y, z

def heading_deg(x, y):
    h = math.degrees(math.atan2(y, x))
    return (h + 360.0) % 360.0

def median3(a,b,c): 
    return sorted([a,b,c])[1]

with SMBus(1) as bus:
    while True:
        x1,y1,z1 = mmc_read_xyz(bus)
        x2,y2,z2 = mmc_read_xyz(bus)
        x3,y3,z3 = mmc_read_xyz(bus)

        x = median3(x1,x2,x3)
        y = median3(y1,y2,y3)
        z = median3(z1,z2,z3)

        hdg = heading_deg(x - X_OFF, y - Y_OFF)

        print(f"heading={hdg:7.2f}  x={x:6d} y={y:6d} z={z:6d}")
        time.sleep(0.25)

