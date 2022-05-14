import serial
import binascii
import time

ser = serial.Serial(
    port='/dev/ttyUSB0',
    baudrate=19200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS
)

print(ser.isOpen())

turn_on = "02h 00h 00h 00h 00h 02h"
turn_off = "01h 30h 41h 30h 41h 30h 43h 02h 43h 32h 30h 33h 44h 36h 30h 30h 30h 34h 03h 76h 0Dh"#"02h 01h 00h 00h 00h 03h"
turn_on_cmd = bytearray([int(elem[:2], 16) for elem in turn_on.split(" ")])
turn_off_cmd = bytearray([int(elem[:2], 16) for elem in turn_off.split(" ")])

print(turn_on_cmd)
print(turn_off_cmd)

print(ser.write(turn_off_cmd))
print(ser.write(turn_off_cmd))

time.sleep(1)
response = ser.read(1)
print(response)

ser.close()

