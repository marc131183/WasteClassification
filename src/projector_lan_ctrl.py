import socket

IP = '192.168.0.10'
PORT = 7142

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(3)

try:
    s.connect((IP, PORT))
    try:
        cmd = ""
        while cmd != "stop":
            cmd = input("Command:")
            if cmd == "on":
                cmd = b'\x02\x00\x00\x00\x00\x02'
            elif cmd == "off":
                cmd = b'\x02\x01\x00\x00\x00\x03'
            elif cmd == "info":
                cmd = b'\x03\x8A\x00\x00\x00\x8D'
            else:
                cmd = b''

            s.send(cmd)
            print('Command sent to the projector')

            try:
                data = s.recv(100)
                print('Response from the projector:')
                print(data)
                print("\r")
            except socket.timeout:
                pass

    except WindowsError as err:
        print(f'Projector communication error: {err}')

except WindowsError as err:
    print(f'Socket connection error: {err}')

s.close()
