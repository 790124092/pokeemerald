import socket
import sys
import time

HOST = '127.0.0.1'
PORT = 9999

def check_port():
    print(f"Checking connection to {HOST}:{PORT}...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        s.connect((HOST, PORT))
        print("SUCCESS: Connected to mGBA!")
        s.close()
        return True
    except ConnectionRefusedError:
        print("FAILURE: Connection refused. Port is closed.")
        return False
    except socket.timeout:
        print("FAILURE: Connection timed out. Port is open but not responding (firewall/binding issue?).")
        return False
    except Exception as e:
        print(f"FAILURE: Unexpected error: {e}")
        return False

if __name__ == "__main__":
    check_port()

