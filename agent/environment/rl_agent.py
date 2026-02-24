import json
import random
import socket
import time

HOST = '127.0.0.1'
PORT = 9999

ACTIONS = ["A", "B", "Select", "Start", "Right", "Left", "Up", "Down", "R", "L"]

def connect_to_mgba():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((HOST, PORT))
        print(f"Connected to mGBA at {HOST}:{PORT}")
        return s
    except ConnectionRefusedError:
        print("Connection refused. Make sure mGBA is running and the Lua script is loaded.")
        return None

def print_party_info(party):
    print(f"Party Size: {len(party)}")
    for i, mon in enumerate(party):
        print(f"  Pokemon {i+1}:")
        print(f"    Species ID: {mon['species']}")
        print(f"    Level: {mon['level']}")
        print(f"    HP: {mon['hp']}/{mon['maxHP']}")
        print(f"    Stats: Atk={mon['attack']}, Def={mon['defense']}, Spd={mon['speed']}, SpAtk={mon['spAttack']}, SpDef={mon['spDefense']}")
        print(f"    IVs: {mon['ivs']}")
        print(f"    EVs: {mon['evs']}")
        print(f"    Moves:")
        for j, move in enumerate(mon['movesInfo']):
            if mon['moves'][j] != 0:
                print(f"      - Move {j+1}: ID={mon['moves'][j]}, Power={move['power']}, Type={move['type']}, PP={mon['pp'][j]}/{move['pp']}")

def main():
    s = connect_to_mgba()
    if not s:
        return

    try:
        while True:
            # Receive state
            # Increase buffer size for larger JSON payload
            data = ""
            while True:
                chunk = s.recv(4096).decode('utf-8')
                if not chunk:
                    raise ConnectionError("Connection closed")
                data += chunk
                if "\n" in data:
                    break

            lines = data.strip().split('\n')
            for line in lines:
                if not line: continue
                try:
                    state = json.loads(line)
                    print(f"Player Pos: ({state['x']}, {state['y']})")
                    if 'party' in state:
                        print_party_info(state['party'])

                    # Simple random agent
                    action = random.choice(ACTIONS)
                    if random.random() < 0.5:
                        action = ""

                    # print(f"Action: {action}")
                    s.sendall((action + "\n").encode('utf-8'))

                except json.JSONDecodeError:
                    print(f"Invalid JSON: {line[:50]}...")

    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        s.close()

if __name__ == "__main__":
    main()

