import json
import os
import re
import struct
import sys
import traceback

try:
    import mgba.core
    import mgba.image
    import mgba.log
except ImportError:
    print("[MGBA]", traceback.format_exc())
    pass

# Constants from pokeemerald
PARTY_SIZE = 6
BOX_POKEMON_SIZE = 80
POKEMON_SIZE = 100
SUBSTRUCT_SIZE = 12
NUM_SUBSTRUCTS = 4

# Offsets in Pokemon struct
OFF_BOX = 0x00
OFF_STATUS = 0x50
OFF_LEVEL = 0x54
OFF_MAIL = 0x55
OFF_HP = 0x56
OFF_MAX_HP = 0x58
OFF_ATK = 0x5A
OFF_DEF = 0x5C
OFF_SPEED = 0x5E
OFF_SPATK = 0x60
OFF_SPDEF = 0x62

# Offsets in BoxPokemon struct
OFF_BP_PERSONALITY = 0x00
OFF_BP_OT_ID = 0x04
OFF_BP_NICKNAME = 0x08
OFF_BP_LANGUAGE = 0x12
OFF_BP_FLAGS = 0x13 # isBadEgg, etc
OFF_BP_OT_NAME = 0x14
OFF_BP_MARKINGS = 0x1B
OFF_BP_CHECKSUM = 0x1C
OFF_BP_UNKNOWN = 0x1E
OFF_BP_SECURE = 0x20

# Map Constants
ADDR_G_SAVE_BLOCK_1_PTR = 0x03005D8C
ADDR_G_SAVE_BLOCK_2_PTR = 0x03005D90

# Bag Pockets
POCKETS = {
    "Items":     {"offset": 0x560, "capacity": 30},
    "KeyItems":  {"offset": 0x5D8, "capacity": 30},
    "PokeBalls": {"offset": 0x650, "capacity": 16},
    "TMsHMs":    {"offset": 0x690, "capacity": 64},
    "Berries":   {"offset": 0x790, "capacity": 46},
}

class TextDecoder:
    def __init__(self, charmap_path="charmap.txt"):
        self.decode_map = {}
        self._load_charmap(charmap_path)

    def _load_charmap(self, path):
        if not os.path.exists(path):
            # Try looking in project root
            root_path = os.path.join(os.path.dirname(__file__), "../../", path)
            if os.path.exists(root_path):
                path = root_path
            else:
                return

        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("@"):
                    continue

                if "=" in line:
                    parts = line.split("=")
                    key = parts[0].strip()
                    value = parts[1].strip()

                    # Parse hex value
                    # Handle comments after value
                    if "@" in value:
                        value = value.split("@")[0].strip()

                    try:
                        hex_bytes = bytes.fromhex(value)
                    except ValueError:
                        continue

                    # Parse key
                    # Remove quotes if present
                    if key.startswith("'") and key.endswith("'"):
                        char = key[1:-1]
                        # Handle escaped chars
                        if char == "\\'": char = "'"
                        elif char == "\\n": char = "\n"
                        elif char == "\\l": char = "" # Scroll
                        elif char == "\\p": char = "\n" # Paragraph
                        self.decode_map[hex_bytes] = char
                    else:
                        # Handle constants like PLAYER, etc.
                        self.decode_map[hex_bytes] = f"<{key}>"

    def decode(self, data):
        result = ""
        i = 0
        while i < len(data):
            # Try to match longest sequence first
            matched = False
            # Check for terminator
            if data[i] == 0xFF:
                break

            for length in [2, 1]: # Try 2 bytes then 1 byte
                if i + length <= len(data):
                    chunk = bytes(data[i:i+length])
                    if chunk in self.decode_map:
                        result += self.decode_map[chunk]
                        i += length
                        matched = True
                        break

            if not matched:
                # Unknown char
                # result += f"\\x{data[i]:02x}"
                result += "?"
                i += 1
        return result

class Emulator:
    def __init__(self, rom_path="pokeemerald.gba", map_file="pokeemerald.map"):
        self.rom_path = os.path.abspath(rom_path)
        print(self.rom_path)

        # Update addresses from map file if it exists
        self._load_addresses_from_map(map_file)
        print(rom_path)
        try:
            self.core = mgba.core.load_path(rom_path)
            if not self.core:
                raise Exception(f"Failed to load ROM: {rom_path}")

            # Initialize video buffer FIRST (before reset!)
            self.width, self.height = 240, 160
            self.image = mgba.image.Image(self.width, self.height)
            self.image.stride = self.width
            self.core.set_video_buffer(self.image)

            # Configure core
            self.core.config["videoRenderer"] = "software"  # Use software renderer for headless
            self.core.config["idleOptimization"] = "ignore"
            self.core.load_config(self.core.config)

            self.core.reset()

            # Check dimensions
            w, h = self.core.desired_video_dimensions()
            print(f"Core video dimensions: {w}x{h}", file=sys.stderr)

            # Check mColor size
            from mgba._pylib import ffi
            print(f"mColor size: {ffi.sizeof('mColor')}", file=sys.stderr)

        except:
            print("Warning: Could not load mGBA core. Emulator functionality will be limited.", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self.core = None

        self.decoder = TextDecoder()
        self.map_names = self._load_map_names()
        self.item_names = self._load_item_names()
        self.species_names = self._load_species_names()

    def _load_addresses_from_map(self, map_file):
        global ADDR_G_SAVE_BLOCK_1_PTR, ADDR_G_SAVE_BLOCK_2_PTR

        if not os.path.exists(map_file):
            # Try looking in project root
            root_path = os.path.join(os.path.dirname(__file__), "../../", map_file)
            if os.path.exists(root_path):
                map_file = root_path
            else:
                return

        try:
            with open(map_file, "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        # Format: 0x03005d8c gSaveBlock1Ptr
                        # or: 0000000003005d8c gSaveBlock1Ptr (depending on linker)
                        symbol = parts[-1]
                        if symbol == "gSaveBlock1Ptr":
                            try:
                                ADDR_G_SAVE_BLOCK_1_PTR = int(parts[0], 16)
                            except ValueError:
                                pass
                        elif symbol == "gSaveBlock2Ptr":
                            try:
                                ADDR_G_SAVE_BLOCK_2_PTR = int(parts[0], 16)
                            except ValueError:
                                pass
        except Exception as e:
            print(f"Warning: Failed to parse map file: {e}")

    def run_frame(self):
        if self.core:
            self.core.run_frame()

    def get_frame_buffer(self):
        if not self.core:
            return None
        if hasattr(self.image, 'to_pil'):
            pil_img = self.image.to_pil()
            # Convert RGBX to RGB (PIL doesn't support RGBX)
            if pil_img.mode == 'RGBX':
                pil_img = pil_img.convert('RGB')
            return pil_img
        return None

    def press_button(self, button):
        if self.core:
            self.core.set_keys(button)

    def release_button(self, button):
        if self.core:
            self.core.clear_keys(button)

    def load_state(self, state_file):
        if self.core and os.path.exists(state_file):
            try:
                with open(state_file, "rb") as f:
                    state_data = f.read()
                    # mgba python binding might not expose load_state directly or it might be named differently
                    # Let's try load_file first if it supports save states, otherwise load_raw_state
                    if state_file.endswith(".ss1") or state_file.endswith(".state"):
                         # Assuming load_file handles state files if extension is recognized,
                         # but usually load_file is for ROMs.
                         # Let's check if we can use load_raw_state
                         self.core.load_raw_state(state_data)
                         return True
            except Exception as e:
                print(f"Failed to load state: {e}")
        return False

    def save_state(self, state_file):
        if self.core:
            try:
                state_data = self.core.save_raw_state()
                if state_data:
                    with open(state_file, "wb") as f:
                        f.write(state_data)
                    return True
            except Exception as e:
                print(f"Failed to save state: {e}")
        return False

    def get_state(self, include_screenshot=True):
        if not self.core:
            return {}

        state = {
            "location": self._read_location(),
            "party": self._read_party(),
            "bag": self._read_bag(),
        }

        if include_screenshot:
            try:
                state["screenshot"] = self.get_frame_buffer()
            except:
                state["screenshot"] = None

        return state

    def _read_u32(self, addr):
        return self.core.memory.u32[addr]

    def _read_u16(self, addr):
        return self.core.memory.u16[addr]

    def _read_u8(self, addr):
        return self.core.memory.u8[addr]

    def _read_bytes(self, addr, length):
        return self.core.memory[addr:addr+length]

    def _read_location(self):
        sb1_addr = self._read_u32(ADDR_G_SAVE_BLOCK_1_PTR)
        if sb1_addr == 0:
            return {"x": 0, "y": 0, "map_group": 0, "map_num": 0, "map_name": "Unknown"}

        x = self._read_u16(sb1_addr + 0x00)
        y = self._read_u16(sb1_addr + 0x02)
        map_group = self._read_u8(sb1_addr + 0x04)
        map_num = self._read_u8(sb1_addr + 0x05)

        map_name = self.map_names.get((map_group, map_num), f"Unknown ({map_group}, {map_num})")

        return {
            "x": x,
            "y": y,
            "map_group": map_group,
            "map_num": map_num,
            "map_name": map_name
        }

    def _read_bag(self):
        sb1_addr = self._read_u32(ADDR_G_SAVE_BLOCK_1_PTR)
        sb2_addr = self._read_u32(ADDR_G_SAVE_BLOCK_2_PTR)

        if sb1_addr == 0 or sb2_addr == 0:
            return {}

        encryption_key = self._read_u32(sb2_addr + 0xAC)
        bag_data = {}

        for pocket_name, config in POCKETS.items():
            items = []
            base_addr = sb1_addr + config["offset"]
            capacity = config["capacity"]

            for i in range(capacity):
                slot_addr = base_addr + (i * 4)
                item_id = self._read_u16(slot_addr)
                encrypted_quantity = self._read_u16(slot_addr + 2)

                if item_id != 0:
                    quantity = encrypted_quantity ^ (encryption_key & 0xFFFF)
                    items.append({
                        "id": item_id,
                        "name": self.item_names.get(item_id, f"Item {item_id}"),
                        "quantity": quantity
                    })
            bag_data[pocket_name] = items

        return bag_data

    def _read_party(self):
        sb1_addr = self._read_u32(ADDR_G_SAVE_BLOCK_1_PTR)
        if sb1_addr == 0:
            return []

        party_addr = sb1_addr + 0x238
        party_count = self._read_u8(sb1_addr + 0x234)

        party = []
        for i in range(min(party_count, PARTY_SIZE)):
            mon_addr = party_addr + (i * POKEMON_SIZE)
            mon_data = self._read_pokemon(mon_addr)
            party.append(mon_data)

        return party

    def _read_pokemon(self, addr):
        personality = self._read_u32(addr + OFF_BP_PERSONALITY)
        ot_id = self._read_u32(addr + OFF_BP_OT_ID)
        nickname_bytes = self._read_bytes(addr + OFF_BP_NICKNAME, 10)
        nickname = self.decoder.decode(nickname_bytes)

        key = ot_id ^ personality
        secure_data = self._read_bytes(addr + OFF_BP_SECURE, 48)
        decrypted_data = bytearray(48)

        for i in range(0, 48, 4):
            word = struct.unpack("<I", secure_data[i:i+4])[0]
            decrypted_word = word ^ key
            decrypted_data[i:i+4] = struct.pack("<I", decrypted_word)

        substructs = [
            decrypted_data[0:12],
            decrypted_data[12:24],
            decrypted_data[24:36],
            decrypted_data[36:48]
        ]

        order = personality % 24
        orders = [
            (0,1,2,3), (0,1,3,2), (0,2,1,3), (0,3,1,2), (0,2,3,1), (0,3,2,1),
            (1,0,2,3), (1,0,3,2), (2,0,1,3), (3,0,1,2), (2,0,3,1), (3,0,2,1),
            (1,2,0,3), (1,3,0,2), (2,1,0,3), (3,1,0,2), (2,3,0,1), (3,2,0,1),
            (1,2,3,0), (1,3,2,0), (2,1,3,0), (3,1,2,0), (2,3,1,0), (3,2,1,0)
        ]
        current_order = orders[order]

        ordered_substructs = [None] * 4
        for i, type_idx in enumerate(current_order):
            ordered_substructs[type_idx] = substructs[i]

        sub0 = ordered_substructs[0]
        sub1 = ordered_substructs[1]
        sub2 = ordered_substructs[2]
        sub3 = ordered_substructs[3]

        species = struct.unpack("<H", sub0[0:2])[0]
        held_item = struct.unpack("<H", sub0[2:4])[0]
        experience = struct.unpack("<I", sub0[4:8])[0]

        moves = []
        for j in range(4):
            move_id = struct.unpack("<H", sub1[j*2:(j+1)*2])[0]
            moves.append(move_id)
        pp = [b for b in sub1[8:12]]

        hp_ev = sub2[0]
        atk_ev = sub2[1]
        def_ev = sub2[2]
        speed_ev = sub2[3]
        spatk_ev = sub2[4]
        spdef_ev = sub2[5]

        iv_word = struct.unpack("<I", sub3[4:8])[0]
        hp_iv = iv_word & 0x1F
        atk_iv = (iv_word >> 5) & 0x1F
        def_iv = (iv_word >> 10) & 0x1F
        speed_iv = (iv_word >> 15) & 0x1F
        spatk_iv = (iv_word >> 20) & 0x1F
        spdef_iv = (iv_word >> 25) & 0x1F
        is_egg = (iv_word >> 30) & 1
        ability_num = (iv_word >> 31) & 1

        status = self._read_u32(addr + OFF_STATUS)
        level = self._read_u8(addr + OFF_LEVEL)
        current_hp = self._read_u16(addr + OFF_HP)
        max_hp = self._read_u16(addr + OFF_MAX_HP)
        attack = self._read_u16(addr + OFF_ATK)
        defense = self._read_u16(addr + OFF_DEF)
        speed = self._read_u16(addr + OFF_SPEED)
        sp_attack = self._read_u16(addr + OFF_SPATK)
        sp_defense = self._read_u16(addr + OFF_SPDEF)

        return {
            "nickname": nickname,
            "species": self.species_names.get(species, f"Species {species}"),
            "species_id": species,
            "level": level,
            "hp": current_hp,
            "max_hp": max_hp,
            "status": status,
            "held_item": self.item_names.get(held_item, f"Item {held_item}"),
            "held_item_id": held_item,
            "moves": moves,
            "pp": pp,
            "stats": {
                "attack": attack,
                "defense": defense,
                "speed": speed,
                "sp_attack": sp_attack,
                "sp_defense": sp_defense
            },
            "ivs": {
                "hp": hp_iv,
                "attack": atk_iv,
                "defense": def_iv,
                "speed": speed_iv,
                "sp_attack": spatk_iv,
                "sp_defense": spdef_iv
            },
            "evs": {
                "hp": hp_ev,
                "attack": atk_ev,
                "defense": def_ev,
                "speed": speed_ev,
                "sp_attack": spatk_ev,
                "sp_defense": spdef_ev
            },
            "experience": experience,
            "is_egg": bool(is_egg),
            "ability_num": ability_num,
            "personality": personality,
            "ot_id": ot_id
        }

    def _load_map_names(self):
        map_names = {}
        try:
            with open("include/constants/map_groups.h", "r") as f:
                content = f.read()
                matches = re.findall(r'(MAP_\w+)\s*=\s*\((\d+)\s*\|\s*\((\d+)\s*<<\s*8\)\)', content)
                for name, num, group in matches:
                    map_names[(int(group), int(num))] = name
        except FileNotFoundError:
            pass
        return map_names

    def _load_item_names(self):
        item_names = {}
        try:
            with open("include/constants/items.h", "r") as f:
                for line in f:
                    match = re.search(r'#define\s+(ITEM_\w+)\s+(\d+)', line)
                    if match:
                        name = match.group(1)
                        item_id = int(match.group(2))
                        item_names[item_id] = name
        except FileNotFoundError:
            pass
        return item_names

    def _load_species_names(self):
        species_names = {}
        try:
            with open("include/constants/species.h", "r") as f:
                for line in f:
                    match = re.search(r'#define\s+(SPECIES_\w+)\s+(\d+)', line)
                    if match:
                        name = match.group(1)
                        species_id = int(match.group(2))
                        species_names[species_id] = name
        except FileNotFoundError:
            pass
        return species_names

if __name__ == "__main__":
    emu = Emulator()
    # emu.run_frame()
    print(json.dumps(emu.get_state(include_screenshot=False), indent=2, default=str))

