import time
import json
import numpy as np
import matplotlib.pyplot as plt
import socket
import struct
import torch
import sys

# -----------------------------
# CONSTANTS
# -----------------------------
FOG_ADDR = 0x0202079A
DAY_ADDR = 0x03004420
UNIT_P1 = 0x02022690
UNIT_P2 = 0x02022990

TERRAIN_COLORS = {
    0: (0, 0, 0),           # Empty / default
    1: (255, 255, 255),     # Plains / white
    2: (120, 120, 120),     # Low mountains / dark gray
    3: (80, 80, 80),        # Nothing / darker gray
    4: (181, 222, 138),     # Plains - Light green
    5: (34, 139, 34),       # Forest - green
    6: (30, 144, 255),      # River - blue
    7: (0, 191, 255),       # Nothing/ cyan
    8: (169, 169, 169),     # Road / gray
    9: (255, 255, 153),     # Neutral city / light yellow
    10: (255, 100, 100),    # Player 1 city / red-orange
    11: (100, 100, 255),    # Player 2 city / blue
    12: (200, 50, 50),      # Player 1 HQ / dark red
    13: (50, 50, 200),      # Player 2 HQ / dark blue
    14: (0, 255, 255),      # Alternate neutral city / blue + yellow/green?
    15: (255, 0, 255),      # Special / cosmetic / magenta
    16: (255, 150, 150),    # P1 factory / light red
    17: (150, 150, 255),    # P2 factory / light blue
}

ARMY_COLORS = {
    0: (255, 0, 0),
    1: (0, 0, 255),
    2: (0, 255, 0),
    3: (255, 255, 0),
}

UNITS_COST = [
    1000,
    3000,
    4000,
    7000,
    16000,
    22000,
    5000,
    6000,
    15000,
    8000,
    12000,
    20000,
    22000,
    9000,
    5000,
    28000,
    18000,
    12000,
    20000
]

WIN_FRAME = 0xFFFFFFFF  # Special frame indicating a win message
EXIT_FRAME = 0x11111111  # Special frame indicating an exit message
MSG_ACTION = 0
MSG_RESET = 1

with open("./data/data.json", "r") as file:
    INFO_DICT = json.load(file)

# -----------------------------
# UTILS
# -----------------------------
def read_bytes_from_memory(addr, size, ewram, iwram, retries=3, delay=0.001):
    """
    Read `size` bytes safely from EWRAM/IWRAM tensors.
    Retry if the read fails.

    Args:
        addr (int): starting memory address
        size (int): number of bytes to read
        ewram (torch.Tensor): EWRAM snapshot tensor (0x40000 bytes)
        iwram (torch.Tensor): IWRAM snapshot tensor (0x8000 bytes)
        retries (int): number of retries if read fails
        delay (float): delay between retries

    Returns:
        bytes or None if all retries fail
    """
    for _ in range(retries):
        try:
            out = bytearray()
            current_addr = addr
            remaining = size

            while remaining > 0:
                if 0x02000000 <= current_addr < 0x02000000 + 0x40000:
                    # EWRAM
                    offset = current_addr - 0x02000000
                    available = 0x40000 - offset
                    chunk_size = min(remaining, available)
                    chunk = ewram[offset:offset + chunk_size]

                elif 0x03000000 <= current_addr < 0x03000000 + 0x8000:
                    # IWRAM
                    offset = current_addr - 0x03000000
                    available = 0x8000 - offset
                    chunk_size = min(remaining, available)
                    chunk = iwram[offset:offset + chunk_size]

                else:
                    raise ValueError(f"Address out of range: {hex(current_addr)}")

                # Convert normalized tensor values back to bytes
                out.extend(chunk.byte().cpu().numpy().tobytes())

                current_addr += chunk_size
                remaining -= chunk_size

            return bytes(out)

        except Exception:
            time.sleep(delay)

    return bytes(0)

def safe_read(addr, size, ewram, iwram, retries=3):
    """Read integer safely from memory blocks."""
    for _ in range(retries):
        b = read_bytes_from_memory(addr, size, ewram, iwram)
        if b is not None and len(b) == size:
            return int.from_bytes(b, "little")
    return -1

def read_unit_from_info(ewram, iwram, address):
    """Read a unit field by field using memory blocks."""
    unit = {}
    unit['id'] = safe_read(address, 1, ewram, iwram, retries=3)
    unit["moved"] = safe_read(address + 1, 1, ewram, iwram, retries=3)
    unit['x'] = safe_read(address + 2, 1, ewram, iwram, retries=3)
    unit['y'] = safe_read(address + 3, 1, ewram, iwram, retries=3)
    hp_ammo = safe_read(address + 4, 1, ewram, iwram, retries=3)
    unit['hp'] = hp_ammo if hp_ammo < 128 else hp_ammo - 128
    unit['ammo'] = 0 if hp_ammo < 128 else 1
    unit['ammo'] += (safe_read(address + 5, 1, ewram, iwram, retries=3) % 8) * 2
    unit['fuel'] = safe_read(address + 6, 1, ewram, iwram, retries=3)
    return unit

# -----------------------------
# CLASS ENVIRONMENT
# -----------------------------
class AdvanceWarsEnv:
    """Class wrapper for Advance Wars using Retro with step/render/reset functions."""

    def __init__(self):
        # -----------------------------
        # Initialize Retro environment
        # -----------------------------
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn = None
        self.terrain = None
        self.reward_history = [0.0]
        self.ewram = torch.zeros(0x40000)
        self.iwram = torch.zeros(0x8000)
        self.info = {}
        self.old_info = None

        self.fog_mask = None

        self.units_p1 = []
        self.units_p1_count = {}
        self.old_units_p1 = []
        self.old_units_p1_count = {}

        self.units_p2 = []
        self.units_p2_count = {}
        self.old_units_p2 = []
        self.old_units_p2_count = {}

        self.turn_steps = 0.0
        self.point = 0.0

        # -----------------------------
        # Initialize Matplotlib figure
        # -----------------------------
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 5))
        self.axes = self.axes.flatten()
        self.axes[0].set_title("Terrain")
        self.axes[1].set_title("Fog")
        self.axes[2].set_title("Units")
        for ax in self.axes:
            ax.axis("off")

        self.axes[3].axis("on")

        # Initialize empty images for updating
        self.height, self.width = 20, 30
        self.terrain_img = self.axes[0].imshow(np.zeros((self.height, self.width, 3), dtype=np.uint8))
        self.fog_img = self.axes[1].imshow(np.zeros((self.height, self.width, 3), dtype=np.uint8))
        self.units_img = self.axes[2].imshow(np.zeros((self.height, self.width, 3), dtype=np.uint8))

        self.axes[3].set_title("Reward over Time")
        self.axes[3].set_xlabel("Step")
        self.axes[3].set_ylabel("Reward")
        self.axes[3].grid(True)
        self.axes[3].tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.axes[3].set_xlim(0, 100)  # initial x-axis
        self.axes[3].set_ylim(-10, 10)  # initial y-axis, adjust to expected reward range
        self.reward_line, = self.axes[3].plot([], [], lw=2, color='orange')

        self.fig.canvas.draw_idle()
        plt.pause(0.01)

    def reset(self):
        self.send_message(b"", MSG_RESET)
        time.sleep(1)  # Sleep a bit for at least 1 frame to pass
        action = torch.zeros(10, dtype=torch.float32)
        action = action.numpy().tobytes()
        if len(self.reward_history) > 1024:
            self.reward_history = [0.0]
            self.turn_steps = 0.0
            self.point = 0.0

        self.info = {}
        self.old_info = None

        self.units_p1 = []
        self.units_p1_count = {}
        self.old_units_p1 = []
        self.old_units_p1_count = {}

        self.units_p2 = []
        self.units_p2_count = {}
        self.old_units_p2 = []
        self.old_units_p2_count = {}

        self.send_and_receive(action)
        self.update_info()
        self.render()

        map_grid = self.encode_map(self.terrain, self.fog_mask)
        units_grid = self.make_units_grid(self.units_p1, self.units_p2)

        extras = np.array([self.info.get("funds",0), self.info.get("cursor_x", 0), self.info.get("cursor_y", 0), self.info.get("cities", 0),
                           self.info.get("factories", 0), self.info.get("nb_units", 0), self.info.get("nb_units_lost", 0), self.info.get("vs_nb_units", 0),
                           self.info.get("vs_nb_units_lost", 0), self.info.get("map_width", 0), self.info.get("map_height", 0), self.info.get("day", 0)]
                           , dtype=np.float32)

        obs = {
            "map": map_grid,
            "units": units_grid,
            "extras": extras
        }
        return obs, self.info

    def read_map(self, ewram, iwram):
        # -----------------------------
        # Read tiles and fog
        # -----------------------------
        map_height, map_width = self.info.get("map_height", 0), self.info.get("map_width", 0)
        tiles = np.zeros((self.height, self.width), dtype=np.uint8)
        addr = INFO_DICT["map"]["address"]
        for y in range(min(map_height, self.height)):
            for x in range(min(map_width, self.width)):
                b = read_bytes_from_memory(addr, 2, ewram, iwram)
                if b:
                    # 1st byte
                    # Bit 0: ???
                    # Bit 1: ???
                    # Bit 2: Forest, allied cities, enemies cities, bridge
                    # Bit 3: River, enemies cities, enemies capital
                    # Bit 4: River + bridge
                    # Bit 5: Shadow + mountains ???
                    # Bit 6: Cities, capitals, turning river, roads and some point i don't know
                    # Bit 7: Cities, capital, forest and other types of road

                    # 2nd byte
                    # Bit 0: cities, capital + weird ass road between 3 cities
                    # Bit 1: Nothing?
                    # Bit 2: Nothing?
                    # Bit 3: Fog?
                    # Bit 4: Nothing?
                    # Bit 5: Nothing?
                    # Bit 6: Nothing?
                    # Bit 7: Nothing?
                    cities = (b[1] >> 0) & 1
                    bit_0 = (b[0] >> 0) & 1
                    bit_1 = (b[0] >> 1) & 1
                    bit_2 = (b[0] >> 2) & 1
                    bit_3 = (b[0] >> 3) & 1
                    bit_4 = (b[0] >> 4) & 1
                    bit_5 = (b[0] >> 5) & 1
                    bit_6 = (b[0] >> 6) & 1
                    bit_7 = (b[0] >> 7) & 1
                    if bit_3 and not bit_2 and  bit_1 and not bit_0 and cities:
                        tiles[y, x] = 13
                    elif not bit_3 and bit_2 and not bit_1 and bit_0 and cities:
                        tiles[y, x] = 12
                    elif bit_3 and bit_2 and not bit_1 and not bit_0 and cities:
                        tiles[y, x] = 11
                    elif not bit_3 and bit_2 and bit_1 and bit_0 and cities:
                        tiles[y, x] = 10
                    elif not bit_7 and not bit_6 and not bit_3 and not bit_2 and not bit_1 and bit_0 and not cities:
                        tiles[y, x] = 4
                    elif not bit_3 and bit_2 and bit_1 and not bit_0 and cities:
                        tiles[y, x] = 16
                    elif not bit_2 and bit_1 and bit_0 and cities:
                        tiles[y, x] = 17
                    elif (bit_3 and not bit_2 and bit_1 and bit_0) or (bit_3 and not bit_2 and not bit_1 and bit_0) or (bit_3 and not bit_2 and not bit_1 and not bit_0):
                        tiles[y, x] = 6
                    elif bit_7 and bit_6 and not bit_5 and not bit_4 and not bit_3 and not bit_2 and bit_1 and not bit_0 and cities:
                        tiles[y, x] = 9
                    elif not bit_4 and not bit_3 and bit_2 and bit_1 and not cities: # Bit 0 doesn't impact forest surely just shadow here
                        tiles[y, x] = 5
                    elif not bit_7 and not bit_6 and not bit_4 and not bit_3 and not bit_2 and not bit_0:
                        tiles[y, x] = 2
                    elif not bit_4 and not bit_3 and not bit_2 and bit_1 and bit_0:
                        tiles[y, x] = 4
                    elif bit_4 and not bit_3 and bit_2 and bit_1 and not bit_0 and not cities:
                        tiles[y, x] = 4
                    elif not bit_4 and not bit_3 and not bit_2 and not bit_1 and bit_0 and not cities:
                        tiles[y, x] = 4
                    elif not bit_4 and not bit_3 and bit_2 and not bit_1 and not cities:
                        tiles[y, x] = 4
                    elif not bit_7 and bit_6 and not bit_5 and not bit_4 and not bit_3 and not bit_2 and not bit_0 and not cities:
                        tiles[y, x] = 4
                    elif bit_7 and not bit_6 and not bit_4 and not bit_3 and not cities:
                        tiles[y, x] = 4
                    elif not bit_7 and bit_6 and bit_5 and not bit_4 and not bit_3 and not cities:
                        tiles[y, x] = 4
                    elif not bit_7 and not bit_6 and not bit_5 and not bit_4 and not bit_3 and not bit_2 and not bit_1 and bit_0 and cities:
                        tiles[y, x] = 4
                    elif not bit_7 and not bit_6 and not bit_5 and bit_4 and not bit_3 and bit_2 and not bit_1 and not bit_0:
                        tiles[y, x] = 4
                    elif bit_7 and bit_6 and bit_5 and not bit_4 and not bit_3 and not bit_2 and not bit_1 and bit_0:
                        tiles[y, x] = 3
                    elif not bit_7 and not bit_6 and bit_5 and not bit_4 and bit_3 and not bit_2 and bit_1 and not bit_0:
                        tiles[y, x] = 6
                    elif not bit_7 and not bit_6 and bit_5 and not bit_4 and bit_3 and bit_2 and not bit_1 and not bit_0:
                        tiles[y, x] = 6
                    elif not bit_7 and not bit_5 and not bit_4 and bit_3 and not bit_2 and bit_1 and not bit_0:
                        tiles[y, x] = 6
                    elif not bit_7 and bit_6 and not bit_5 and not bit_4 and bit_3 and bit_2 and not bit_1 and not bit_0:
                        tiles[y, x] = 6
                    elif not bit_7 and not bit_6 and not bit_5 and not bit_4 and bit_3 and bit_2 and not bit_1:
                        tiles[y, x] = 6
                    elif bit_7 and bit_6 and not bit_5 and not bit_4 and bit_3 and bit_2 and not bit_1 and bit_0:
                        tiles[y, x] = 6
                    elif bit_7 and bit_6 and not bit_5 and bit_4 and not bit_3 and bit_2 and bit_1 and bit_0:
                        tiles[y, x] = 6
                    elif not bit_7 and not bit_6 and not bit_5 and bit_4 and bit_3 and bit_2 and not bit_1 and bit_0:
                        tiles[y, x] = 6
                    elif not bit_7 and not bit_6 and not bit_5 and bit_4 and not bit_3 and not bit_2 and bit_1 and bit_0:
                        tiles[y, x] = 6
                    elif bit_7 and bit_6 and not bit_5 and not bit_4 and bit_3 and bit_2 and bit_1 and not bit_0:
                        tiles[y, x] = 6
                    elif bit_7 and bit_6 and not bit_5 and bit_4 and not bit_3 and not bit_2 and not bit_1 and not bit_0:
                        tiles[y, x] = 6
                    elif bit_7 and not bit_6 and bit_5 and bit_4 and not bit_3 and not bit_2 and bit_1 and bit_0:
                        tiles[y, x] = 6
                    elif not bit_7 and not bit_6 and not bit_5 and bit_4 and not bit_3 and not bit_2 and bit_1 and not bit_0:
                        tiles[y, x] = 6
                    else:
                        tiles[y, x] = (b[0] >> 4) & 0b1111
                    # tiles[y, x] = not bit_2 and bit_1 and bit_0 and cities
                addr += 2

        return tiles

    def update_info(self):
        self.old_info = self.info
        for k, v in INFO_DICT.items():
            address = v["address"]
            size = v["size"]
            # print(k, address)
            bytes = read_bytes_from_memory(address, size, self.ewram, self.iwram)
            self.info[k] = int.from_bytes(bytes, byteorder="little") if bytes else 0

    def extract_state(self):
        """Extract map tiles, fog mask, and units from memory blocks."""

        self.terrain = self.read_map(self.ewram, self.iwram)
        map_height, map_width = self.info.get("map_height", 0), self.info.get("map_width", 0)

        fog_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        fog_addr = FOG_ADDR
        for y in range(min(map_height, self.height)):
            for x in range(min(map_width, self.width)):
                f = read_bytes_from_memory(fog_addr,1, self.ewram, self.iwram)
                if f:
                    fog_mask[y, x] =  1 if f[0] > 0 else 0
                fog_addr += 1


        # -----------------------------
        # Read units for player 1
        # -----------------------------
        units_p1 = []
        if self.info.get("turn", 0) != 3:
            self.old_units_p1 = self.units_p1
            self.old_units_p1_count = self.units_p1_count
        self.units_p1_count = {k: 0 for k in range(1,20)}  # Regroup units by ID
        unit = read_unit_from_info(self.ewram, self.iwram, UNIT_P1)
        unit['ai'] = 0
        index = 0
        while UNIT_P1 + index * 12 < UNIT_P2:  # 12 -> Size of unit in memory
            if unit["id"] > 0:
                units_p1.append({**unit})
                self.units_p1_count[unit["id"]] += 1
            index += 1
            unit = read_unit_from_info(self.ewram, self.iwram, UNIT_P1 + index * 12)
            unit['ai'] = 0


        # -----------------------------
        # Read units for player 2
        # -----------------------------
        units_p2 = []
        if self.info.get("turn", 0) != 3:
            self.old_units_p2 = self.units_p2
            self.old_units_p2_count = self.units_p2_count
        self.units_p2_count = {k: 0 for k in range(1,20)}  # Regroup units by ID
        unit = read_unit_from_info(self.ewram, self.iwram, UNIT_P2)
        unit['ai'] = 1
        index = 0
        while UNIT_P2 + index * 12 < UNIT_P2 + 512:  # 512 -> Total bytes for a player's units
            if unit["id"] > 0:
                units_p2.append({**unit})
                self.units_p2_count[unit["id"]] += 1
            index += 1
            unit = read_unit_from_info(self.ewram, self.iwram, UNIT_P2 + index * 12)
            unit['ai'] = 1

        self.fog_mask = fog_mask
        self.units_p1 = units_p1
        self.units_p2 = units_p2

    def encode_map(self, terrain, fog):
        # terrain, fog: numpy arrays (H, W)
        encoded = (terrain & 0b11111) | ((fog.astype(np.uint8) & 0b1) << 5)
        return encoded.astype(np.uint8)

    def render(self):
        """Render the map, fog, units, and reward using matplotlib."""
        self.extract_state()
        fog_mask, units_p1, units_p2 = self.fog_mask, self.units_p1, self.units_p2
        tiles = self.terrain

        terrain = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        fog = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        units_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for y in range(self.height):
            for x in range(self.width):
                terrain[y, x] = TERRAIN_COLORS.get(tiles[y, x], (150, 150, 150))
                fog[y, x] = (255, 255, 255) if fog_mask[y, x] else (0, 0, 0)

        for u in units_p1 + units_p2:
            px, py = u["x"], u["y"]
            if 0 <= px < self.width and 0 <= py < self.height:
                units_map[py, px] = ARMY_COLORS.get(u["ai"], (255, 255, 255)) if fog_mask[py, px] else (0, 0, 0)

        self.terrain_img.set_data(terrain)
        self.fog_img.set_data(fog)
        self.units_img.set_data(units_map)

        # -----------------------------
        # Update reward plot
        # -----------------------------
        self.reward_line.set_data(range(len(self.reward_history)), self.reward_history)

        # Update axes limits dynamically
        self.axes[3].set_xlim(0, max(100, len(self.reward_history)))
        current_min = min(self.reward_history) - 1
        current_max = max(self.reward_history) + 1
        self.axes[3].set_ylim(current_min, current_max)

        self.axes[3].relim()
        self.axes[3].autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def make_units_grid(self, units_p1, units_p2):

        grid = np.zeros((self.height, self.width), dtype=np.int32)
        for u1 in units_p1:
            if u1['hp'] <= 0:
                continue
            hp_bucket = min(u1['hp'] // 3, 3)
            code = (u1['ai'] << 6) | (u1['id'] << 2) | hp_bucket
            grid[u1['y'], u1['x']] = code
        for u2 in units_p2:
            if u2['hp'] <= 0:
                continue
            hp_bucket = min(u2['hp'] // 3, 3)
            code = (u2['ai'] << 6) | (u2['id'] << 2) | hp_bucket
            grid[u2['y'], u2['x']] = code
        return grid

    def calculatereward(self):
        reward = 0.0

        funds = self.info.get("funds", 0)
        old_funds = self.old_info.get("funds", 0)
        reward += np.abs(funds - old_funds) / 10000.0 + 0.0

        cities = self.info.get("cities", 0)
        old_cities = self.old_info.get("cities", 0)
        reward += np.abs(cities - old_cities) + 0.0

        factories = self.info.get("factories", 0)
        old_factories = self.old_info.get("factories", 0)
        reward += np.abs(factories - old_factories) + 0.0

        day = self.info.get("day", 0)
        reward -= day * 0.5

        for unit in self.units_p1:
            if unit["moved"] == 6:  # Moving
                reward += 0.1
            if unit["moved"] == 1:  # Moved
                reward += 1.0

        if self.old_units_p1_count:
            for k, v in self.units_p1_count.items():
                reward += (v - self.old_units_p1_count.get(k, 0)) * UNITS_COST[k - 1] / 10000.0

        if self.old_units_p2_count:
            for k, v in self.units_p2_count.items():
                reward += min(v - self.old_units_p2_count.get(k, 0), 0) * UNITS_COST[k - 1] / 10000.0

        reward -= self.turn_steps * 0.005

        reward += self.point
        self.point = 0.0

        return reward

    def check_win(self):
        if len(self.units_p1) == 0:
            return False

    def adjust_action(self, action):
        res = torch.zeros(10, dtype=torch.float32)
        if action >= 1:
            action += 1  # No R (position 1)
        if action >= 6:
            action += 2  # No start or select (position 6 and 7)
        res[action] = 1.0
        return res.numpy().tobytes()

    def send_message(self, payload, message_type=0):
        header = struct.pack(">II", len(payload), message_type)
        self.conn.sendall(header + payload)
        
    def send_and_receive(self, action):
        self.send_message(action, MSG_ACTION)
        # Receive 8-byte header
        header = recv_exact(self.conn, 8)
        if not header:
            return None

        length, frame = struct.unpack(">II", header)
        data = recv_exact(self.conn, length)
        if not data:
            return None

        if frame == WIN_FRAME:
            # This is a win message
            try:
                self.point = 50.0
                self.reset()
            except UnicodeDecodeError:
                return "[WIN] Unable to decode message"
        elif frame == EXIT_FRAME:
            # This is an exit message
            try:
                self.point = -50.0
                self.reset()
            except UnicodeDecodeError:
                return "[EXIT] Unable to decode message"
        else:
            # This is a memory snapshot
            mem = np.frombuffer(data, dtype=np.uint8)
            tensor = torch.from_numpy(mem)

            self.ewram = tensor[:0x40000]
            self.iwram = tensor[0x40000:]

    def step(self, action=None):
        """Step forward in the game using a given action or random action."""
        start = time.time()
        done = False
        if action is None:
            action = torch.zeros(10, dtype=torch.float32)
            pos = np.random.randint(10)
            while pos == 7 or pos == 6 or pos == 1: # NO start, select or R
                pos = np.random.randint(10)
            action[pos] = 1.0
            action = action.numpy().tobytes()
        else:
            action = self.adjust_action(action)
        self.send_and_receive(action)
        self.update_info()
        self.render()
        if self.info.get("turn", 0) != 3:
            self.turn_steps = 0.0
        else:
            self.turn_steps += 1.0
        while self.info.get("turn", -1) != 3:
            action_start = torch.zeros(10, dtype=torch.float32)
            action_start = action_start.numpy().tobytes()
            self.send_and_receive(action_start)
            self.update_info()
            self.render()
            if self.info.get("defeat", 0) != 0:  
                self.send_message(b"", MSG_RESET)
                time.sleep(1) 
                self.point -= 50.0

        map_grid = self.encode_map(self.terrain, self.fog_mask)
        units_grid = self.make_units_grid(self.units_p1, self.units_p2)

        extras = np.array([self.info.get("funds",0), self.info.get("cursor_x", 0), self.info.get("cursor_y", 0), self.info.get("cities", 0),
                            self.info.get("factories", 0), self.info.get("nb_units", 0), self.info.get("nb_units_lost", 0), self.info.get("vs_nb_units", 0),
                            self.info.get("vs_nb_units_lost", 0), self.info.get("map_width", 0), self.info.get("map_height", 0), self.info.get("day", 0)]
                            , dtype=np.float32)

        obs = {
            "map": map_grid,
            "units": units_grid,
            "extras": extras
        }

        reward = self.calculatereward()

        self.reward_history.append(reward)

        frame_duration = 1.0 / 25.0
        elapsed = time.time() - start
        if elapsed < frame_duration:
            time.sleep(frame_duration - elapsed)

        return obs, reward, done, self.info

    def close(self):
        """Close the environment and matplotlib figure."""
        plt.close(self.fig)
        self.socket.close()
        sys.exit(1)

# -----------------------------
# USAGE EXAMPLE
# -----------------------------
HOST, PORT = "127.0.0.1", 5000

def recv_exact(sock, size):
    buf = bytearray()

    while len(buf) < size:
        data = sock.recv(size - len(buf))

        if not data:
            raise ConnectionError(
                f"Connection closed: received {len(buf)}/{size} bytes"
            )

        buf.extend(data)

    return bytes(buf)

if __name__ == "__main__":
    env = AdvanceWarsEnv()

    with env.socket as s:
        try:
            s.bind((HOST, PORT))
            s.listen(1)
            print(f"Listening on {HOST}:{PORT}...")
            conn, addr = s.accept()
            print(f"Connected by {addr}")

            with conn:
                env.conn = conn
                while env:
                    env.step()
            env.close()
        except KeyboardInterrupt:
            s.close()
