import time
import os
import numpy as np
import matplotlib.pyplot as plt
from actionwrapper import AWActionWrapper
from pygba import PyGBA, PyGBAEnv

# -----------------------------
# CONSTANTS
# -----------------------------
MAP_DIM_ADDR = 0x0201E450
MAP_DATA_ADDR = 0x0201F882
FOG_ADDR = 0x0202079A
DAY_ADDR = 0x03004420
CURSOR_ADDR = 0x030033f0
UNIT_P1 = 0x02022690
UNIT_P2 = 0x02022990

YIELD = 0x020232F0

NB_UNITS_P1 = 0x020232FA
NB_UNITS_LOST_P1 = 0x020232FB

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

LOSSES_NAMES = [
    "nb_apc_lost",
    "nb_artlry_lost",
    "nb_infantry_lost",
    "nb_md_tank_lost",
    "nb_mech_lost",
    "nb_missiles_lost",
    "nb_recon_lost",
    "nb_rockets_lost",
    "nb_sub_lost",
    "nb_t_cptr_lost",
    "nb_tank_lost",
]

BUILDS_NAMES = [
    "nb_apc",
    "nb_artlry",
    "nb_infantry",
    "nb_md_tank",
    "nb_mech",
    "nb_missiles",
    "nb_recon",
    "nb_rockets",
    "nb_sub",
    "nb_t_cptr",
    "nb_tank",
]

VS_LOSSES_NAMES = [
    "vs_nb_apc_lost",
    "vs_nb_artlry_lost",
    "vs_nb_infantry_lost",
    "vs_nb_md_tank_lost",
    "vs_nb_mech_lost",
    "vs_nb_missiles_lost",
    "vs_nb_recon_lost",
    "vs_nb_rockets_lost",
    "vs_nb_sub_lost",
    "vs_nb_t_cptr_lost",
    "vs_nb_tank_lost",
]

VS_BUILDS_NAMES = [
    "vs_nb_apc",
    "vs_nb_artlry",
    "vs_nb_infantry",
    "vs_nb_md_tank",
    "vs_nb_mech",
    "vs_nb_missiles",
    "vs_nb_recon",
    "vs_nb_rockets",
    "vs_nb_sub",
    "vs_nb_t_cptr",
    "vs_nb_tank",
]

FACILITIES_NAMES = [
    "cities",
    "factories",
    "ports",
    "airports",
]

# -----------------------------
# UTILS
# -----------------------------
def read_bytes_from_blocks(addr, size, blocks, retries=3, delay=0.001):
    """
    Read `size` bytes safely from memory.blocks.
    Retry if the read fails.

    Args:
        addr (int): starting memory address
        size (int): number of bytes to read
        blocks (dict): memory blocks from the emulator
        retries (int): number of retries if read fails
        delay (float): delay (in seconds) between retries
    Returns:
        bytes or None if all retries fail
    """
    for attempt in range(retries):
        out = bytearray()
        remaining = size
        current_addr = addr
        while remaining > 0:
            found = False
            for block_base, data in blocks.items():
                block_size = len(data)
                if block_base <= current_addr < block_base + block_size:
                    start = current_addr - block_base
                    chunk_size = min(remaining, block_size - start)
                    out += data[start:start + chunk_size]
                    current_addr += chunk_size
                    remaining -= chunk_size
                    found = True
                    break
            if not found:
                # Failed to read this chunk
                break
        if len(out) == size:
            return bytes(out)
        else:
            time.sleep(delay)  # wait a bit before retrying
    # All retries failed
    return None

def safe_read(addr, size, blocks, retries=3):
    """Read integer safely from memory blocks."""
    for _ in range(retries):
        b = read_bytes_from_blocks(addr, size, blocks)
        if b is not None and len(b) == size:
            return int.from_bytes(b, "little")
    return None

def read_unit_from_info(blocks, address):
    """Read a unit field by field using memory blocks."""
    unit = {}
    unit['x'] = safe_read(address, 1, blocks, retries=3)
    unit['y'] = safe_read(address + 1, 1, blocks, retries=3)
    unit['id'] = safe_read(address - 2, 1, blocks, retries=3)
    unit["moved"] = safe_read(address - 1, 1, blocks, retries=3)
    hp_ammo = safe_read(address + 2, 1, blocks, retries=3)
    unit['hp'] = hp_ammo if hp_ammo < 128 else hp_ammo - 128
    unit['ammo'] = 0 if hp_ammo < 128 else 1
    unit['ammo'] += (safe_read(address + 3, 1, blocks, retries=3) % 8) * 2
    unit['fuel'] = safe_read(address + 4, 1, blocks, retries=3)
    return unit

# -----------------------------
# CLASS ENVIRONMENT
# -----------------------------
class AdvanceWarsEnv:
    """Class wrapper for Advance Wars using Retro with step/render/reset functions."""

    def __init__(self, rom_path="./roms/integration"):
        # -----------------------------
        # Initialize Retro environment
        # -----------------------------
        self.terrain = None
        self.reward_history = [0.0]

        rom_path = "/home/yro/code/rom/AW2.gba"
        self.gba = PyGBA.load(rom_path, autoload_save=True)
        self.env = PyGBAEnv(self.gba)
        self.wrapped = AWActionWrapper(self.env)
        _, info = self.reset()


        # -----------------------------
        # Initialize Matplotlib figure
        # -----------------------------
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 4, figsize=(20, 5))
        self.axes = self.axes.flatten()
        self.axes[0].set_title("Game")
        self.axes[1].set_title("Terrain")
        self.axes[2].set_title("Fog")
        self.axes[3].set_title("Units")
        for ax in self.axes:
            ax.axis("off")

        self.axes[4].axis("on")

        frame = self.env.render()
        self.game_img = self.axes[0].imshow(frame)

        # Initialize empty images for updating
        self.height, self.width = 20, 30
        self.terrain_img = self.axes[1].imshow(np.zeros((self.height, self.width, 3), dtype=np.uint8))
        self.fog_img = self.axes[2].imshow(np.zeros((self.height, self.width, 3), dtype=np.uint8))
        self.units_img = self.axes[3].imshow(np.zeros((self.height, self.width, 3), dtype=np.uint8))

        self.axes[4].set_title("Reward over Time")
        self.axes[4].set_xlabel("Step")
        self.axes[4].set_ylabel("Reward")
        self.axes[4].grid(True)
        self.axes[4].tick_params(axis='both', which='both', direction='in', top=True, right=True)
        self.axes[4].set_xlim(0, 100)  # initial x-axis
        self.axes[4].set_ylim(-10, 10)  # initial y-axis, adjust to expected reward range
        self.reward_line, = self.axes[4].plot([], [], lw=2, color='orange')

        self.funds = info.get("funds", 0)

        # Initialize dicts to calculate rewards
        self.losses = {}
        for loss_name in LOSSES_NAMES:
            self.losses[loss_name] = 0

        self.builds = {}
        for build_name in BUILDS_NAMES:
            self.builds[build_name] = info.get(build_name, 0)

        self.facilities = {}
        for facility_name in FACILITIES_NAMES:
            self.facilities[facility_name] = info.get(facility_name, 0)

        self.vs_losses = {}
        for vs_loss_name in VS_LOSSES_NAMES:
            self.vs_losses[vs_loss_name] = info.get(vs_loss_name, 0)

        self.units_p1 = None

        self.turn_steps = 0.0

    def reset(self):
        """Reset the Retro environment."""
        self.obs = self.env.reset()
        self.reward_history = [0.0]
        self.units_p1 = None
        _, _, done, _, info = self.wrapped.step(3) # Press UP -> no impact on game
        fog, units_p1, units_p2 = self.extract_state()
        terrain = self.terrain

        map_grid = self.encode_map(terrain, fog)
        units_grid = self.make_units_grid(units_p1, units_p2)

        extras = np.array([info.get("funds",0)], dtype=np.float32)

        obs = {
            "map": map_grid,
            "units": units_grid,
            "extras": extras
        }
        self.env.reset()
        return obs, info

    def read_map(self, height, width, blocks):
        # -----------------------------
        # Read tiles and fog
        # -----------------------------
        tiles = np.zeros((20, 30), dtype=np.uint8)
        addr = MAP_DATA_ADDR
        for y in range(height):
            for x in range(width):
                b = read_bytes_from_blocks(addr, 2, blocks)
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
                    elif not bit_3 and bit_2 and bit_1 and not bit_0:
                        tiles[y, x] = 16
                    elif bit_3 and bit_2 and bit_1 and not bit_0 and cities:
                        tiles[y, x] = 17
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
                    # tiles[y, x] = bit_4
                addr += 2

        return tiles


    def extract_state(self):
        """Extract map tiles, fog mask, and units from memory blocks."""
        blocks = self.env.data.memory.blocks

        # -----------------------------
        # Read map dimensions
        # -----------------------------
        dim_bytes = read_bytes_from_blocks(MAP_DIM_ADDR, 4, blocks)
        width = int.from_bytes(dim_bytes[0:2], "little")
        height = int.from_bytes(dim_bytes[2:4], "little")

        self.terrain = self.read_map(height, width, blocks)

        fog_mask = np.zeros((20, 30), dtype=np.uint8)
        fog_addr = FOG_ADDR
        for y in range(height):
            for x in range(width):
                f = read_bytes_from_blocks(fog_addr,1, blocks)
                if f:
                    fog_mask[y, x] =  1 if f[0] > 0 else 0
                fog_addr += 1


        # -----------------------------
        # Read units for player 1
        # -----------------------------
        units_p1 = []
        unit = read_unit_from_info(blocks, UNIT_P1)
        unit['ai'] = 0
        index = 0
        while unit['id'] != 0 and UNIT_P1 + index * 0xc < UNIT_P2:
            units_p1.append({**unit})
            index += 1
            unit = read_unit_from_info(blocks, UNIT_P1 + index * 0xc)
            unit['ai'] = 0

        if self.units_p1 is None:
            self.units_p1 = units_p1

        # -----------------------------
        # Read units for player 2
        # -----------------------------
        units_p2 = []
        unit = read_unit_from_info(blocks, UNIT_P2)
        unit['ai'] = 1
        index = 0
        while unit['id'] != 0:
            units_p2.append({**unit})
            index += 1
            unit = read_unit_from_info(blocks, UNIT_P2 + index * 0xc)
            unit['ai'] = 1


        return fog_mask, units_p1, units_p2

    def encode_map(self, terrain, fog):
        # terrain, fog: numpy arrays (H, W)
        encoded = (terrain & 0b11111) | ((fog.astype(np.uint8) & 0b1) << 5)
        return encoded.astype(np.uint8)

    def render(self):
        """Render the map, fog, units, and reward using matplotlib."""
        fog_mask, units_p1, units_p2 = self.extract_state()
        tiles = self.terrain

        # Update terrain, fog, units (same as before)
        height, width = tiles.shape
        terrain = np.zeros((20, 30, 3), dtype=np.uint8)
        fog = np.zeros((20, 30, 3), dtype=np.uint8)
        units_map = np.zeros((20, 30, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                terrain[y, x] = TERRAIN_COLORS.get(tiles[y, x], (150, 150, 150))
                fog[y, x] = (255, 255, 255) if fog_mask[y, x] else (0, 0, 0)

        for u in units_p1 + units_p2:
            px, py = u["x"], u["y"]
            if 0 <= px < width and 0 <= py < height:
                units_map[py, px] = ARMY_COLORS.get(u["ai"], (255, 255, 255))

        frame = self.env.render()

        self.game_img.set_data(frame)
        self.terrain_img.set_data(terrain)
        self.fog_img.set_data(fog)
        self.units_img.set_data(units_map)

        # -----------------------------
        # Update reward plot
        # -----------------------------
        self.reward_line.set_data(range(len(self.reward_history)), self.reward_history)

        # Update axes limits dynamically
        self.axes[4].set_xlim(0, max(100, len(self.reward_history)))
        current_min = min(self.reward_history) - 1
        current_max = max(self.reward_history) + 1
        self.axes[4].set_ylim(current_min, current_max)

        self.axes[4].relim()
        self.axes[4].autoscale_view()

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def make_units_grid(self, units_p1, units_p2):
        grid = np.zeros((20, 30), dtype=np.int32)
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

    def calculatereward(self, info, units_p1):
        reward = 0.0

        # funds = info.get("funds", 0)
        # reward += np.abs(self.funds - funds) / 1000.0

        day = info.get("day", 0)
        reward -= day * 0.5

        for loss_name in LOSSES_NAMES:
            loss = info.get(loss_name, 0)
            reward -= (loss - self.losses[loss_name]) * 2.0
            self.losses[loss_name] = loss

        for build_name in BUILDS_NAMES:
            build = info.get(build_name, 0)
            reward += (build - self.builds[build_name]) * 3.0
            self.builds[build_name] = build

        for facility_name in FACILITIES_NAMES:
            facility = info.get(facility_name, 0)
            reward += (facility - self.facilities[facility_name]) * 2.0
            self.facilities[facility_name] = facility

        for vs_loss_name in VS_LOSSES_NAMES:
            vs_loss = info.get(vs_loss_name, 0)
            reward += (vs_loss - self.vs_losses[vs_loss_name]) * 5.0
            self.vs_losses[vs_loss_name] = vs_loss

        for idx, unit in enumerate(units_p1):
            if idx > len(self.units_p1):
                self.units_p1.append(unit)
            elif unit["moved"] != self.units_p1[idx]["moved"]:
                if unit["moved"] == 1:
                    reward += 1.0
                elif unit["moved"] == 6:
                    reward += 0.05
                self.units_p1[idx] = unit

        reward -= self.turn_steps * 0.005

        if info.get("win", 0) == 1:
            reward += 100.0

        return reward

    def check_win(self):
        if len(self.units_p1) == 0:
            return False


    def step(self, action=None):
        """Step forward in the game using a given action or random action."""
        start = time.time()
        done = False
        if action is None:
            action = self.wrapped.action_space.sample()
        _, _, _, _, info = self.wrapped.step(action)
        if info.get("turn", 0) != 1:
            self.turn_steps = 0.0
        else:
            self.turn_steps += 1.0

        while info.get("turn", -1) != 1:
            for _ in range(10):
                _, _, _, _, info = self.wrapped.step(3)  # Press UP -> no impact on game
            if info.get("finish", 0) == 5:
                done = True
            if info.get("turn", 0) == 1:
                action_start = np.zeros(12, dtype=np.int8)
                action_start[3] = 1
                for _ in range(5):
                    self.wrapped.step(action_start)

        if info.get("menu", 0) == 0 or info.get("win", 0) == 1:
            done = True

        fog, units_p1, units_p2 = self.extract_state()
        terrain = self.terrain

        map_grid = self.encode_map(terrain, fog)
        units_grid = self.make_units_grid(units_p1, units_p2)

        extras = np.array([info.get("funds",0)], dtype=np.float32)

        obs = {
            "map": map_grid,
            "units": units_grid,
            "extras": extras
        }

        reward = self.calculatereward(info, units_p1)

        self.reward_history.append(reward)

        frame_duration = 1.0 / 25.0
        elapsed = time.time() - start
        if elapsed <  elapsed:
            time.sleep(frame_duration - elapsed)

        return obs, reward, done, info

    def close(self):
        """Close the environment and matplotlib figure."""
        self.env.close()
        plt.close(self.fig)

# -----------------------------
# USAGE EXAMPLE
# -----------------------------
if __name__ == "__main__":
    env = AdvanceWarsEnv()
    print("Press SPACE + Enter to step, 'q' to quit.")

    while True:
        key = input("Step? [space/q]: ")
        if key.lower() == "q":
            break

        # Step with random action
        obs, reward, done, info = env.step()
        env.render()

        # Reset environment when done
        if done:
            print("Episode finished, resetting...")
            env.reset()

    env.close()
