import json
import sys

# 尝试导入 mGBA 绑定
try:
    import mgba.core
    import mgba.image
    import mgba.log
except ImportError:
    print("Error: mGBA python bindings not found.")
    print("Please ensure you have mGBA installed with Python support.")
    sys.exit(1)

# 内存地址常量 (从 pokeemerald.map 获取)
# 注意：这些地址是编译后的 ROM 特有的，如果重新编译可能会改变
ADDR_G_SAVE_BLOCK_1_PTR = 0x03005D8C
ADDR_G_SAVE_BLOCK_2_PTR = 0x03005D90

# 物品 ID 到名称的映射 (部分示例，完整列表在 include/constants/items.h)
ITEM_NAMES = {
    0: "None",
    1: "Master Ball",
    2: "Ultra Ball",
    3: "Great Ball",
    4: "Poke Ball",
    # ... 更多物品
}

# 口袋配置 (基于 src/load_save.c 和 include/global.h)
POCKETS = {
    "Items":     {"offset": 0x560, "capacity": 30},
    "KeyItems":  {"offset": 0x5D8, "capacity": 30},
    "PokeBalls": {"offset": 0x650, "capacity": 16},
    "TMsHMs":    {"offset": 0x690, "capacity": 64},
    "Berries":   {"offset": 0x790, "capacity": 46},
}

def read_u32(core, addr):
    """读取 4 字节无符号整数"""
    try:
        return core.bus.read32(addr)
    except AttributeError:
        print("Error: core.bus.read32 not found. Check mGBA Python API documentation.")
        return 0

def read_u16(core, addr):
    """读取 2 字节无符号整数"""
    try:
        return core.bus.read16(addr)
    except AttributeError:
        print("Error: core.bus.read16 not found. Check mGBA Python API documentation.")
        return 0

def get_bag_items(core):
    """
    获取背包内所有物品
    :param core: mGBA 核心对象
    """
    # 1. 获取 SaveBlock 的实际动态地址
    # 指针本身存储在内存中，我们需要读取指针变量的值
    sb1_addr = read_u32(core, ADDR_G_SAVE_BLOCK_1_PTR)
    sb2_addr = read_u32(core, ADDR_G_SAVE_BLOCK_2_PTR)

    if sb1_addr == 0 or sb2_addr == 0:
        print("SaveBlock pointers are null. Game might not be loaded yet.")
        return {}

    # 2. 获取加密密钥
    # encryptionKey 位于 SaveBlock2 + 0xAC
    encryption_key = read_u32(core, sb2_addr + 0xAC)

    bag_data = {}

    # 3. 遍历所有口袋
    for pocket_name, config in POCKETS.items():
        items = []
        # 注意：SaveBlock1 的地址是动态的，所以我们需要加上偏移量
        # 但是，SaveBlock1 的指针指向的是 SaveBlock1 结构体的起始位置
        # 所以我们需要加上口袋在 SaveBlock1 中的偏移量
        base_addr = sb1_addr + config["offset"]
        capacity = config["capacity"]

        for i in range(capacity):
            # 每个 ItemSlot 占 4 字节
            slot_addr = base_addr + (i * 4)

            # 读取 Item ID (前 2 字节)
            item_id = read_u16(core, slot_addr)

            # 读取加密的数量 (后 2 字节)
            encrypted_quantity = read_u16(core, slot_addr + 2)

            if item_id != 0: # 忽略空槽位
                # 4. 解密数量
                # 算法: 真实数量 = 加密数量 ^ 密钥
                # 注意：密钥是 32 位的，这里只取低 16 位参与异或（因为 quantity 是 u16）
                quantity = encrypted_quantity ^ (encryption_key & 0xFFFF)

                items.append({
                    "id": item_id,
                    "name": ITEM_NAMES.get(item_id, f"Unknown Item {item_id}"),
                    "quantity": quantity,
                    "slot_index": i
                })

        bag_data[pocket_name] = items

    return bag_data

def main():
    # 初始化 mGBA
    core = mgba.core.loadPath("pokeemerald.gba")
    if not core:
        print("Failed to load ROM")
        return

    core.reset()

    # 模拟运行几帧以确保内存初始化
    # 注意：这可能需要更长时间才能进入游戏并加载存档
    # 在实际使用中，你可能需要加载一个存档状态 (save state)
    for _ in range(600): # 运行 10 秒 (60fps)
        core.runFrame()

    # 获取背包数据
    bag = get_bag_items(core)

    print(json.dumps(bag, indent=2))

if __name__ == "__main__":
    main()

