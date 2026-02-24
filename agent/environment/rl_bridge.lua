-- rl_bridge.lua
-- Note: Do NOT require("socket") as it is provided globally by mGBA

local server = nil
local client = nil
local port = 9999
local buffer = ""

-- Addresses from pokeemerald.map
local ADDR_PLAYER_PARTY = 0x020244ec
local ADDR_PLAYER_PARTY_COUNT = 0x020244e9
local ADDR_PLAYER_AVATAR = 0x02037590
local ADDR_OBJECT_EVENTS = 0x02037350
local ADDR_SPECIES_INFO = 0x083203cc
local ADDR_BATTLE_MOVES = 0x0831c898

-- Key mapping
local KEY_BITS = {
    ["A"] = 1, ["B"] = 2, ["Select"] = 4, ["Start"] = 8,
    ["Right"] = 16, ["Left"] = 32, ["Up"] = 64, ["Down"] = 128,
    ["R"] = 256, ["L"] = 512
}

-- Helper to read bits
function read_bits(val, start, len)
    return (val >> start) & ((1 << len) - 1)
end

function get_player_position()
    local objectEventId = emu:read8(ADDR_PLAYER_AVATAR + 4)
    local objectEventAddr = ADDR_OBJECT_EVENTS + objectEventId * 36
    local x = emu:read16(objectEventAddr + 16)
    local y = emu:read16(objectEventAddr + 18)
    return x, y
end

function get_species_info(species_id)
    local addr = ADDR_SPECIES_INFO + species_id * 28
    return {
        baseHP = emu:read8(addr + 0),
        baseAttack = emu:read8(addr + 1),
        baseDefense = emu:read8(addr + 2),
        baseSpeed = emu:read8(addr + 3),
        baseSpAttack = emu:read8(addr + 4),
        baseSpDefense = emu:read8(addr + 5),
        type1 = emu:read8(addr + 6),
        type2 = emu:read8(addr + 7),
        catchRate = emu:read8(addr + 8),
        expYield = emu:read8(addr + 9),
    }
end

function get_move_info(move_id)
    local addr = ADDR_BATTLE_MOVES + move_id * 12
    return {
        effect = emu:read8(addr + 0),
        power = emu:read8(addr + 1),
        type = emu:read8(addr + 2),
        accuracy = emu:read8(addr + 3),
        pp = emu:read8(addr + 4),
        secondaryEffectChance = emu:read8(addr + 5),
        target = emu:read8(addr + 6),
        priority = emu:read8(addr + 7),
        flags = emu:read8(addr + 8),
    }
end

function decrypt_pokemon_data(personality, otId, data_addr)
    local key = otId ~ personality
    local substructSelector = {
        [ 0] = {0, 1, 2, 3}, [ 1] = {0, 1, 3, 2}, [ 2] = {0, 2, 1, 3}, [ 3] = {0, 3, 1, 2},
        [ 4] = {0, 2, 3, 1}, [ 5] = {0, 3, 2, 1}, [ 6] = {1, 0, 2, 3}, [ 7] = {1, 0, 3, 2},
        [ 8] = {2, 0, 1, 3}, [ 9] = {3, 0, 1, 2}, [10] = {2, 0, 3, 1}, [11] = {3, 0, 2, 1},
        [12] = {1, 2, 0, 3}, [13] = {1, 3, 0, 2}, [14] = {2, 1, 0, 3}, [15] = {3, 1, 0, 2},
        [16] = {2, 3, 0, 1}, [17] = {3, 2, 0, 1}, [18] = {1, 2, 3, 0}, [19] = {1, 3, 2, 0},
        [20] = {2, 1, 3, 0}, [21] = {3, 1, 2, 0}, [22] = {2, 3, 1, 0}, [23] = {3, 2, 1, 0},
    }

    local pSel = substructSelector[personality % 24]
    local decrypted = {}

    for i = 0, 3 do
        local ss_idx = pSel[i + 1]
        local ss_addr = data_addr + ss_idx * 12
        local ss_data = {}
        for j = 0, 2 do
            local val = emu:read32(ss_addr + j * 4)
            val = val ~ key
            table.insert(ss_data, val)
        end
        decrypted[i] = ss_data
    end

    return decrypted
end

function get_party_info()
    local count = emu:read8(ADDR_PLAYER_PARTY_COUNT)
    local party = {}

    for i = 0, count - 1 do
        local base = ADDR_PLAYER_PARTY + i * 100
        local personality = emu:read32(base + 0)
        local otId = emu:read32(base + 4)
        local decrypted = decrypt_pokemon_data(personality, otId, base + 32)

        local ss0 = decrypted[0]
        local species = ss0[1] & 0xFFFF
        local heldItem = (ss0[1] >> 16) & 0xFFFF
        local experience = ss0[2]
        local ppBonuses = ss0[3] & 0xFF
        local friendship = (ss0[3] >> 8) & 0xFF

        local ss1 = decrypted[1]
        local moves = {
            ss1[1] & 0xFFFF, (ss1[1] >> 16) & 0xFFFF,
            ss1[2] & 0xFFFF, (ss1[2] >> 16) & 0xFFFF
        }
        local pp = {
            ss1[3] & 0xFF, (ss1[3] >> 8) & 0xFF,
            (ss1[3] >> 16) & 0xFF, (ss1[3] >> 24) & 0xFF
        }

        local ss2 = decrypted[2]
        local hpEV = ss2[1] & 0xFF
        local attackEV = (ss2[1] >> 8) & 0xFF
        local defenseEV = (ss2[1] >> 16) & 0xFF
        local speedEV = (ss2[1] >> 24) & 0xFF
        local spAttackEV = ss2[2] & 0xFF
        local spDefenseEV = (ss2[2] >> 8) & 0xFF

        local ss3 = decrypted[3]
        local iv_data = ss3[2]
        local hpIV = iv_data & 0x1F
        local attackIV = (iv_data >> 5) & 0x1F
        local defenseIV = (iv_data >> 10) & 0x1F
        local speedIV = (iv_data >> 15) & 0x1F
        local spAttackIV = (iv_data >> 20) & 0x1F
        local spDefenseIV = (iv_data >> 25) & 0x1F
        local isEgg = (iv_data >> 30) & 1

        local status = emu:read32(base + 80)
        local level = emu:read8(base + 84)
        local hp = emu:read16(base + 86)
        local maxHP = emu:read16(base + 88)
        local attack = emu:read16(base + 90)
        local defense = emu:read16(base + 92)
        local speed = emu:read16(base + 94)
        local spAttack = emu:read16(base + 96)
        local spDefense = emu:read16(base + 98)

        local speciesInfo = get_species_info(species)
        local movesInfo = {}
        for _, move_id in ipairs(moves) do
            table.insert(movesInfo, get_move_info(move_id))
        end

        table.insert(party, {
            species = species,
            speciesInfo = speciesInfo,
            level = level,
            hp = hp,
            maxHP = maxHP,
            attack = attack,
            defense = defense,
            speed = speed,
            spAttack = spAttack,
            spDefense = spDefense,
            moves = moves,
            movesInfo = movesInfo,
            pp = pp,
            status = status,
            experience = experience,
            heldItem = heldItem,
            friendship = friendship,
            evs = {hp=hpEV, atk=attackEV, def=defenseEV, spd=speedEV, spatk=spAttackEV, spdef=spDefenseEV},
            ivs = {hp=hpIV, atk=attackIV, def=defenseIV, spd=speedIV, spatk=spAttackIV, spdef=spDefenseIV},
            isEgg = isEgg
        })
    end
    return party
end

function table_to_json(tbl)
    local str = "{"
    local first = true
    for k, v in pairs(tbl) do
        if not first then str = str .. "," end
        first = false
        str = str .. '"' .. k .. '":'
        if type(v) == "table" then
            str = str .. table_to_json(v)
        elseif type(v) == "number" then
            str = str .. v
        elseif type(v) == "boolean" then
            str = str .. (v and "true" or "false")
        else
            str = str .. '"' .. tostring(v) .. '"'
        end
    end
    str = str .. "}"
    return str
end

function array_to_json(arr)
    local str = "["
    local first = true
    for _, v in ipairs(arr) do
        if not first then str = str .. "," end
        first = false
        if type(v) == "table" then
            str = str .. table_to_json(v)
        else
            str = str .. v
        end
    end
    str = str .. "]"
    return str
end

function send_state()
    if not client then return end
    local x, y = get_player_position()
    local party = get_party_info()
    local state = string.format('{"x": %d, "y": %d, "party": %s}\n', x, y, array_to_json(party))
    client:send(state)
end

function process_line(line)
    local keys = 0
    for k, v in pairs(KEY_BITS) do
        if string.find(line, k) then
            keys = keys | v
        end
    end
    emu:setKeys(keys)
end

function on_receive()
    -- Limit the number of reads per frame to prevent freezing/crashing
    local max_reads = 10
    local reads = 0
    while reads < max_reads do
        local p, err = client:receive(1024)
        if p then
            buffer = buffer .. p
            while true do
                local line_end = string.find(buffer, "\n")
                if not line_end then break end
                local line = string.sub(buffer, 1, line_end - 1)
                buffer = string.sub(buffer, line_end + 1)
                process_line(line)
            end
        else
            if err ~= socket.ERRORS.AGAIN then
                console:error("Socket Error: " .. tostring(err))
                client:close()
                client = nil
            end
            return
        end
        reads = reads + 1
    end
end

function on_accept()
    local c, err = server:accept()
    if c then
        client = c
        console:log("Client connected")
        client:add("received", on_receive)
        client:add("error", function()
            console:log("Client disconnected")
            client = nil
        end)
    else
        console:error("Accept error: " .. err)
    end
end

function on_frame()
    if not server then
        -- Bind to 127.0.0.1 explicitly to ensure IPv4 localhost
        local err
        server, err = socket.bind("127.0.0.1", port)
        if server then
            local ok
            ok, err = server:listen()
            if not ok then
                console:error("Failed to listen on port " .. port .. ": " .. tostring(err))
                server:close()
                server = nil
                return
            end
            console:log("Server started on " .. "127.0.0.1" .. ":" .. port)
            server:add("received", on_accept)
        else
            console:error("Failed to bind port " .. port .. ": " .. tostring(err))
        end
    end

    if client then
        send_state()
    end
end

callbacks:add("frame", on_frame)
console:log("RL Bridge loaded. Waiting for connection on port " .. port)

-- Debug info
if emu.pause then console:log("emu.pause available") else console:log("emu.pause NOT available") end

