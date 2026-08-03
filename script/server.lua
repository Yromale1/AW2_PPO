-- === TCP connection to Python ===
local sock = socket.connect("127.0.0.1", 5000)
if not sock then
    console:error("Failed to connect to Python server!")
    return
end
console:log("Connected to Python receiver")

-- === Utility values ===
local KEY_NAMES = {}

for name, value in pairs(C.GBA_KEY) do
    KEY_NAMES[value] = name
end

-- === Utility functions ===
local function keysToString(keys)
    local name = ""

    key = 0
    for i = 15, 0, -1 do
        key = key + ((keys >> i) & 1) * i
    end
    name = KEY_NAMES[key] or utf8.char(key)

    return name
end

local function pack_uint32_be(n)
    n = n % 2^32
    local b1 = math.floor(n / 2^24)
    local b2 = math.floor((n % 2^24) / 2^16)
    local b3 = math.floor((n % 2^16) / 2^8)
    local b4 = n % 256
    return string.char(b1, b2, b3, b4)
end

local function send_message(payload, frame)
    local header = pack_uint32_be(#payload) .. pack_uint32_be(frame)
    sock:send(header .. payload)
end

local function busy_wait(n)
    local x = 0
    for i = 1, n do x = x + i end
end

local function unpack_floats(data)
    print(data)
    local values = {}
    local pos = 1

    while pos <= #data do
        values[#values + 1], pos = string.unpack("<f", data, pos)
    end

    return values
end

local function tensor_to_string(data)
    local inputs = {}

    for i, v in ipairs(data) do
        if v == 1.0 then
            inputs[#inputs+1] = i
        end
    end

    return "{" .. table.concat(inputs, ", ") .. "}"
end

local function tensor_to_bitmask(data)
    local mask = 0

    for _, v in ipairs(data) do
        mask = (mask << 1) | v
    end

    return mask
end

local function receive_and_send(frame)

    if sock:hasdata() then
        local msg, err = sock:receive(4096)
        local success, result = pcall(string.unpack, "s4", msg)
        if success and result == "RESET" then
            console:log("RESET")
            emu:loadStateFile("/home/yro/code/rom/AW2.ss2")
        else
            local mask = tensor_to_bitmask(unpack_floats(msg))
            msg = tensor_to_string(unpack_floats(msg))
            if msg then
                console:log("[From Python] " .. msg)
                emu:setKeys(mask)
                console:log(keysToString(emu:getKeys()))
            elseif err and err ~= socket.ERRORS.WOULD_BLOCK then
                console:error("Socket receive error: " .. tostring(err))
                return
            end
            -- Send memory snapshot
            local ewram = emu:readRange(0x02000000, 0x40000)
            local iwram = emu:readRange(0x03000000, 0x8000)
            local payload = ewram .. iwram
            send_message(payload, frame)
            return mask
        end
    end
    return nil
end

-- Win routine tracking
local WIN_ROUTINE = 0x0803861D
local WIN_FRAME = 0xFFFFFFFF

local end_bp = emu:setBreakpoint(function(addr)
    local winner = emu:readRegister("r0")
    if winner ~= 0 then
        local msg = string.format("Player %d won", winner)
        send_message(msg, WIN_FRAME)
        last_winner = winner
    end
end, WIN_ROUTINE)

local EXIT_ROUTINE = 0x080185C8
local EXIT_FRAME = 0x11111111

local exit_bp = emu:setBreakpoint(function(addr)
    local msg = "Exit map triggered"
    send_message(msg, EXIT_FRAME)
    console:log(msg)
end, EXIT_ROUTINE)

-- === Frame variables ===
local currentframe = -1
local old_mask = nil  -- Action mask from the last frame

-- === Frame callback ===
callbacks:add("frame", function()
    local frame = emu:currentFrame()
    if currentframe ~= frame then
        if old_mask ~= nil then
            emu:clearKeys(old_mask)
        end

        currentframe = frame

        old_mask = receive_and_send(currentframe)

        
    end
    
end)
