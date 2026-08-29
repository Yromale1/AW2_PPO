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

local pending_send = nil
local pending_pos = 1

local function queue_send(data)
    if pending_send ~= nil then
        console:error("Send already pending")
        return false
    end

    pending_send = data
    pending_pos = 1
    return true
end

local function flush_send()
    if pending_send == nil then
        return true
    end

    local last, err = sock:send(
        pending_send,
        pending_pos,
        #pending_send
    )

    if last then
        pending_pos = last + 1

        if pending_pos > #pending_send then
            pending_send = nil
            pending_pos = 1
            return true
        end

        return false
    end

    if err == socket.ERRORS.AGAIN then
        -- Socket is temporarily unable to accept more data.
        -- Keep pending_send and try again later.
        return false
    end

    error("Socket send failed: " .. tostring(err))
end

local function send_message(payload, frame)
    local header = pack_uint32_be(#payload) .. pack_uint32_be(frame)
    queue_send(header .. payload)
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

local MSG_ACTION = 0
local MSG_RESET  = 1

local recv_buffer = ""

local function unpack_uint32_be(data, pos)
    local b1, b2, b3, b4 = string.byte(data, pos, pos + 3)

    return b1 * 2^24 +
           b2 * 2^16 +
           b3 * 2^8 +
           b4
end

local function receive_messages()
    while sock:hasdata() do
        local chunk, err = sock:receive(4096)

        if not chunk then
            if err == socket.ERRORS.WOULD_BLOCK then
                return
            end

            console:error("Socket receive error: " .. tostring(err))
            return
        end

        recv_buffer = recv_buffer .. chunk
    end
end

local function get_message()
    -- Not enough data for header
    if #recv_buffer < 8 then
        return nil
    end

    local length = unpack_uint32_be(recv_buffer, 1)
    local message_type = unpack_uint32_be(recv_buffer, 5)

    -- Don't have complete payload yet
    if #recv_buffer < 8 + length then
        return nil
    end

    local payload = ""

    if length > 0 then
        payload = string.sub(recv_buffer, 9, 8 + length)
    end

    -- Remove consumed message from buffer
    recv_buffer = string.sub(recv_buffer, 9 + length)

    return message_type, payload
end


local function receive_and_send(frame)

    -- Read whatever TCP data is currently available
    receive_messages()

    local message_type, msg = get_message()

    if not message_type then
        return nil
    end

    if message_type == MSG_RESET then
        console:log("RESET")

        emu:loadStateFile(
            "/home/yro/code/rom/AW2.ss5"
        )

        return nil
    end

    if message_type == MSG_ACTION then
        local values = unpack_floats(msg)

        local mask = tensor_to_bitmask(values)
        local msg_string = tensor_to_string(values)

        console:log("[From Python] " .. msg_string)

        emu:setKeys(mask)

        console:log(
            keysToString(emu:getKeys())
        )

        -- A breakpoint triggered while processing this action.
        -- Its result is the response to THIS input.
        if terminal_frame ~= nil then
            send_message(terminal_payload, terminal_frame)

            -- Consume the terminal event so it cannot be sent again.
            terminal_frame = nil
            terminal_payload = nil

            return mask
        end

        -- Send memory snapshot
        local ewram = emu:readRange(
            0x02000000,
            0x40000
        )

        local iwram = emu:readRange(
            0x03000000,
            0x8000
        )

        local payload = ewram .. iwram

        send_message(payload, frame)

        return mask
    end

    console:error(
        "Unknown message type: " .. tostring(message_type)
    )

    return nil
end

local terminal_frame = nil
local terminal_payload = nil

-- Win routine tracking
local WIN_ROUTINE = 0x0803861C
local WIN_FRAME = 0xFFFFFFFF

local end_bp = emu:setBreakpoint(function(addr)
    local winner = emu:readRegister("r0")

    if winner ~= 0 then
        terminal_frame = WIN_FRAME
        terminal_payload = string.format("Player %d won", winner)
    end
end, WIN_ROUTINE)

local EXIT_ROUTINE = 0x080185C8
local EXIT_FRAME = 0x11111111

local exit_bp = emu:setBreakpoint(function(addr)
    terminal_frame = EXIT_FRAME
    terminal_payload = "Exit map triggered"
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
    flush_send()
    
end)
