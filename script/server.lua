-- === TCP connection to Python ===
local sock = socket.connect("127.0.0.1", 5000)
if not sock then
    console:error("Failed to connect to Python server!")
    return
end
console:log("Connected to Python receiver")

-- === Utility functions ===
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

local function receive_and_print()
    if sock:hasdata() then
        local msg, err = sock:receive(4096)
        if msg then
            console:log("[From Python] " .. msg)
        elseif err and err ~= socket.ERRORS.WOULD_BLOCK then
            console:error("Socket receive error: " .. tostring(err))
        end
    end
end

-- === Frame variables ===
local currentframe = -1

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

-- === Frame callback ===
callbacks:add("frame", function()
    local frame = emu:currentFrame()
    if currentframe ~= frame then
        currentframe = frame

        -- Send memory snapshot
        local ewram = emu:readRange(0x02000000, 0x40000)
        local iwram = emu:readRange(0x03000000, 0x8000)
        local payload = ewram .. iwram
        send_message(payload, frame)

        receive_and_print()
    end
end)
