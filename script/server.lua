-- Connect to your Python receiver
local sock = socket.connect("127.0.0.1", 5000)
if not sock then
    console:error("Failed to connect to Python server!")
    return
end
console:log("Connected to Python receiver")

-- === Utility ===
local function pack_uint32_be(n)
    n = n % 2^32
    local b1 = math.floor(n / 2^24)
    local b2 = math.floor((n % 2^24) / 2^16)
    local b3 = math.floor((n % 2^16) / 2^8)
    local b4 = n % 256
    return string.char(b1, b2, b3, b4)
end

local function busy_wait(n)
    local x = 0
    for i = 1, n do
        x = x + i
    end
end

-- === NEW FUNCTION ===
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

-- === Frame callback ===
local currentframe = -1

callbacks:add("frame", function()
    local frame = emu:currentFrame()
    if currentframe ~= frame then
        currentframe = frame

        -- Send game memory snapshot
        local ewram = emu:readRange(0x02000000, 0x40000)
        local iwram = emu:readRange(0x03000000, 0x8000)
        local payload = ewram .. iwram

        local length = #payload
        local header = pack_uint32_be(length) .. pack_uint32_be(frame)

        sock:send(header .. payload)

        -- Check for messages from Python
        receive_and_print()

        busy_wait(100000)
    end
end)
