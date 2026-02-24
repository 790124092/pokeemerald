-- inspect.lua
console:log("Inspecting emu table:")
for k, v in pairs(emu) do
    console:log("emu." .. k .. " (" .. type(v) .. ")")
end

console:log("Inspecting socket table:")
if socket then
    for k, v in pairs(socket) do
        console:log("socket." .. k .. " (" .. type(v) .. ")")
    end
else
    console:log("socket is nil")
end

