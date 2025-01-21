const std = @import("std");
const Window = @import("engine/window.zig").Window;
const Renderer = @import("engine/renderer.zig").Renderer;
const glfw = @import("mach-glfw");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var window = try Window.init(800, 600, "Vulkan Triangle");
    defer window.deinit();
    try window.create();

    var renderer = try Renderer.init(&window, allocator);
    defer renderer.deinit();

    while (!window.shouldClose()) {
        glfw.pollEvents();
        try renderer.drawFrame();
    }
}
