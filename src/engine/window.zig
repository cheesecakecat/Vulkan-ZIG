const std = @import("std");
const glfw = @import("mach-glfw");

pub const Window = struct {
    handle: ?glfw.Window,
    width: u32,
    height: u32,
    title: [*:0]const u8,
    framebuffer_resized: bool,

    pub fn init(width: u32, height: u32, title: [*:0]const u8) !Window {
        if (!glfw.init(.{})) {
            return error.GLFWInitFailed;
        }

        return Window{
            .handle = null,
            .width = width,
            .height = height,
            .title = title,
            .framebuffer_resized = false,
        };
    }

    pub fn create(self: *Window) !void {
        self.handle = glfw.Window.create(
            self.width,
            self.height,
            self.title,
            null,
            null,
            .{ .client_api = .no_api },
        ) orelse return error.WindowCreationFailed;

        self.handle.?.setUserPointer(self);
        _ = self.handle.?.setFramebufferSizeCallback(framebufferResizeCallback);
    }

    pub fn deinit(self: *Window) void {
        if (self.handle) |window| {
            window.destroy();
        }
        glfw.terminate();
    }

    pub fn shouldClose(self: *Window) bool {
        return if (self.handle) |window| window.shouldClose() else true;
    }

    pub fn getFramebufferSize(self: *Window) struct { width: u32, height: u32 } {
        if (self.handle) |window| {
            const size = window.getFramebufferSize();
            return .{
                .width = @intCast(size.width),
                .height = @intCast(size.height),
            };
        }
        return .{ .width = 0, .height = 0 };
    }

    fn framebufferResizeCallback(window: glfw.Window, width: u32, height: u32) void {
        _ = width;
        _ = height;
        const self = window.getUserPointer(Window) orelse return;
        self.framebuffer_resized = true;
    }
};
