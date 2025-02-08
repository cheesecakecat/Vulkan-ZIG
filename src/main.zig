const std = @import("std");
const game = @import("game");
const Logger = game.logger.Logger;

pub fn main() !void {
    const logger = try Logger.init(std.heap.page_allocator, .{});
    defer logger.deinit();

    logger.record(.info, "hello world!", .{}, @src());
    std.time.sleep(100 * std.time.ns_per_ms);
}
