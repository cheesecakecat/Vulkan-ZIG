const std = @import("std");

pub const LogLevel = enum {
    debug,
    info,
    warn,
    err,
};

var min_level: LogLevel = .debug;
var buffer_writer: ?std.io.BufferedWriter(4096, std.fs.File.Writer) = null;
var start_time: i128 = 0;

pub fn init() !void {
    buffer_writer = std.io.bufferedWriter(std.io.getStdErr().writer());
    start_time = std.time.nanoTimestamp();
}

pub fn deinit() void {
    if (buffer_writer) |*writer| {
        writer.flush() catch {};
    }
}

pub fn setLevel(level: LogLevel) void {
    min_level = level;
}

var time_buffer: [32]u8 = undefined;

fn getTimestamp() []const u8 {
    const elapsed = std.time.nanoTimestamp() - start_time;
    const total_millis = @divFloor(elapsed, std.time.ns_per_ms);

    const millis = @mod(total_millis, 1000);
    const total_seconds = @divFloor(total_millis, 1000);
    const hours = @divFloor(total_seconds, 3600);
    const mins = @mod(@divFloor(total_seconds, 60), 60);
    const secs = @mod(total_seconds, 60);

    return std.fmt.bufPrint(&time_buffer, "{:0>2}:{:0>2}:{:0>2}.{:0>3}", .{
        @as(u8, @intCast(hours)),
        @as(u8, @intCast(mins)),
        @as(u8, @intCast(secs)),
        @as(u16, @intCast(millis)),
    }) catch "??:??:??.???";
}

fn log(level: LogLevel, comptime fmt: []const u8, args: anytype) void {
    if (@intFromEnum(level) < @intFromEnum(min_level)) return;
    if (buffer_writer == null) return;

    const writer = buffer_writer.?.writer();
    const timestamp = getTimestamp();

    const prefix = switch (level) {
        .debug => "\x1b[90m[dbg]\x1b[0m", // gray
        .info => "\x1b[32m[inf]\x1b[0m", // green
        .warn => "\x1b[33m[wrn]\x1b[0m", // yellow
        .err => "\x1b[31m[err]\x1b[0m", // red
    };

    // [HH:MM:SS.mmm] [level] message
    writer.print("\x1b[90m[{s}]\x1b[0m {s} ", .{ timestamp, prefix }) catch return;
    writer.print(fmt ++ "\n", args) catch return;
    buffer_writer.?.flush() catch {};
}

pub fn debug(comptime fmt: []const u8, args: anytype) void {
    log(.debug, fmt, args);
}

pub fn info(comptime fmt: []const u8, args: anytype) void {
    log(.info, fmt, args);
}

pub fn warn(comptime fmt: []const u8, args: anytype) void {
    log(.warn, fmt, args);
}

pub fn err(comptime fmt: []const u8, args: anytype) void {
    log(.err, fmt, args);
}
