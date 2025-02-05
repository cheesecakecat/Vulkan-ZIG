const std = @import("std");

pub const LogLevel = enum {
    debug,
    info,
    warn,
    err,
};

var min_level: LogLevel = .debug;
var buffer_writer: ?std.io.BufferedWriter(4096, std.fs.File.Writer) = null;
var file_writer: ?std.fs.File.Writer = null;
var start_time: i128 = 0;

fn getLogFilename(allocator: std.mem.Allocator) ![]const u8 {
    const current_time = std.time.timestamp();
    const time_info = std.time.epoch.EpochSeconds{ .secs = @intCast(current_time) };
    const day_seconds = time_info.getDaySeconds();

    const total_secs = day_seconds.secs;

    const hours = @divFloor(total_secs, 3600);
    const minutes = @mod(@divFloor(total_secs, 60), 60);
    const seconds = @mod(total_secs, 60);

    return std.fmt.allocPrint(allocator, "{d:0>2}_{d:0>2}_{d:0>2}-{s}.log", .{
        hours,
        minutes,
        seconds,
        @tagName(min_level),
    });
}

pub fn init() !void {
    try std.fs.cwd().makePath("logs");

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const filename = try getLogFilename(allocator);
    var dir = try std.fs.cwd().openDir("logs", .{});
    defer dir.close();

    const log_file = try dir.createFile(filename, .{});
    file_writer = log_file.writer();
    buffer_writer = std.io.bufferedWriter(std.io.getStdErr().writer());
    start_time = std.time.nanoTimestamp();
}

pub fn deinit() void {
    if (buffer_writer) |*writer| {
        writer.flush() catch {};
    }
    if (file_writer) |fw| {
        fw.context.sync() catch {};
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
        .debug => "[dbg]",
        .info => "[inf]",
        .warn => "[wrn]",
        .err => "[err]",
    };

    const colored_prefix = switch (level) {
        .debug => "\x1b[90m[dbg]\x1b[0m",
        .info => "\x1b[32m[inf]\x1b[0m",
        .warn => "\x1b[33m[wrn]\x1b[0m",
        .err => "\x1b[31m[err]\x1b[0m",
    };
    writer.print("\x1b[90m[{s}]\x1b[0m {s} ", .{ timestamp, colored_prefix }) catch return;
    writer.print(fmt ++ "\n", args) catch return;
    buffer_writer.?.flush() catch {};

    if (file_writer) |fw| {
        fw.print("[{s}] {s} ", .{ timestamp, prefix }) catch return;
        fw.print(fmt ++ "\n", args) catch return;

        fw.context.sync() catch {};
    }
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
