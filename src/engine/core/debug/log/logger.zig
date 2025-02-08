const std = @import("std");
const builtin = @import("builtin");

const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;
const Atomic = std.atomic.Value;
const Mutex = std.Thread.Mutex;

const LogLevel = enum { debug, info, warn, err, fatal };
const LogEntry = struct {
    level: LogLevel,
    msg: []const u8,
    file: []const u8,
    line: u32,
    executor_id: std.Thread.Id,
    observed_at: i128,
    arena: ArenaAllocator,
};

pub const Logger = struct {
    allocator: Allocator,
    buffer: []?*LogEntry,
    head: Atomic(usize),
    tail: Atomic(usize),
    threads: []std.Thread,
    running: Atomic(bool),
    log_file: std.fs.File,
    config: Configuration,
    file_lock: Mutex,
    semaphore: Atomic(usize),
    start_time: i128,

    pub const Configuration = struct {
        buffer_size: usize = 1 << 20,
        worker_count: u16 = 4,
    };

    pub fn init(allocator: Allocator, config: Configuration) !*Logger {
        var self = try allocator.create(Logger);
        self.* = .{
            .allocator = allocator,
            .buffer = try allocator.alloc(?*LogEntry, config.buffer_size),
            .head = Atomic(usize).init(0),
            .tail = Atomic(usize).init(0),
            .threads = try allocator.alloc(std.Thread, config.worker_count),
            .running = Atomic(bool).init(true),
            .log_file = try openLogFile(allocator),
            .config = config,
            .file_lock = .{},
            .semaphore = Atomic(usize).init(config.buffer_size),
            .start_time = std.time.nanoTimestamp(),
        };

        @memset(self.buffer, null);

        for (0..config.worker_count) |i| {
            self.threads[i] = try std.Thread.spawn(.{}, processEntries, .{self});
        }
        return self;
    }

    fn processEntries(self: *Logger) void {
        while (self.running.load(.seq_cst)) {
            const idx = self.head.fetchAdd(1, .monotonic);
            const slot = &self.buffer[idx % self.config.buffer_size];

            if (slot.*) |event| {
                defer {
                    event.arena.deinit();
                    slot.* = null;
                    _ = self.semaphore.fetchAdd(1, .monotonic);
                }

                self.file_lock.lock();
                defer self.file_lock.unlock();

                const time = formatTime(event.observed_at - self.start_time);
                const msg = std.fmt.allocPrint(self.allocator, "[{d:0>2}:{d:0>2}:{d:0>2}.{d:0>3}] {s} ({s}:{d}) [{x}]: {s}\n", .{
                    time.hours,
                    time.minutes,
                    time.seconds,
                    time.millis,
                    @tagName(event.level),
                    std.fs.path.basename(event.file),
                    event.line,
                    event.executor_id,
                    event.msg,
                }) catch continue;
                defer self.allocator.free(msg);

                _ = self.log_file.write(msg) catch null;
                _ = std.io.getStdErr().write(msg) catch null;
            } else {
                std.time.sleep(10_000);
            }
        }
    }

    pub fn deinit(self: *Logger) void {
        self.running.store(false, .seq_cst);
        for (self.threads) |thread| {
            thread.join();
        }
        self.allocator.free(self.threads);
        self.allocator.free(self.buffer);
        self.log_file.close();
        self.allocator.destroy(self);
    }

    pub inline fn record(
        self: *Logger,
        comptime level: LogLevel,
        comptime fmt: []const u8,
        args: anytype,
        comptime src: std.builtin.SourceLocation,
    ) void {
        //if (@intFromEnum(level) < @intFromEnum(std.log.Level.debug)) return;

        while (true) {
            const available = self.semaphore.load(.monotonic);
            if (available == 0) return;

            const result = self.semaphore.cmpxchgStrong(available, available - 1, .monotonic, .monotonic);

            if (result == null) break;
        }

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        errdefer arena.deinit();

        const msg = std.fmt.allocPrintZ(arena.allocator(), fmt, args) catch return;
        const event = arena.allocator().create(LogEntry) catch return;
        const event_src = std.builtin.SourceLocation{
            .file = src.file,
            .line = src.line,
            .column = 0,
            .fn_name = src.fn_name,
            .module = src.module,
        };

        event.* = .{
            .level = level,
            .msg = msg,
            .file = event_src.file,
            .line = event_src.line,
            .executor_id = std.Thread.getCurrentId(),
            .observed_at = std.time.nanoTimestamp(),
            .arena = arena,
        };

        var idx = self.tail.load(.monotonic);
        while (true) {
            const buffer_idx = idx % self.config.buffer_size;
            if (self.buffer[buffer_idx] != null) {
                idx = self.tail.load(.monotonic);
                continue;
            }

            if (self.tail.cmpxchgStrong(idx, idx + 1, .release, .monotonic) != null) {
                self.buffer[buffer_idx] = event;
                return;
            }
        }
    }
};

fn openLogFile(allocator: std.mem.Allocator) !std.fs.File {
    try std.fs.cwd().makePath("logs");
    const timestamp = std.time.timestamp();
    const path = try std.fmt.allocPrintZ(allocator, "logs/app_{d}.log", .{timestamp});
    defer allocator.free(path);

    return std.fs.cwd().createFile(path, .{
        .read = true,
        .truncate = false,
        .lock = .exclusive,
    });
}

fn formatTime(timestamp: i128) struct { hours: u8, minutes: u8, seconds: u8, millis: u16 } {
    const seconds_total = @divFloor(timestamp, std.time.ns_per_s);
    const millis = @divFloor(@mod(timestamp, std.time.ns_per_s), std.time.ns_per_ms);
    const hours = @mod(@divFloor(seconds_total, 3600), 24);
    const minutes = @mod(@divFloor(seconds_total, 60), 60);
    const seconds = @mod(seconds_total, 60);
    return .{
        .hours = @intCast(hours),
        .minutes = @intCast(minutes),
        .seconds = @intCast(seconds),
        .millis = @intCast(millis),
    };
}
