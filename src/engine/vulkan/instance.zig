const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const glfw = @import("mach-glfw");
const logger = @import("../core/logger.zig");
const types = @import("context.types.zig");

const VK_KHR_SURFACE_EXTENSION_NAME = "VK_KHR_surface";
const VK_KHR_WIN32_SURFACE_EXTENSION_NAME = "VK_KHR_win32_surface";
const VK_KHR_XLIB_SURFACE_EXTENSION_NAME = "VK_KHR_xlib_surface";
const VK_EXT_DEBUG_UTILS_EXTENSION_NAME = "VK_EXT_debug_utils";
const VK_EXT_DEBUG_REPORT_EXTENSION_NAME = "VK_EXT_debug_report";

const INITIAL_EXTENSION_CAPACITY: u32 = 32;
const INITIAL_LAYER_CAPACITY: u32 = 16;
const DEBUG_MESSAGE_BUFFER_SIZE: u32 = 1024;
const MAX_DEBUG_PRINTF_MESSAGES: u32 = 1024;
const MESSAGE_CACHE_SIZE: u32 = 1024;
const POWER_UPDATE_INTERVAL = std.time.ns_per_s * 5;
const MAX_VALIDATION_FEATURES: u32 = 5;
const MESSAGE_CACHE_SHARDS: u32 = 8;
const POWER_STATE_HISTORY_SIZE: u32 = 8;
const STRING_POOL_SIZE: u32 = INITIAL_EXTENSION_CAPACITY + INITIAL_LAYER_CAPACITY;
const SMALL_STRING_THRESHOLD: u32 = 32;
const BLOOM_FILTER_SIZE: u32 = 2048;
const MESSAGE_HISTORY_SIZE: u32 = 64;
const OBJECT_POOL_SIZE: u32 = 256;
const THERMAL_SAMPLES: u32 = 16;

const MAX_THREAD_COUNT: u32 = 32;
const CACHE_LINE_SIZE = 64;

threadlocal var debug_msg_buf: [DEBUG_MESSAGE_BUFFER_SIZE]u8 align(64) = undefined;
threadlocal var printf_msg_buf: [DEBUG_MESSAGE_BUFFER_SIZE]u8 align(64) = undefined;
threadlocal var msg_history_buf: [MESSAGE_HISTORY_SIZE][DEBUG_MESSAGE_BUFFER_SIZE]u8 align(64) = undefined;
threadlocal var msg_history_idx: usize = 0;

const ObjectPool = struct {
    const Block = struct {
        data: [64]u8 align(8),
        in_use: bool,
    };

    blocks: [OBJECT_POOL_SIZE]Block align(64),
    free_list: std.atomic.Value(u32) align(64),

    fn init() ObjectPool {
        var pool: ObjectPool = .{
            .blocks = undefined,
            .free_list = std.atomic.Value(u32).init(0),
        };
        for (0..OBJECT_POOL_SIZE) |i| {
            pool.blocks[i] = .{
                .data = undefined,
                .in_use = false,
            };
        }
        return pool;
    }

    fn allocate(self: *ObjectPool) ?*Block {
        var current = self.free_list.load(.acquire);
        while (current < OBJECT_POOL_SIZE) {
            if (!self.blocks[current].in_use) {
                const old = current;
                const new = current + 1;
                if (self.free_list.cmpxchgStrong(old, new, .acq_rel, .acquire)) |_| {
                    continue;
                }
                self.blocks[current].in_use = true;
                return &self.blocks[current];
            }
            current += 1;
        }
        return null;
    }

    fn free(self: *ObjectPool, block: *Block) void {
        block.in_use = false;
        _ = self.free_list.fetchSub(1, .release);
    }
};

const StringPool = struct {
    const Entry = struct {
        hash: u64,
        str: []const u8,
    };

    var entries: [STRING_POOL_SIZE]Entry = undefined;
    var count: usize = 0;
    var bloom: [BLOOM_FILTER_SIZE / 8]u8 align(8) = [_]u8{0} ** (BLOOM_FILTER_SIZE / 8);

    fn fnv1a(str: []const u8) u64 {
        const FNV_PRIME: u64 = 0x100000001b3;
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        var hash: u64 = FNV_OFFSET;
        for (str) |byte| {
            hash ^= byte;
            hash *%= FNV_PRIME;
        }
        return hash;
    }

    fn bloomAdd(hash: u64) void {
        const idx1: u32 = @truncate(hash & (BLOOM_FILTER_SIZE - 1));
        const idx2: u32 = @truncate((hash >> 16) & (BLOOM_FILTER_SIZE - 1));
        const byte1 = idx1 >> 3;
        const byte2 = idx2 >> 3;
        const bit1 = @as(u8, 1) << @truncate(idx1 & 7);
        const bit2 = @as(u8, 1) << @truncate(idx2 & 7);
        @atomicStore(u8, &bloom[byte1], bloom[byte1] | bit1, .release);
        @atomicStore(u8, &bloom[byte2], bloom[byte2] | bit2, .release);
    }

    fn bloomMayContain(hash: u64) bool {
        const idx1: u32 = @truncate(hash & (BLOOM_FILTER_SIZE - 1));
        const idx2: u32 = @truncate((hash >> 16) & (BLOOM_FILTER_SIZE - 1));
        const byte1 = idx1 >> 3;
        const bit1 = @as(u8, 1) << @truncate(idx1 & 7);
        const byte2 = idx2 >> 3;
        const bit2 = @as(u8, 1) << @truncate(idx2 & 7);
        return (@atomicLoad(u8, &bloom[byte1], .acquire) & bit1) != 0 and
            (@atomicLoad(u8, &bloom[byte2], .acquire) & bit2) != 0;
    }

    fn add(str: []const u8) void {
        if (count >= STRING_POOL_SIZE) @panic("String pool full");
        const hash = if (str.len <= SMALL_STRING_THRESHOLD) fnv1a(str) else std.hash.Wyhash.hash(0, str);
        bloomAdd(hash);
        entries[count] = .{
            .hash = hash,
            .str = str,
        };
        count += 1;
    }

    fn get(hash: u64) ?[]const u8 {
        if (!bloomMayContain(hash)) return null;
        for (entries[0..count]) |entry| {
            if (entry.hash == hash) return entry.str;
        }
        return null;
    }

    fn contains(str: []const u8) bool {
        const hash = if (str.len <= SMALL_STRING_THRESHOLD) fnv1a(str) else std.hash.Wyhash.hash(0, str);
        logger.debug("vulkan: checking string pool for '{s}' (hash: 0x{x})", .{ str, hash });
        return get(hash) != null;
    }
};

const RingBuffer = struct {
    pub fn Buffer(comptime T: type, comptime size: u32) type {
        return struct {
            data: [size]T align(64) = undefined,
            read_idx: std.atomic.Value(u32) align(64) = std.atomic.Value(u32).init(0),
            write_idx: std.atomic.Value(u32) align(64) = std.atomic.Value(u32).init(0),
            pad: [48]u8 align(64) = undefined,

            const Self = @This();
            const CACHE_LINE = 64;
            comptime {
                std.debug.assert(@sizeOf(Self) % CACHE_LINE == 0);
                std.debug.assert(@alignOf(Self) == CACHE_LINE);
            }

            pub fn push(self: *Self, item: T) bool {
                const write = self.write_idx.load(.acquire);
                const next: u32 = @intCast((write +% 1) % size);
                const read = self.read_idx.load(.acquire);
                if (next == read) return false;

                self.data[write] = item;
                self.write_idx.store(next, .release);
                return true;
            }

            pub fn pop(self: *Self) ?T {
                const read = self.read_idx.load(.acquire);
                const write = self.write_idx.load(.acquire);
                if (read == write) return null;

                const item = self.data[read];
                const next: u32 = @intCast((read +% 1) % size);
                self.read_idx.store(next, .release);
                return item;
            }
        };
    }
};

const MessageCacheShard = struct {
    entries: [MESSAGE_CACHE_SIZE / MESSAGE_CACHE_SHARDS]struct {
        batch: std.atomic.Value(u64) align(8),
    } align(64),
    lookup: [256 / MESSAGE_CACHE_SHARDS]?u8 align(64) = .{null} ** (256 / MESSAGE_CACHE_SHARDS),
    index: std.atomic.Value(u8) align(8) = std.atomic.Value(u8).init(0),
    pad: [47]u8 align(64) = undefined,
};

const MessageCache = struct {
    shards: [MESSAGE_CACHE_SHARDS]MessageCacheShard align(64),

    fn getShard(id: u64) u8 {
        return @truncate((id >> 3) % MESSAGE_CACHE_SHARDS);
    }

    fn shouldLog(self: *MessageCache, id: u64) bool {
        const shard_idx = getShard(id);
        var shard = &self.shards[shard_idx];

        const hash: u8 = @truncate(id);
        const lookup_idx = hash % (256 / MESSAGE_CACHE_SHARDS);

        if (shard.lookup[lookup_idx]) |idx| {
            const batch = shard.entries[idx].batch.load(.acquire);
            const entry_id: u32 = @truncate(batch);
            if (entry_id == @as(u32, @truncate(id))) {
                const count: u16 = @truncate(batch >> 32);
                const timestamp: u16 = @truncate(batch >> 48);
                const now: u16 = @truncate(@as(u64, @intCast(@divTrunc(std.time.nanoTimestamp(), std.time.ns_per_s))));

                if (count < 3 or now -% timestamp > 60) {
                    const new_batch = (@as(u64, now) << 48) |
                        (@as(u64, count +% 1) << 32) |
                        entry_id;
                    shard.entries[idx].batch.store(new_batch, .release);
                    return true;
                }
                return false;
            }
        }

        const idx = shard.index.load(.acquire);
        const entries_per_shard = MESSAGE_CACHE_SIZE / MESSAGE_CACHE_SHARDS;
        const new_idx = @as(u8, @truncate((idx +% 1) % entries_per_shard));
        shard.index.store(new_idx, .release);

        const now: u16 = @truncate(@as(u64, @intCast(@divTrunc(std.time.nanoTimestamp(), std.time.ns_per_s))));
        const new_batch = (@as(u64, now) << 48) | (@as(u64, 1) << 32) | @as(u32, @truncate(id));
        shard.entries[idx].batch.store(new_batch, .release);
        shard.lookup[lookup_idx] = idx;

        return true;
    }
};

const DebugPrintfData = struct {
    const Message = struct {
        text_ptr: [*]const u8,
        len: u32,
        timestamp: i64,
        thread_id: u64,
        repeat_count: u32 = 1,
    };

    messages: RingBuffer.Buffer(Message, MAX_DEBUG_PRINTF_MESSAGES) align(64),
    thread_buffers: [MAX_THREAD_COUNT]RingBuffer.Buffer(Message, 64) align(64) = undefined,
    history: [MESSAGE_HISTORY_SIZE]Message align(64) = undefined,
    history_idx: std.atomic.Value(u32) align(8) = std.atomic.Value(u32).init(0),
    object_pool: ObjectPool align(64) = ObjectPool.init(),

    fn init() DebugPrintfData {
        return .{
            .messages = .{},
        };
    }

    fn addMessage(self: *DebugPrintfData, text: []const u8) void {
        const thread_id = std.Thread.getCurrentId();
        const buffer_idx = thread_id % MAX_THREAD_COUNT;

        const hist_idx = self.history_idx.load(.acquire);
        const last_msg = &self.history[(hist_idx -% 1) % MESSAGE_HISTORY_SIZE];
        if (last_msg.len == text.len and std.mem.eql(u8, text, last_msg.text_ptr[0..last_msg.len])) {
            last_msg.repeat_count += 1;
            return;
        }

        if (self.object_pool.allocate()) |block| {
            @memcpy(block.data[0..text.len], text);
            const msg = Message{
                .text_ptr = @ptrCast(&block.data),
                .len = @intCast(text.len),
                .timestamp = @truncate(std.time.nanoTimestamp()),
                .thread_id = thread_id,
            };

            const new_idx = (hist_idx +% 1) % MESSAGE_HISTORY_SIZE;
            self.history[hist_idx] = msg;
            self.history_idx.store(new_idx, .release);

            if (!self.thread_buffers[buffer_idx].push(msg)) {
                _ = self.messages.push(msg);
            }
        } else {
            const msg = Message{
                .text_ptr = text.ptr,
                .len = @intCast(text.len),
                .timestamp = @truncate(std.time.nanoTimestamp()),
                .thread_id = thread_id,
            };
            _ = self.messages.push(msg);
        }
    }
};

const StringCache = struct {
    const MAX_STRING_LENGTH = 256;
    const Entry = struct {
        hash: u64,
        len: u32,
        data: [MAX_STRING_LENGTH]u8 align(8),
    };

    entries: []Entry align(64),
    lookup: [256]std.atomic.Value(u32) align(64) = init: {
        @setEvalBranchQuota(1024 * 4);
        var lookup: [256]std.atomic.Value(u32) align(64) = undefined;
        for (&lookup) |*val| {
            val.* = std.atomic.Value(u32).init(std.math.maxInt(u32));
        }
        break :init lookup;
    },
    count: std.atomic.Value(u32) align(8) = std.atomic.Value(u32).init(0),
    allocator: std.mem.Allocator,
    bloom: [BLOOM_FILTER_SIZE / 8]u8 align(8) = [_]u8{0} ** (BLOOM_FILTER_SIZE / 8),

    fn init(allocator: std.mem.Allocator, capacity: u32) !StringCache {
        const entries = try allocator.alignedAlloc(Entry, 64, capacity);
        return .{
            .entries = entries,
            .allocator = allocator,
        };
    }

    fn deinit(self: *StringCache) void {
        self.allocator.free(self.entries);
    }

    fn bloomAdd(self: *StringCache, hash: u64) void {
        const idx1: u32 = @truncate(hash & (BLOOM_FILTER_SIZE - 1));
        const idx2: u32 = @truncate((hash >> 16) & (BLOOM_FILTER_SIZE - 1));
        const byte1 = idx1 >> 3;
        const byte2 = idx2 >> 3;
        const bit1 = @as(u8, 1) << @truncate(idx1 & 7);
        const bit2 = @as(u8, 1) << @truncate(idx2 & 7);
        @atomicStore(u8, &self.bloom[byte1], self.bloom[byte1] | bit1, .release);
        @atomicStore(u8, &self.bloom[byte2], self.bloom[byte2] | bit2, .release);
    }

    fn bloomMayContain(self: *const StringCache, hash: u64) bool {
        const idx1: u32 = @truncate(hash & (BLOOM_FILTER_SIZE - 1));
        const idx2: u32 = @truncate((hash >> 16) & (BLOOM_FILTER_SIZE - 1));
        const byte1 = idx1 >> 3;
        const bit1 = @as(u8, 1) << @truncate(idx1 & 7);
        const byte2 = idx2 >> 3;
        const bit2 = @as(u8, 1) << @truncate(idx2 & 7);
        return (@atomicLoad(u8, &self.bloom[byte1], .acquire) & bit1) != 0 and
            (@atomicLoad(u8, &self.bloom[byte2], .acquire) & bit2) != 0;
    }

    fn add(self: *StringCache, str: []const u8) ![]const u8 {
        const count = self.count.load(.acquire);
        for (self.entries[0..count]) |entry| {
            if (entry.len == str.len and std.mem.eql(u8, entry.data[0..str.len], str)) {
                return entry.data[0..str.len];
            }
        }

        if (count >= self.entries.len) return error.OutOfMemory;
        if (str.len >= MAX_STRING_LENGTH) return error.StringTooLong;

        const hash = if (str.len <= SMALL_STRING_THRESHOLD) StringPool.fnv1a(str) else std.hash.Wyhash.hash(0, str);
        const lookup_idx: u8 = @truncate(hash);

        if (self.count.cmpxchgStrong(count, count + 1, .acq_rel, .acquire)) |_| {
            return error.ConcurrentModification;
        }

        self.lookup[lookup_idx].store(count, .release);
        self.bloomAdd(hash);

        var entry = &self.entries[count];
        entry.hash = hash;
        entry.len = @intCast(str.len);
        @memcpy(entry.data[0..str.len], str);

        return entry.data[0..str.len];
    }

    fn contains(self: *const StringCache, str: []const u8) bool {
        const hash = if (str.len <= SMALL_STRING_THRESHOLD) StringPool.fnv1a(str) else std.hash.Wyhash.hash(0, str);
        logger.debug("vulkan: checking string cache for '{s}' (hash: 0x{x})", .{ str, hash });

        if (!self.bloomMayContain(hash)) {
            logger.debug("vulkan: bloom filter says string not present", .{});
            return false;
        }

        const count = self.count.load(.acquire);
        for (self.entries[0..count]) |entry| {
            if (entry.len == str.len and std.mem.eql(u8, entry.data[0..str.len], str)) {
                logger.debug("vulkan: found string '{s}' in cache", .{str});
                return true;
            }
        }

        logger.debug("vulkan: string '{s}' not found in cache", .{str});
        return false;
    }
};

const PowerState = struct {
    const Mode = enum(u8) {
        normal = 0,
        power_saving = 1,
        performance = 2,
    };

    const State = packed struct {
        mode: Mode,
        on_battery: bool,
        thermal_throttling: bool,
        _pad: u5 = 0,
    };

    const ThermalInfo = struct {
        temperature: f32,
        timestamp: i64,
    };

    mode: Mode align(64) = .normal,
    battery_level: ?f32 align(8) = null,
    state: State align(8) = .{
        .mode = .normal,
        .on_battery = false,
        .thermal_throttling = false,
    },
    history: [POWER_STATE_HISTORY_SIZE]State align(8) = undefined,
    history_idx: std.atomic.Value(u8) align(8) = std.atomic.Value(u8).init(0),
    last_update: std.atomic.Value(i64) align(8) = std.atomic.Value(i64).init(0),
    throttle_count: std.atomic.Value(u32) align(8) = std.atomic.Value(u32).init(0),
    thermal_samples: [THERMAL_SAMPLES]ThermalInfo align(8) = undefined,
    thermal_idx: std.atomic.Value(u8) align(8) = std.atomic.Value(u8).init(0),
    thermal_throttling: bool = false,

    fn getThermalInfo() ?f32 {
        return null;
    }

    fn updateThermalState(self: *PowerState) void {
        if (getThermalInfo()) |temp| {
            const idx = self.thermal_idx.load(.acquire);
            const new_idx: u8 = @truncate((idx +% 1) % THERMAL_SAMPLES);

            self.thermal_samples[idx] = .{
                .temperature = temp,
                .timestamp = @truncate(std.time.nanoTimestamp()),
            };
            self.thermal_idx.store(new_idx, .release);

            var sum: f32 = 0;
            var count: u32 = 0;
            var prev_temp = temp;
            var increasing: u32 = 0;

            for (self.thermal_samples[0..THERMAL_SAMPLES]) |sample| {
                if (sample.timestamp > 0) {
                    sum += sample.temperature;
                    count += 1;
                    if (sample.temperature > prev_temp) {
                        increasing += 1;
                    }
                    prev_temp = sample.temperature;
                }
            }

            if (count > 0) {
                const avg = sum / @as(f32, @floatFromInt(count));
                const trend_up = increasing > count / 2;
                self.state.thermal_throttling = avg > 80.0 or (avg > 70.0 and trend_up);
            }
        }
    }

    fn update(self: *PowerState) void {
        const now = std.time.nanoTimestamp();
        const last = self.last_update.load(.acquire);
        const throttle = self.throttle_count.load(.acquire);
        const next_check = last + @as(i64, @intCast(throttle)) * (std.time.ns_per_s / 2);

        if (@as(i64, @truncate(now)) < next_check) return;

        self.updateThermalState();

        var new_state = self.state;

        new_state.on_battery = false;
        self.battery_level = null;

        const hist_idx = self.history_idx.load(.acquire);
        const new_hist_idx: u8 = @truncate((hist_idx +% 1) % POWER_STATE_HISTORY_SIZE);
        self.history[hist_idx] = new_state;
        self.history_idx.store(new_hist_idx, .release);

        var counts = [_]u8{0} ** 3;
        for (self.history[0..POWER_STATE_HISTORY_SIZE]) |state| {
            counts[@intFromEnum(state.mode)] += 1;
        }

        const threshold = POWER_STATE_HISTORY_SIZE / 2;
        const new_mode = if (counts[@intFromEnum(Mode.power_saving)] > threshold)
            Mode.power_saving
        else if (counts[@intFromEnum(Mode.performance)] > threshold)
            Mode.performance
        else
            Mode.normal;

        if (new_mode != self.mode) {
            self.throttle_count.store(0, .release);
            self.mode = new_mode;
        } else {
            _ = self.throttle_count.fetchAdd(1, .release);
        }

        self.last_update.store(@truncate(now), .release);
    }
};

const ExtensionDependency = struct {
    name: []const u8,
    required_by: []const []const u8,
    alternatives: []const []const u8,
    platform_required: bool = false,
};

const EXTENSION_DEPENDENCIES = [_]ExtensionDependency{
    .{
        .name = VK_KHR_SURFACE_EXTENSION_NAME,
        .required_by = &[_][]const u8{},
        .alternatives = &[_][]const u8{},
        .platform_required = true,
    },
    .{
        .name = VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
        .required_by = &[_][]const u8{},
        .alternatives = &[_][]const u8{VK_EXT_DEBUG_REPORT_EXTENSION_NAME},
        .platform_required = false,
    },
};

fn debugCallback(
    message_severity: c.VkDebugUtilsMessageSeverityFlagBitsEXT,
    message_type: c.VkDebugUtilsMessageTypeFlagsEXT,
    p_callback_data: ?*const c.VkDebugUtilsMessengerCallbackDataEXT,
    p_user_data: ?*anyopaque,
) callconv(.C) c.VkBool32 {
    const instance = @as(*Instance, @ptrCast(@alignCast(p_user_data orelse return c.VK_FALSE)));

    if (p_callback_data) |callback_data| {
        if (message_severity == c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT and
            @import("builtin").mode != .Debug)
        {
            return c.VK_FALSE;
        }

        const msg_id = @abs(callback_data.messageIdNumber);
        if (!instance.debug.message_cache.shouldLog(msg_id)) {
            return c.VK_FALSE;
        }

        const severity = switch (message_severity) {
            c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT => "vrb",
            c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT => "inf",
            c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT => "wrn",
            c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT => "err",
            else => "unknown",
        };

        const msg_type = switch (message_type) {
            c.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT => "general",
            c.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT => "validation",
            c.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT => "performance",
            else => "unknown",
        };

        const msg = std.fmt.bufPrint(&debug_msg_buf, "{s}", .{callback_data.pMessage}) catch "message too long";

        if (message_type == c.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT and
            std.mem.startsWith(u8, msg, "DEBUG PRINTF"))
        {
            if (instance.debug.printf_data) |*data| {
                data.addMessage(msg);
            }
            return c.VK_FALSE;
        }

        logger.debug("vulkan: [{s}] [{s}] {s}", .{ severity, msg_type, msg });

        return if (message_severity == c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) c.VK_TRUE else c.VK_FALSE;
    }

    return c.VK_FALSE;
}

pub const ValidationSeverity = struct {
    verbose: bool = false,
    info: bool = true,
    warning: bool = true,
    severity_error: bool = true,
    perf_warning: bool = true,
};

pub const ValidationMessageType = struct {
    general: bool = true,
    validation: bool = true,
    performance: bool = true,
};

pub const InstanceConfig = struct {
    application_name: [*:0]const u8,
    application_version: u32,
    engine_name: [*:0]const u8,
    engine_version: u32,
    api_version: u32,
    enable_validation: bool = false,
    enable_debug_utils: bool = false,
    enable_surface_extensions: bool = false,
    enable_portability: bool = false,
    validation_features: ValidationFeatures = .{},
    debug_severity: ValidationSeverity = .{},
    debug_message_type: ValidationMessageType = .{},
    required_extensions: [][*:0]const u8 = &.{},
    required_layers: [][*:0]const u8 = &.{},
    allocation_callbacks: ?*const c.VkAllocationCallbacks = null,
};

pub const ValidationFeatures = struct {
    enabled_features: [MAX_VALIDATION_FEATURES]c.VkValidationFeatureEnableEXT = undefined,
    disabled_features: [MAX_VALIDATION_FEATURES]c.VkValidationFeatureDisableEXT = undefined,
    enabled_count: u32 = 0,
    disabled_count: u32 = 0,

    pub fn init() ValidationFeatures {
        var features = ValidationFeatures{};

        features.enabled_features[features.enabled_count] = c.VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT;
        features.enabled_count += 1;

        features.enabled_features[features.enabled_count] = c.VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT;
        features.enabled_count += 1;

        return features;
    }

    pub fn getCreateInfo(self: *const ValidationFeatures) c.VkValidationFeaturesEXT {
        return c.VkValidationFeaturesEXT{
            .sType = c.VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT,
            .pNext = null,
            .enabledValidationFeatureCount = self.enabled_count,
            .pEnabledValidationFeatures = &self.enabled_features,
            .disabledValidationFeatureCount = self.disabled_count,
            .pDisabledValidationFeatures = &self.disabled_features,
        };
    }
};

pub const Instance = struct {
    handle: ?*c.struct_VkInstance_T align(64),
    debug_messenger: c.VkDebugUtilsMessengerEXT,
    api_version: u32,

    debug: struct {
        message_cache: MessageCache align(64),
        printf_data: ?DebugPrintfData,
    } = .{
        .message_cache = .{
            .shards = [_]MessageCacheShard{.{
                .entries = undefined,
                .lookup = .{null} ** (256 / MESSAGE_CACHE_SHARDS),
                .index = std.atomic.Value(u8).init(0),
                .pad = undefined,
            }} ** MESSAGE_CACHE_SHARDS,
        },
        .printf_data = null,
    },

    config: InstanceConfig align(64),

    extensions: struct {
        available: StringCache align(64),
        enabled: StringCache,
        layers: StringCache,
    },

    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,

    power: PowerState align(64) = .{},

    validation_features_ext: ?*c.VkValidationFeaturesEXT = null,

    const ValidationError = error{
        ValidationLayerNotAvailable,
        RequiredExtensionNotAvailable,
        RequiredLayerNotAvailable,
        DebugUtilsNotAvailable,
        APIVersionNotSupported,
    };

    const CreationError = error{
        InstanceCreationFailed,
        DebugMessengerCreationFailed,
        AllocationFailed,
    };

    pub const Error = ValidationError || CreationError;

    pub fn init(config: InstanceConfig, alloc: std.mem.Allocator) !Instance {
        var self: Instance = .{
            .handle = undefined,
            .debug_messenger = undefined,
            .api_version = undefined,
            .config = config,
            .extensions = undefined,
            .allocator = alloc,
            .arena = std.heap.ArenaAllocator.init(alloc),
        };
        errdefer self.arena.deinit();

        self.extensions = .{
            .available = try StringCache.init(alloc, INITIAL_EXTENSION_CAPACITY),
            .enabled = try StringCache.init(alloc, INITIAL_EXTENSION_CAPACITY),
            .layers = try StringCache.init(alloc, INITIAL_LAYER_CAPACITY),
        };
        errdefer {
            self.extensions.available.deinit();
            self.extensions.enabled.deinit();
            self.extensions.layers.deinit();
        }

        try self.validateAPIVersion();
        try self.enumerateExtensionsAndLayers();
        try self.validateRequiredExtensionsAndLayers();
        try self.createInstance();

        if (config.enable_validation) {
            try self.setupDebugMessenger();
            self.debug.printf_data = DebugPrintfData.init();
        }

        return self;
    }

    pub fn deinit(self: *Instance) void {
        if (self.config.enable_validation) {
            const DestroyFn = *const fn (c.VkInstance, c.VkDebugUtilsMessengerEXT, ?*const anyopaque) callconv(.C) void;
            const destroyFn: ?DestroyFn = @ptrCast(c.vkGetInstanceProcAddr(self.handle, "vkDestroyDebugUtilsMessengerEXT"));
            if (destroyFn != null) {
                destroyFn.?(self.handle, self.debug_messenger, null);
            }
        }

        c.vkDestroyInstance(self.handle, null);
        self.extensions.available.deinit();
        self.extensions.enabled.deinit();
        self.extensions.layers.deinit();
    }

    fn validateAPIVersion(self: *Instance) !void {
        var api_version: u32 = undefined;
        if (c.vkEnumerateInstanceVersion(&api_version) != c.VK_SUCCESS) {
            api_version = c.VK_API_VERSION_1_0;
        }

        if (api_version < self.config.api_version) {
            logger.err("vulkan: required Vulkan API version {d}.{d}.{d} not supported. Maximum supported version is {d}.{d}.{d}", .{
                c.VK_VERSION_MAJOR(self.config.api_version),
                c.VK_VERSION_MINOR(self.config.api_version),
                c.VK_VERSION_PATCH(self.config.api_version),
                c.VK_VERSION_MAJOR(api_version),
                c.VK_VERSION_MINOR(api_version),
                c.VK_VERSION_PATCH(api_version),
            });
            return error.APIVersionNotSupported;
        }

        self.api_version = api_version;
        logger.info("vulkan: API version: {d}.{d}.{d}", .{
            c.VK_VERSION_MAJOR(api_version),
            c.VK_VERSION_MINOR(api_version),
            c.VK_VERSION_PATCH(api_version),
        });
    }

    fn enumerateExtensionsAndLayers(self: *Instance) !void {
        var extension_count: u32 = 0;
        _ = c.vkEnumerateInstanceExtensionProperties(null, &extension_count, null);
        const extensions = try self.allocator.alloc(c.VkExtensionProperties, extension_count);
        defer self.allocator.free(extensions);
        _ = c.vkEnumerateInstanceExtensionProperties(null, &extension_count, extensions.ptr);

        logger.debug("vulkan: available extensions:", .{});
        for (extensions) |ext| {
            const name = std.mem.span(@as([*:0]const u8, @ptrCast(&ext.extensionName)));
            logger.debug("  - {s} (spec version: {}.{}.{})", .{
                name,
                c.VK_VERSION_MAJOR(ext.specVersion),
                c.VK_VERSION_MINOR(ext.specVersion),
                c.VK_VERSION_PATCH(ext.specVersion),
            });
            _ = try self.extensions.available.add(name);
        }

        var layer_count: u32 = 0;
        _ = c.vkEnumerateInstanceLayerProperties(&layer_count, null);
        const layers = try self.allocator.alloc(c.VkLayerProperties, layer_count);
        defer self.allocator.free(layers);
        _ = c.vkEnumerateInstanceLayerProperties(&layer_count, layers.ptr);

        logger.debug("vulkan: available layers:", .{});
        for (layers) |layer| {
            const name = std.mem.span(@as([*:0]const u8, @ptrCast(&layer.layerName)));
            logger.debug("  - {s} (spec: {}.{}.{}, impl: {}.{}.{})", .{
                name,
                c.VK_VERSION_MAJOR(layer.specVersion),
                c.VK_VERSION_MINOR(layer.specVersion),
                c.VK_VERSION_PATCH(layer.specVersion),
                c.VK_VERSION_MAJOR(layer.implementationVersion),
                c.VK_VERSION_MINOR(layer.implementationVersion),
                c.VK_VERSION_PATCH(layer.implementationVersion),
            });
            _ = try self.extensions.available.add(name);
            _ = try self.extensions.layers.add(name);
        }
    }

    fn validateRequiredExtensionsAndLayers(self: *Instance) !void {
        var dep_graph = std.ArrayList([]const u8).init(self.allocator);
        defer dep_graph.deinit();

        if (glfw.getRequiredInstanceExtensions()) |glfw_extensions| {
            logger.debug("vulkan: GLFW required extensions:", .{});
            for (glfw_extensions) |ext| {
                const ext_str = std.mem.span(ext);
                logger.debug("  - {s}", .{ext_str});
                try dep_graph.append(ext_str);
            }
        } else {
            logger.err("vulkan: GLFW did not return any required extensions", .{});
            return error.RequiredExtensionNotAvailable;
        }

        if (self.config.enable_validation) {
            try dep_graph.append(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        for (dep_graph.items) |ext| {
            logger.debug("vulkan: checking required extension: {s}", .{ext});
            if (!self.extensions.available.contains(ext)) {
                var found = false;
                for (EXTENSION_DEPENDENCIES) |dep| {
                    if (std.mem.eql(u8, dep.name, ext)) {
                        for (dep.alternatives) |alt| {
                            if (self.extensions.available.contains(alt)) {
                                _ = try self.extensions.enabled.add(alt);
                                found = true;
                                logger.debug("vulkan: using alternative extension {s} for {s}", .{ alt, ext });
                                break;
                            }
                        }
                    }
                }
                if (!found) {
                    logger.err("vulkan: required extension {s} not available", .{ext});
                    return error.RequiredExtensionNotAvailable;
                }
            } else {
                logger.debug("vulkan: enabling extension {s}", .{ext});
                _ = try self.extensions.enabled.add(ext);
            }
        }

        self.extensions.layers.count.store(0, .release);

        if (self.config.enable_validation) {
            const validation_layer = "VK_LAYER_KHRONOS_validation";
            if (!self.extensions.available.contains(validation_layer)) {
                logger.err("vulkan: validation layer not available", .{});
                return error.ValidationLayerNotAvailable;
            }
            _ = try self.extensions.layers.add(validation_layer);
        }

        for (self.config.required_layers) |layer| {
            const layer_name = std.mem.span(layer);
            if (!self.extensions.available.contains(layer_name)) {
                logger.err("vulkan: required layer {s} not available", .{layer_name});
                return error.RequiredLayerNotAvailable;
            }
            _ = try self.extensions.layers.add(layer_name);
        }

        logger.info("vulkan: enabled extensions:", .{});
        const enabled_count = self.extensions.enabled.count.load(.acquire);
        for (self.extensions.enabled.entries[0..enabled_count]) |entry| {
            logger.info("  - {s}", .{entry.data[0..entry.len]});
        }

        logger.info("vulkan: enabled layers:", .{});
        const layer_count = self.extensions.layers.count.load(.acquire);
        for (self.extensions.layers.entries[0..layer_count]) |entry| {
            logger.info("  - {s}", .{entry.data[0..entry.len]});
        }
    }

    fn createInstance(self: *Instance) !void {
        var validation_features_storage: ?c.VkValidationFeaturesEXT = null;

        if (self.config.enable_validation) {
            self.config.validation_features = ValidationFeatures.init();
            validation_features_storage = self.config.validation_features.getCreateInfo();
        }

        const app_info = c.VkApplicationInfo{
            .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = self.config.application_name,
            .applicationVersion = self.config.application_version,
            .pEngineName = self.config.engine_name,
            .engineVersion = self.config.engine_version,
            .apiVersion = self.api_version,
            .pNext = null,
        };

        const enabled_count = self.extensions.enabled.count.load(.acquire);
        const layer_count = self.extensions.layers.count.load(.acquire);

        var extension_names = try self.allocator.alloc([*:0]const u8, enabled_count + @intFromBool(self.config.enable_portability));
        defer self.allocator.free(extension_names);

        var extension_strings = try self.allocator.allocSentinel(u8, (enabled_count + @intFromBool(self.config.enable_portability)) * StringCache.MAX_STRING_LENGTH, 0);
        defer self.allocator.free(extension_strings);

        for (self.extensions.enabled.entries[0..enabled_count], 0..) |entry, i| {
            const start = i * StringCache.MAX_STRING_LENGTH;
            @memcpy(extension_strings[start .. start + entry.len], entry.data[0..entry.len]);
            extension_strings[start + entry.len] = 0;
            extension_names[i] = @ptrCast(extension_strings[start .. start + entry.len :0]);
        }

        if (self.config.enable_portability) {
            const portability_ext = "VK_KHR_portability_enumeration";
            const start = enabled_count * StringCache.MAX_STRING_LENGTH;
            @memcpy(extension_strings[start .. start + portability_ext.len], portability_ext);
            extension_strings[start + portability_ext.len] = 0;
            extension_names[enabled_count] = @ptrCast(extension_strings[start .. start + portability_ext.len :0]);
        }

        var layer_names = try self.allocator.alloc([*:0]const u8, layer_count);
        defer self.allocator.free(layer_names);

        var layer_strings = try self.allocator.allocSentinel(u8, layer_count * StringCache.MAX_STRING_LENGTH, 0);
        defer self.allocator.free(layer_strings);

        for (self.extensions.layers.entries[0..layer_count], 0..) |entry, i| {
            const start = i * StringCache.MAX_STRING_LENGTH;
            @memcpy(layer_strings[start .. start + entry.len], entry.data[0..entry.len]);
            layer_strings[start + entry.len] = 0;
            layer_names[i] = @ptrCast(layer_strings[start .. start + entry.len :0]);
        }

        const create_info = c.VkInstanceCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo = &app_info,
            .enabledExtensionCount = enabled_count + @intFromBool(self.config.enable_portability),
            .ppEnabledExtensionNames = if (enabled_count > 0 or self.config.enable_portability) extension_names.ptr else null,
            .enabledLayerCount = layer_count,
            .ppEnabledLayerNames = if (layer_count > 0) layer_names.ptr else null,
            .flags = if (self.config.enable_portability) c.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR else 0,
            .pNext = if (validation_features_storage) |*vf| @ptrCast(vf) else null,
        };

        if (c.vkCreateInstance(&create_info, self.config.allocation_callbacks, &self.handle) != c.VK_SUCCESS) {
            return error.InstanceCreationFailed;
        }

        if (validation_features_storage) |vf| {
            var mut_vf = vf;
            self.validation_features_ext = @ptrCast(&mut_vf);
        }

        logger.info("vulkan: instance created successfully", .{});
    }

    fn setupDebugMessenger(self: *Instance) !void {
        var severity_flags: c.VkDebugUtilsMessageSeverityFlagsEXT = 0;
        if (self.config.debug_severity.verbose) severity_flags |= c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT;
        if (self.config.debug_severity.info) severity_flags |= c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
        if (self.config.debug_severity.warning) severity_flags |= c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
        if (self.config.debug_severity.severity_error) severity_flags |= c.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

        var message_type_flags: c.VkDebugUtilsMessageTypeFlagsEXT = 0;
        if (self.config.debug_message_type.general) message_type_flags |= c.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
        if (self.config.debug_message_type.validation) message_type_flags |= c.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
        if (self.config.debug_message_type.performance) message_type_flags |= c.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

        const debug_info = c.VkDebugUtilsMessengerCreateInfoEXT{
            .sType = c.VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = severity_flags,
            .messageType = message_type_flags,
            .pfnUserCallback = debugCallback,
            .pUserData = self,
            .flags = 0,
            .pNext = null,
        };

        const vkCreateDebugUtilsMessengerEXT = @as(
            ?*const fn (
                c.VkInstance,
                *const c.VkDebugUtilsMessengerCreateInfoEXT,
                ?*const anyopaque,
                *c.VkDebugUtilsMessengerEXT,
            ) callconv(.C) c.VkResult,
            @ptrCast(c.vkGetInstanceProcAddr(self.handle, "vkCreateDebugUtilsMessengerEXT")),
        );

        if (vkCreateDebugUtilsMessengerEXT) |createFn| {
            if (createFn(self.handle, &debug_info, self.config.allocation_callbacks, &self.debug_messenger) != c.VK_SUCCESS) {
                return error.DebugMessengerCreationFailed;
            }
        } else {
            return error.DebugUtilsNotAvailable;
        }

        logger.info("vulkan: debug messenger setup successfully", .{});
    }

    pub fn supportsExtension(self: *const Instance, extension_name: []const u8) bool {
        return self.extensions.available.contains(extension_name);
    }

    pub fn supportsLayer(self: *const Instance, layer_name: []const u8) bool {
        return self.extensions.layers.contains(layer_name);
    }

    pub fn getAPIVersion(self: *const Instance) u32 {
        return self.api_version;
    }

    pub fn hasValidation(self: *const Instance) bool {
        return self.config.enable_validation;
    }

    pub fn hasDebugUtils(self: *const Instance) bool {
        return self.config.enable_debug_utils;
    }

    pub fn update(self: *Instance) void {
        self.power.update();

        if (self.config.enable_validation) {
            self.config.validation_features = ValidationFeatures.init();

            if (self.power.mode == .power_saving or self.power.state.thermal_throttling) {
                self.config.validation_features.enabled_count = 0;
            }
        }
    }

    fn countRequiredExtensions(config: *const InstanceConfig) u32 {
        var count: u32 = 0;
        inline for (EXTENSION_DEPENDENCIES) |dep| {
            count += @intFromBool(dep.platform_required);
            count += dep.required_by.len;
        }
        count += config.required_extensions.len;
        return count;
    }

    fn getRequiredExtensions(config: InstanceConfig, alloc: std.mem.Allocator) ![]const [*:0]const u8 {
        var extensions = std.ArrayList([*:0]const u8).init(alloc);
        errdefer extensions.deinit();

        const glfw_extensions = glfw.getRequiredInstanceExtensions() orelse {
            logger.err("vulkan: failed to get required instance extensions", .{});
            return error.ExtensionQueryFailed;
        };

        try extensions.appendSlice(glfw_extensions);

        if (config.enable_debug_utils) {
            const debug_utils_ext = "VK_EXT_debug_utils";
            if (try checkExtensionSupport(debug_utils_ext, alloc)) {
                try extensions.append(debug_utils_ext);
            }
        }

        return extensions.toOwnedSlice();
    }

    fn getRequiredLayers(config: InstanceConfig, alloc: std.mem.Allocator) ![]const [*:0]const u8 {
        var layers = std.ArrayList([*:0]const u8).init(alloc);
        errdefer layers.deinit();

        const validation_layer = "VK_LAYER_KHRONOS_validation";

        if (config.enable_validation) {
            if (try checkValidationLayerSupport(validation_layer, alloc)) {
                try layers.append(validation_layer);
            }
        }

        return layers.toOwnedSlice();
    }

    fn checkExtensionSupport(extension_name: [*:0]const u8, alloc: std.mem.Allocator) !bool {
        var extension_count: u32 = 0;
        _ = try Instance.checkResult(c.vkEnumerateInstanceExtensionProperties(null, &extension_count, null));

        const extensions = try alloc.alloc(c.VkExtensionProperties, extension_count);
        defer alloc.free(extensions);

        _ = try Instance.checkResult(c.vkEnumerateInstanceExtensionProperties(null, &extension_count, extensions.ptr));

        for (extensions) |extension| {
            if (std.cstr.cmp(extension_name, &extension.extensionName) == 0) {
                return true;
            }
        }

        return false;
    }

    fn checkValidationLayerSupport(layer_name: [*:0]const u8, alloc: std.mem.Allocator) !bool {
        var layer_count: u32 = 0;
        _ = try Instance.checkResult(c.vkEnumerateInstanceLayerProperties(&layer_count, null));

        const available_layers = try alloc.alloc(c.VkLayerProperties, layer_count);
        defer alloc.free(available_layers);

        _ = try Instance.checkResult(c.vkEnumerateInstanceLayerProperties(&layer_count, available_layers.ptr));

        for (available_layers) |layer| {
            if (std.cstr.cmp(layer_name, &layer.layerName) == 0) {
                return true;
            }
        }

        return false;
    }
};

fn getRequiredExtensions(allocator: std.mem.Allocator) ![]const [*:0]const u8 {
    const glfw_extensions = glfw.getRequiredInstanceExtensions() orelse return error.NoGLFWExtensions;
    const extensions = try allocator.alloc([*:0]const u8, glfw_extensions.len + 1);
    @memcpy(extensions[0..glfw_extensions.len], glfw_extensions);
    extensions[glfw_extensions.len] = "VK_EXT_debug_utils";
    return extensions;
}
