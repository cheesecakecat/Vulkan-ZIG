const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const logger = @import("../core/logger.zig");

const CACHE_LINE_SIZE = 64;
const MAX_FRAMES_HISTORY = 240;
const FRAME_TIME_SMOOTHING = 0.95;
const BATCH_SIZE = 8;
const PREDICTION_WINDOW = 16;
const TIMING_PRECISION_NS = 100_000;
const TARGET_60FPS_TIME_NS: u64 = 16_666_667;
const MIN_FRAME_TIME_NS: u64 = 1_000_000;
const MAX_FRAME_TIME_NS: u64 = 100_000_000;

const FrameTimingStats = struct {
    wait_count: std.atomic.Value(u64) align(CACHE_LINE_SIZE) = std.atomic.Value(u64).init(0),
    total_wait_time_ns: std.atomic.Value(u64) align(CACHE_LINE_SIZE) = std.atomic.Value(u64).init(0),
    max_wait_time_ns: std.atomic.Value(u64) align(CACHE_LINE_SIZE) = std.atomic.Value(u64).init(0),
    last_wait_time_ns: std.atomic.Value(u64) align(CACHE_LINE_SIZE) = std.atomic.Value(u64).init(0),

    frame_times: [MAX_FRAMES_HISTORY]u32 align(CACHE_LINE_SIZE) = [_]u32{0} ** MAX_FRAMES_HISTORY,
    frame_index: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),

    avg_frame_time: f32 align(CACHE_LINE_SIZE) = 0,
    avg_frame_variance: f32 align(CACHE_LINE_SIZE) = 0,

    predicted_next_frame: u64 align(CACHE_LINE_SIZE) = 0,
    prediction_accuracy: f32 align(CACHE_LINE_SIZE) = 0,

    fn predictNextFrameTime(self: *const @This()) u64 {
        var sum: u64 = 0;
        var count: u32 = 0;
        var min_time: u64 = MAX_FRAME_TIME_NS;
        var max_time: u64 = MIN_FRAME_TIME_NS;

        const start_idx = self.frame_index.load(.Monotonic);
        var i: u32 = 0;
        while (i < PREDICTION_WINDOW) : (i += 1) {
            const idx = (start_idx -% i) % MAX_FRAMES_HISTORY;
            const time = self.frame_times[idx];
            if (time > 0) {
                sum += time;
                count += 1;
                min_time = @min(min_time, time);
                max_time = @max(max_time, time);
            }
        }

        if (count == 0) {
            return TARGET_60FPS_TIME_NS;
        }

        const avg = @divTrunc(sum, count);

        if (avg < TARGET_60FPS_TIME_NS) {
            return @min(avg +% (avg >> 2), TARGET_60FPS_TIME_NS);
        } else {
            return @min(avg +% (avg >> 1), MAX_FRAME_TIME_NS);
        }
    }

    fn updateFrameTime(self: *@This(), frame_time: u32) void {
        const clamped_time = @min(@max(frame_time, MIN_FRAME_TIME_NS), MAX_FRAME_TIME_NS);
        const idx = self.frame_index.fetchAdd(1, .Monotonic) % MAX_FRAMES_HISTORY;
        self.frame_times[idx] = clamped_time;

        self.avg_frame_time = self.avg_frame_time * FRAME_TIME_SMOOTHING +
            @as(f32, @floatFromInt(clamped_time)) * (1.0 - FRAME_TIME_SMOOTHING);

        const variance = std.math.pow(f32, @as(f32, @floatFromInt(clamped_time)) - self.avg_frame_time, 2);
        self.avg_frame_variance = self.avg_frame_variance * FRAME_TIME_SMOOTHING +
            variance * (1.0 - FRAME_TIME_SMOOTHING);

        const predicted = self.predicted_next_frame;
        if (predicted > 0) {
            const err = if (predicted > clamped_time)
                predicted - clamped_time
            else
                clamped_time - predicted;

            const accuracy = 1.0 - @min(@as(f32, @floatFromInt(err)) / @as(f32, @floatFromInt(predicted)), 1.0);
            self.prediction_accuracy = self.prediction_accuracy * FRAME_TIME_SMOOTHING +
                accuracy * (1.0 - FRAME_TIME_SMOOTHING);
        }

        self.predicted_next_frame = self.predictNextFrameTime();
    }
};

const SyncConfig = struct {
    max_frames_in_flight: u32,

    enable_validation: bool = false,

    semaphore_flags: c.VkSemaphoreCreateFlags = 0,

    fence_flags: c.VkFenceCreateFlags = c.VK_FENCE_CREATE_SIGNALED_BIT,

    enable_prediction: bool = true,

    batch_size: u32 = BATCH_SIZE,

    initial_timeline_value: u64 = 0,
};

pub const SyncObjects = struct {
    image_available_semaphores: []c.VkSemaphore align(CACHE_LINE_SIZE),
    render_finished_semaphores: []c.VkSemaphore align(CACHE_LINE_SIZE),
    in_flight_fences: []c.VkFence align(CACHE_LINE_SIZE),

    frame_pacing_semaphore: c.VkSemaphore align(CACHE_LINE_SIZE),
    current_timeline_value: std.atomic.Value(u64) align(CACHE_LINE_SIZE),

    device: c.VkDevice,
    allocator: std.mem.Allocator,
    max_frames_in_flight: u32,
    batch_size: u32,
    enable_prediction: bool,

    stats: ?*FrameTimingStats,

    const Self = @This();

    pub fn init(
        device: c.VkDevice,
        config: SyncConfig,
        alloc: std.mem.Allocator,
    ) !Self {
        logger.debug("sync: initializing with {} frames in flight, batch size {}", .{
            config.max_frames_in_flight, config.batch_size,
        });

        if (config.max_frames_in_flight == 0) {
            logger.err("sync: invalid frame count (0)", .{});
            return error.InvalidFrameCount;
        }
        if (device == null) {
            logger.err("sync: invalid device (null)", .{});
            return error.InvalidDevice;
        }
        if (config.batch_size == 0 or config.batch_size > config.max_frames_in_flight) {
            logger.err("sync: invalid batch size ({} for {} frames)", .{ config.batch_size, config.max_frames_in_flight });
            return error.InvalidBatchSize;
        }

        const image_available_semaphores = try alloc.alloc(c.VkSemaphore, config.max_frames_in_flight);
        errdefer alloc.free(image_available_semaphores);

        const render_finished_semaphores = try alloc.alloc(c.VkSemaphore, config.max_frames_in_flight);
        errdefer alloc.free(render_finished_semaphores);

        const in_flight_fences = try alloc.alloc(c.VkFence, config.max_frames_in_flight);
        errdefer alloc.free(in_flight_fences);

        const timeline_semaphore_info = c.VkSemaphoreTypeCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
            .semaphoreType = c.VK_SEMAPHORE_TYPE_TIMELINE,
            .initialValue = config.initial_timeline_value,
            .pNext = null,
        };

        const semaphore_info = c.VkSemaphoreCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .flags = config.semaphore_flags,
            .pNext = &timeline_semaphore_info,
        };

        logger.debug("sync: creating timeline semaphore for frame pacing", .{});
        var frame_pacing_semaphore: c.VkSemaphore = undefined;
        if (c.vkCreateSemaphore(device, &semaphore_info, null, &frame_pacing_semaphore) != c.VK_SUCCESS) {
            logger.err("sync: failed to create timeline semaphore", .{});
            return error.SemaphoreCreationFailed;
        }
        errdefer c.vkDestroySemaphore(device, frame_pacing_semaphore, null);

        const binary_semaphore_info = c.VkSemaphoreCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .flags = config.semaphore_flags,
            .pNext = null,
        };

        const fence_info = c.VkFenceCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = config.fence_flags,
            .pNext = null,
        };

        var i: u32 = 0;
        while (i < config.max_frames_in_flight) : (i += 1) {
            if (c.vkCreateSemaphore(device, &binary_semaphore_info, null, &image_available_semaphores[i]) != c.VK_SUCCESS) {
                return error.SemaphoreCreationFailed;
            }
            errdefer c.vkDestroySemaphore(device, image_available_semaphores[i], null);

            if (c.vkCreateSemaphore(device, &binary_semaphore_info, null, &render_finished_semaphores[i]) != c.VK_SUCCESS) {
                return error.SemaphoreCreationFailed;
            }
            errdefer c.vkDestroySemaphore(device, render_finished_semaphores[i], null);

            if (c.vkCreateFence(device, &fence_info, null, &in_flight_fences[i]) != c.VK_SUCCESS) {
                return error.FenceCreationFailed;
            }
            errdefer c.vkDestroyFence(device, in_flight_fences[i], null);
        }

        const stats = if (config.enable_validation)
            try alloc.create(FrameTimingStats)
        else
            null;

        if (stats) |s| {
            s.* = .{};
        }

        logger.info("sync: initialized successfully", .{});
        logger.debug("sync: configuration:", .{});
        logger.debug("  - Frames in flight: {}", .{config.max_frames_in_flight});
        logger.debug("  - Batch size: {}", .{config.batch_size});
        logger.debug("  - Prediction enabled: {}", .{config.enable_prediction});
        logger.debug("  - Validation enabled: {}", .{config.enable_validation});

        return Self{
            .image_available_semaphores = image_available_semaphores,
            .render_finished_semaphores = render_finished_semaphores,
            .in_flight_fences = in_flight_fences,
            .frame_pacing_semaphore = frame_pacing_semaphore,
            .current_timeline_value = std.atomic.Value(u64).init(config.initial_timeline_value),
            .device = device,
            .allocator = alloc,
            .max_frames_in_flight = config.max_frames_in_flight,
            .batch_size = config.batch_size,
            .enable_prediction = config.enable_prediction,
            .stats = stats,
        };
    }

    pub fn deinit(self: *Self) void {
        logger.debug("sync: cleaning up {} sync objects", .{self.max_frames_in_flight});

        var i: u32 = 0;
        while (i < self.max_frames_in_flight) : (i += 1) {
            c.vkDestroySemaphore(self.device, self.image_available_semaphores[i], null);
            c.vkDestroySemaphore(self.device, self.render_finished_semaphores[i], null);
            c.vkDestroyFence(self.device, self.in_flight_fences[i], null);
        }

        c.vkDestroySemaphore(self.device, self.frame_pacing_semaphore, null);

        self.allocator.free(self.image_available_semaphores);
        self.allocator.free(self.render_finished_semaphores);
        self.allocator.free(self.in_flight_fences);

        if (self.stats) |stats| {
            self.allocator.destroy(stats);
        }

        logger.debug("sync: cleanup complete", .{});
    }

    pub fn waitForFence(self: *Self, frame_index: u32, timeout_ns: u64) !void {
        if (frame_index >= self.max_frames_in_flight) {
            logger.err("sync: invalid frame index {} (max {})", .{ frame_index, self.max_frames_in_flight });
            return error.InvalidFrameIndex;
        }

        const start_time = if (self.stats != null) std.time.nanoTimestamp() else 0;

        const effective_timeout = if (self.enable_prediction and self.stats != null)
            self.stats.?.predictNextFrameTime()
        else
            timeout_ns;

        logger.debug("sync: waiting for fence {} (timeout: {}ns)", .{ frame_index, effective_timeout });

        const result = c.vkWaitForFences(
            self.device,
            1,
            &self.in_flight_fences[frame_index],
            c.VK_TRUE,
            effective_timeout,
        );

        if (result != c.VK_SUCCESS) {
            logger.err("sync: fence wait failed with {}", .{result});
            return switch (result) {
                c.VK_TIMEOUT => error.FenceWaitTimeout,
                c.VK_ERROR_DEVICE_LOST => error.DeviceLost,
                else => error.UnknownError,
            };
        }

        if (self.stats) |stats| {
            const wait_time = @as(u32, @intCast((std.time.nanoTimestamp() - start_time) / TIMING_PRECISION_NS)) * TIMING_PRECISION_NS;
            stats.updateFrameTime(wait_time);

            _ = stats.wait_count.fetchAdd(1, .Monotonic);
            _ = stats.total_wait_time_ns.fetchAdd(wait_time, .Release);
            _ = stats.last_wait_time_ns.store(wait_time, .Release);

            var current_max = stats.max_wait_time_ns.load(.Acquire);
            while (wait_time > current_max) {
                _ = stats.max_wait_time_ns.compareAndSwap(
                    current_max,
                    wait_time,
                    .AcqRel,
                    .Monotonic,
                ) orelse break;
                current_max = stats.max_wait_time_ns.load(.Acquire);
            }
        }
    }

    pub fn waitForFenceBatch(self: *Self, start_index: u32, count: u32, timeout_ns: u64) !void {
        if (start_index + count > self.max_frames_in_flight) {
            logger.err("sync: invalid frame range {}..{} (max {})", .{ start_index, start_index + count, self.max_frames_in_flight });
            return error.InvalidFrameRange;
        }
        if (count == 0 or count > self.batch_size) {
            logger.err("sync: invalid batch count {} (max {})", .{ count, self.batch_size });
            return error.InvalidBatchSize;
        }

        logger.debug("sync: waiting for {} fences starting at {} (timeout: {}ns)", .{ count, start_index, timeout_ns });

        const fences = self.in_flight_fences[start_index..][0..count];
        const result = c.vkWaitForFences(
            self.device,
            @intCast(count),
            fences.ptr,
            c.VK_TRUE,
            timeout_ns,
        );

        if (result != c.VK_SUCCESS) {
            return switch (result) {
                c.VK_TIMEOUT => error.FenceWaitTimeout,
                c.VK_ERROR_DEVICE_LOST => error.DeviceLost,
                else => error.UnknownError,
            };
        }
    }

    pub fn resetFence(self: *Self, frame_index: u32) !void {
        if (frame_index >= self.max_frames_in_flight) {
            logger.err("sync: invalid frame index {} (max {})", .{ frame_index, self.max_frames_in_flight });
            return error.InvalidFrameIndex;
        }

        logger.debug("sync: resetting fence {}", .{frame_index});

        const result = c.vkResetFences(self.device, 1, &self.in_flight_fences[frame_index]);
        if (result != c.VK_SUCCESS) {
            logger.err("sync: fence reset failed with {}", .{result});
            return error.FenceResetFailed;
        }
    }

    pub fn signalFramePacing(self: *Self) !void {
        const next_value = self.current_timeline_value.fetchAdd(1, .Monotonic);

        logger.debug("sync: signaling frame pacing semaphore (value: {})", .{next_value});

        const signal_info = c.VkSemaphoreSignalInfo{
            .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
            .semaphore = self.frame_pacing_semaphore,
            .value = next_value,
            .pNext = null,
        };

        if (c.vkSignalSemaphore(self.device, &signal_info) != c.VK_SUCCESS) {
            return error.SemaphoreSignalFailed;
        }
    }

    pub fn waitFramePacing(self: *Self, value: u64, timeout_ns: u64) !void {
        logger.debug("sync: waiting for frame pacing value {} (timeout: {}ns)", .{ value, timeout_ns });

        const wait_info = c.VkSemaphoreWaitInfo{
            .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .flags = 0,
            .semaphoreCount = 1,
            .pSemaphores = &self.frame_pacing_semaphore,
            .pValues = &value,
            .pNext = null,
        };

        if (c.vkWaitSemaphores(self.device, &wait_info, timeout_ns) != c.VK_SUCCESS) {
            return error.SemaphoreWaitFailed;
        }
    }

    pub fn getStats(self: *const Self) ?FrameTimingStats {
        if (self.stats) |stats| {
            logger.debug("sync: stats snapshot:", .{});
            logger.debug("  - Wait count: {}", .{stats.wait_count.load(.Acquire)});
            logger.debug("  - Avg frame time: {d:.2}ms", .{stats.avg_frame_time / 1_000_000.0});
            logger.debug("  - Max wait time: {d:.2}ms", .{@as(f32, @floatFromInt(stats.max_wait_time_ns.load(.Acquire))) / 1_000_000.0});
        }
        return if (self.stats) |stats| stats.* else null;
    }

    pub fn resetStats(self: *Self) void {
        if (self.stats) |stats| {
            stats.wait_count.store(0, .release);
            stats.total_wait_time_ns.store(0, .release);
            stats.max_wait_time_ns.store(0, .release);
            stats.last_wait_time_ns.store(0, .release);

            @memset(&stats.frame_times, 0);
            stats.frame_index.store(0, .release);
            stats.avg_frame_time = 0;
            stats.avg_frame_variance = 0;
            stats.predicted_next_frame = 0;
            stats.prediction_accuracy = 0;
        }
    }

    pub const Error = error{
        InvalidFrameCount,
        InvalidDevice,
        InvalidFrameIndex,
        InvalidFrameRange,
        InvalidBatchSize,
        SemaphoreCreationFailed,
        FenceCreationFailed,
        FenceWaitTimeout,
        FenceResetFailed,
        DeviceLost,
        SemaphoreSignalFailed,
        SemaphoreWaitFailed,
        UnknownError,
    };
};
