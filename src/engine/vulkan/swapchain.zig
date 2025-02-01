const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const glfw = @import("mach-glfw");
const logger = @import("../core/logger.zig");
const device = @import("device/logical.zig");
const physical = @import("device/physical.zig");
const commands = @import("commands.zig");

fn checkVkResult(result: c.VkResult) !void {
    return switch (result) {
        c.VK_SUCCESS => {},
        c.VK_ERROR_OUT_OF_HOST_MEMORY,
        c.VK_ERROR_OUT_OF_DEVICE_MEMORY,
        => error.OutOfMemory,
        c.VK_ERROR_INITIALIZATION_FAILED => error.InitializationFailed,
        c.VK_ERROR_DEVICE_LOST => error.DeviceLost,
        c.VK_ERROR_SURFACE_LOST_KHR => error.SurfaceLost,
        else => {
            logger.err("vulkan: unexpected error {}", .{result});
            return error.Unknown;
        },
    };
}

pub const SwapchainConfig = struct {
    preferred_formats: []const c.VkFormat,
    preferred_color_space: c.VkColorSpaceKHR,
    preferred_present_modes: []const c.VkPresentModeKHR,
    min_image_count: u32,
    image_usage_flags: c.VkImageUsageFlags,
    transform_flags: c.VkSurfaceTransformFlagBitsKHR,
    composite_alpha: c.VkCompositeAlphaFlagBitsKHR,
    old_swapchain: ?c.VkSwapchainKHR,
    enable_vsync: bool,
    enable_hdr: bool,
    enable_triple_buffering: bool,
    enable_vrr: bool,
    enable_low_latency: bool,
    enable_frame_pacing: bool,
    target_fps: ?u32,
    power_save_mode: PowerSaveMode,
    max_fps: ?u32 = 144,
};

pub const PowerSaveMode = enum {
    Performance,
    Balanced,
    PowerSave,
    Adaptive,
};

pub const DEFAULT_2D_CONFIG = SwapchainConfig{
    .preferred_formats = &[_]c.VkFormat{
        c.VK_FORMAT_B8G8R8A8_SRGB,
        c.VK_FORMAT_R8G8B8A8_SRGB,
        c.VK_FORMAT_A2B10G10R10_UNORM_PACK32,
    },
    .preferred_color_space = c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
    .preferred_present_modes = &[_]c.VkPresentModeKHR{
        c.VK_PRESENT_MODE_MAILBOX_KHR,
        c.VK_PRESENT_MODE_IMMEDIATE_KHR,
        c.VK_PRESENT_MODE_FIFO_RELAXED_KHR,
        c.VK_PRESENT_MODE_FIFO_KHR,
    },
    .min_image_count = 3,
    .image_usage_flags = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
        c.VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        c.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
    .transform_flags = c.VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
    .composite_alpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    .old_swapchain = null,
    .enable_vsync = true,
    .enable_hdr = true,
    .enable_triple_buffering = true,
    .enable_vrr = true,
    .enable_low_latency = true,
    .enable_frame_pacing = true,
    .target_fps = 60,
    .power_save_mode = .Balanced,
};

pub const PERFORMANCE_2D_CONFIG = SwapchainConfig{
    .preferred_formats = &[_]c.VkFormat{
        c.VK_FORMAT_B8G8R8A8_UNORM,
        c.VK_FORMAT_R8G8B8A8_UNORM,
    },
    .preferred_color_space = c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
    .preferred_present_modes = &[_]c.VkPresentModeKHR{
        c.VK_PRESENT_MODE_IMMEDIATE_KHR,
        c.VK_PRESENT_MODE_MAILBOX_KHR,
        c.VK_PRESENT_MODE_FIFO_RELAXED_KHR,
    },
    .min_image_count = 2,
    .image_usage_flags = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
    .transform_flags = c.VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
    .composite_alpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    .old_swapchain = null,
    .enable_vsync = false,
    .enable_hdr = false,
    .enable_triple_buffering = false,
    .enable_vrr = true,
    .enable_low_latency = true,
    .enable_frame_pacing = false,
    .target_fps = null,
    .power_save_mode = .Performance,
};

pub const QUALITY_2D_CONFIG = SwapchainConfig{
    .preferred_formats = &[_]c.VkFormat{
        c.VK_FORMAT_A2B10G10R10_UNORM_PACK32,
        c.VK_FORMAT_R16G16B16A16_SFLOAT,
        c.VK_FORMAT_B8G8R8A8_SRGB,
    },
    .preferred_color_space = c.VK_COLOR_SPACE_HDR10_ST2084_EXT,
    .preferred_present_modes = &[_]c.VkPresentModeKHR{
        c.VK_PRESENT_MODE_FIFO_KHR,
        c.VK_PRESENT_MODE_MAILBOX_KHR,
    },
    .min_image_count = 3,
    .image_usage_flags = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
        c.VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        c.VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        c.VK_IMAGE_USAGE_STORAGE_BIT,
    .transform_flags = c.VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
    .composite_alpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
    .old_swapchain = null,
    .enable_vsync = true,
    .enable_hdr = true,
    .enable_triple_buffering = true,
    .enable_vrr = true,
    .enable_low_latency = false,
    .enable_frame_pacing = true,
    .target_fps = 60,
    .power_save_mode = .Balanced,
};

const SwapchainMetrics = struct {
    frame_index: u64,
    present_time_ns: u64,
    acquire_time_ns: u64,
    frame_time_ns: u64,
    vsync_on: bool,
    dropped_frames: u32,
    refresh_rate_hz: f32,
    actual_present_mode: c.VkPresentModeKHR,
    actual_format: c.VkFormat,
    actual_color_space: c.VkColorSpaceKHR,
    frame_latency_ns: u64,
    tear_factor: f32,
    frame_time_stability: f32,
    vrr_active: bool,
    low_latency_active: bool,
    frame_pacing_error_ns: i64,
    power_state: PowerState,
    gpu_active_time_ns: u64,
    cpu_wait_time_ns: u64,
    power_efficiency: f32,
    is_minimized: bool = false,
    current_width: u32 = 0,
    current_height: u32 = 0,
};

const PowerState = struct {
    current_mode: PowerSaveMode,
    battery_level: ?f32,
    temperature: ?f32,
    throttling: bool,
    power_limit_active: bool,
};

pub const SwapChainSupportDetails = struct {
    capabilities: c.VkSurfaceCapabilitiesKHR,
    formats: []c.VkSurfaceFormatKHR,
    present_modes: []c.VkPresentModeKHR,
    allocator: std.mem.Allocator,

    pub fn init(
        physical_device: c.VkPhysicalDevice,
        surface: c.VkSurfaceKHR,
        alloc: std.mem.Allocator,
    ) !SwapChainSupportDetails {
        var details: SwapChainSupportDetails = undefined;
        details.allocator = alloc;

        try checkVkResult(c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            physical_device,
            surface,
            &details.capabilities,
        ));

        var format_count: u32 = 0;
        try checkVkResult(c.vkGetPhysicalDeviceSurfaceFormatsKHR(
            physical_device,
            surface,
            &format_count,
            null,
        ));

        if (format_count != 0) {
            details.formats = try alloc.alloc(c.VkSurfaceFormatKHR, format_count);
            try checkVkResult(c.vkGetPhysicalDeviceSurfaceFormatsKHR(
                physical_device,
                surface,
                &format_count,
                details.formats.ptr,
            ));
        } else {
            details.formats = &[_]c.VkSurfaceFormatKHR{};
        }

        var present_mode_count: u32 = 0;
        try checkVkResult(c.vkGetPhysicalDeviceSurfacePresentModesKHR(
            physical_device,
            surface,
            &present_mode_count,
            null,
        ));

        if (present_mode_count != 0) {
            details.present_modes = try alloc.alloc(c.VkPresentModeKHR, present_mode_count);
            try checkVkResult(c.vkGetPhysicalDeviceSurfacePresentModesKHR(
                physical_device,
                surface,
                &present_mode_count,
                details.present_modes.ptr,
            ));
        } else {
            details.present_modes = &[_]c.VkPresentModeKHR{};
        }

        return details;
    }

    pub fn deinit(self: *SwapChainSupportDetails) void {
        if (self.formats.len > 0) self.allocator.free(self.formats);
        if (self.present_modes.len > 0) self.allocator.free(self.present_modes);
    }
};

pub const Swapchain = struct {
    handle: c.VkSwapchainKHR,
    images: []c.VkImage,
    image_views: []c.VkImageView,
    framebuffers: []c.VkFramebuffer,
    format: c.VkFormat,
    extent: c.VkExtent2D,
    device: c.VkDevice,
    allocator: std.mem.Allocator,
    config: SwapchainConfig,
    metrics: SwapchainMetrics,
    current_frame: u32,
    surface: c.VkSurfaceKHR,
    physical_device: c.VkPhysicalDevice,
    render_pass: c.VkRenderPass,
    frame_pacing: FramePacing,
    power_manager: PowerManager,
    graphics_queue: c.VkQueue,
    present_queue: c.VkQueue,

    const MAX_FRAMES_IN_FLIGHT = 3;

    const FramePacing = struct {
        target_frame_time_ns: ?u64,
        last_frame_time_ns: u64,
        accumulated_error_ns: i64,
        frame_time_history: [32]u64,
        history_index: usize,

        fn init(target_fps: ?u32) FramePacing {
            return .{
                .target_frame_time_ns = if (target_fps) |fps| @divTrunc(1_000_000_000, fps) else null,
                .last_frame_time_ns = 0,
                .accumulated_error_ns = 0,
                .frame_time_history = [_]u64{0} ** 32,
                .history_index = 0,
            };
        }

        fn updateAndWait(self: *FramePacing, now: u64) void {
            const target = self.target_frame_time_ns orelse return;

            const frame_time = now - self.last_frame_time_ns;
            const @"error" = @as(i64, @intCast(frame_time)) - @as(i64, @intCast(target));
            self.accumulated_error_ns += @"error";

            self.accumulated_error_ns = std.math.clamp(
                self.accumulated_error_ns,
                -target * 2,
                target * 2,
            );

            const wait_time = @max(0, target - frame_time - @divTrunc(self.accumulated_error_ns, 8));

            if (wait_time > 0) {
                std.time.sleep(wait_time);
            }

            self.frame_time_history[self.history_index] = frame_time;
            self.history_index = (self.history_index + 1) % self.frame_time_history.len;

            self.last_frame_time_ns = now;
        }

        fn getFrameTimeStability(self: *const FramePacing) f32 {
            if (self.target_frame_time_ns == null) return 1.0;

            var sum: u64 = 0;
            var count: u32 = 0;

            for (self.frame_time_history) |time| {
                if (time > 0) {
                    sum += time;
                    count += 1;
                }
            }

            if (count == 0) return 1.0;

            const avg = @as(f32, @floatFromInt(sum)) / @as(f32, @floatFromInt(count));
            const target = @as(f32, @floatFromInt(self.target_frame_time_ns.?));

            var variance: f32 = 0;
            for (self.frame_time_history) |time| {
                if (time > 0) {
                    const diff = @as(f32, @floatFromInt(time)) - avg;
                    variance += diff * diff;
                }
            }
            variance /= @as(f32, @floatFromInt(count));

            const stability = 1.0 - std.math.clamp(@sqrt(variance) / target, 0.0, 1.0);

            return stability;
        }
    };

    const PowerManager = struct {
        mode: PowerSaveMode,
        last_power_check: u64,
        power_check_interval: u64,
        state: PowerState,

        fn init(mode: PowerSaveMode) PowerManager {
            return .{
                .mode = mode,
                .last_power_check = 0,
                .power_check_interval = 1_000_000_000,
                .state = .{
                    .current_mode = mode,
                    .battery_level = null,
                    .temperature = null,
                    .throttling = false,
                    .power_limit_active = false,
                },
            };
        }

        fn update(self: *PowerManager, now: u64) void {
            if (now - self.last_power_check < self.power_check_interval) {
                return;
            }

            if (getBatteryLevel()) |level| {
                self.state.battery_level = level;

                if (self.mode == .Adaptive) {
                    self.state.current_mode = switch (level) {
                        0.0...0.2 => .PowerSave,
                        0.2...0.5 => .Balanced,
                        else => .Performance,
                    };
                }
            } else {
                self.state.battery_level = null;
                if (self.mode == .Adaptive) {
                    self.state.current_mode = .Performance;
                }
            }

            if (getGPUTemperature()) |temp| {
                self.state.temperature = temp;
                self.state.throttling = temp > 80.0;

                if (self.state.throttling and
                    self.state.current_mode == .Performance)
                {
                    self.state.current_mode = .Balanced;
                }
            }

            self.last_power_check = now;
        }

        fn shouldSkipFrame(self: *const PowerManager) bool {
            return switch (self.state.current_mode) {
                .PowerSave => self.state.battery_level != null and self.state.battery_level.? < 0.2,
                .Balanced => self.state.throttling,
                else => false,
            };
        }

        fn getBatteryLevel() ?f32 {
            return null;
        }

        fn getGPUTemperature() ?f32 {
            return null;
        }
    };

    const SyncObjects = struct {
        image_available_semaphores: []c.VkSemaphore,
        render_finished_semaphores: []c.VkSemaphore,
        in_flight_fences: []c.VkFence,
    };

    pub fn init(
        logical_device: c.VkDevice,
        queue_indices: physical.QueueFamilyIndices,
        surface: c.VkSurfaceKHR,
        physical_device: c.VkPhysicalDevice,
        width: u32,
        height: u32,
        alloc: std.mem.Allocator,
        render_pass: c.VkRenderPass,
        config: ?SwapchainConfig,
    ) !*Swapchain {
        const self = try alloc.create(Swapchain);
        errdefer alloc.destroy(self);

        var support_details = try SwapChainSupportDetails.init(physical_device, surface, alloc);
        defer support_details.deinit();

        const actual_config = config orelse DEFAULT_2D_CONFIG;

        const surface_format = try chooseBestSurfaceFormat(
            support_details.formats,
            actual_config.preferred_formats,
            actual_config.preferred_color_space,
            actual_config.enable_hdr,
        );

        const present_mode = try chooseBestPresentMode(
            support_details.present_modes,
            actual_config.enable_vsync,
            actual_config.enable_triple_buffering,
            actual_config.enable_vrr,
            actual_config.enable_low_latency,
        );

        const extent = chooseSwapExtent(support_details.capabilities, width, height);

        var image_count = if (actual_config.enable_triple_buffering)
            @max(support_details.capabilities.minImageCount, 3)
        else
            support_details.capabilities.minImageCount + 1;

        if (support_details.capabilities.maxImageCount > 0) {
            image_count = @min(image_count, support_details.capabilities.maxImageCount);
        }

        const queue_family_indices = [_]u32{ queue_indices.graphics_family.?, queue_indices.present_family.? };

        const sharing_mode: c.VkSharingMode = if (queue_indices.graphics_family.? != queue_indices.present_family.?)
            c.VK_SHARING_MODE_CONCURRENT
        else
            c.VK_SHARING_MODE_EXCLUSIVE;

        const swapchain_create_info = c.VkSwapchainCreateInfoKHR{
            .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = surface,
            .minImageCount = image_count,
            .imageFormat = surface_format.format,
            .imageColorSpace = surface_format.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,
            .imageUsage = actual_config.image_usage_flags,
            .imageSharingMode = sharing_mode,
            .queueFamilyIndexCount = if (sharing_mode == c.VK_SHARING_MODE_CONCURRENT) 2 else 0,
            .pQueueFamilyIndices = if (sharing_mode == c.VK_SHARING_MODE_CONCURRENT) &queue_family_indices else null,
            .preTransform = actual_config.transform_flags,
            .compositeAlpha = actual_config.composite_alpha,
            .presentMode = present_mode,
            .clipped = c.VK_TRUE,
            .oldSwapchain = if (actual_config.old_swapchain) |old| old else null,
            .flags = 0,
            .pNext = null,
        };

        var swapchain: c.VkSwapchainKHR = undefined;
        try checkVkResult(c.vkCreateSwapchainKHR(logical_device, &swapchain_create_info, null, &swapchain));
        errdefer c.vkDestroySwapchainKHR(logical_device, swapchain, null);

        const images = try getSwapchainImages(logical_device, swapchain, alloc);
        errdefer alloc.free(images);

        const image_views = try createImageViews(logical_device, images, surface_format.format, alloc);
        errdefer {
            for (image_views) |view| c.vkDestroyImageView(logical_device, view, null);
            alloc.free(image_views);
        }

        const framebuffers = try createFramebuffers(
            logical_device,
            render_pass,
            image_views,
            extent,
            alloc,
        );
        errdefer {
            for (framebuffers) |fb| c.vkDestroyFramebuffer(logical_device, fb, null);
            alloc.free(framebuffers);
        }

        var graphics_queue: c.VkQueue = undefined;
        var present_queue: c.VkQueue = undefined;
        c.vkGetDeviceQueue(logical_device, queue_indices.graphics_family.?, 0, &graphics_queue);
        c.vkGetDeviceQueue(logical_device, queue_indices.present_family.?, 0, &present_queue);

        const metrics = SwapchainMetrics{
            .frame_index = 0,
            .present_time_ns = 0,
            .acquire_time_ns = 0,
            .frame_time_ns = 0,
            .vsync_on = actual_config.enable_vsync,
            .dropped_frames = 0,
            .refresh_rate_hz = 60.0,
            .actual_present_mode = present_mode,
            .actual_format = surface_format.format,
            .actual_color_space = surface_format.colorSpace,
            .frame_latency_ns = 0,
            .tear_factor = 0.0,
            .frame_time_stability = 1.0,
            .vrr_active = false,
            .low_latency_active = false,
            .frame_pacing_error_ns = 0,
            .power_state = .{
                .current_mode = actual_config.power_save_mode,
                .battery_level = null,
                .temperature = null,
                .throttling = false,
                .power_limit_active = false,
            },
            .gpu_active_time_ns = 0,
            .cpu_wait_time_ns = 0,
            .power_efficiency = switch (actual_config.power_save_mode) {
                .PowerSave => 0.9,
                .Balanced => 0.7,
                .Performance => 0.5,
                .Adaptive => 0.7,
            },
            .is_minimized = false,
            .current_width = width,
            .current_height = height,
        };

        self.* = .{
            .handle = swapchain,
            .images = images,
            .image_views = image_views,
            .framebuffers = framebuffers,
            .format = surface_format.format,
            .extent = extent,
            .device = logical_device,
            .allocator = alloc,
            .config = actual_config,
            .metrics = metrics,
            .current_frame = 0,
            .surface = surface,
            .physical_device = physical_device,
            .render_pass = render_pass,
            .frame_pacing = FramePacing.init(actual_config.target_fps),
            .power_manager = PowerManager.init(actual_config.power_save_mode),
            .graphics_queue = graphics_queue,
            .present_queue = present_queue,
        };

        logger.info("swapchain: initialized with {d} images ({d}x{d} {s})", .{
            images.len,
            extent.width,
            extent.height,
            if (actual_config.enable_hdr) "HDR" else "SDR",
        });

        return self;
    }

    pub fn deinit(self: *Swapchain) void {
        logger.info("swapchain: shutting down (processed {d} frames, dropped {d})", .{
            self.metrics.frame_index,
            self.metrics.dropped_frames,
        });

        _ = c.vkDeviceWaitIdle(self.device);

        for (self.framebuffers) |fb| {
            c.vkDestroyFramebuffer(self.device, fb, null);
        }
        self.allocator.free(self.framebuffers);

        for (self.image_views) |view| {
            c.vkDestroyImageView(self.device, view, null);
        }
        self.allocator.free(self.image_views);
        self.allocator.free(self.images);

        c.vkDestroySwapchainKHR(self.device, self.handle, null);

        const avg_frame_time = if (self.metrics.frame_index > 0)
            @as(f32, @floatFromInt(self.metrics.present_time_ns)) / @as(f32, @floatFromInt(self.metrics.frame_index))
        else
            0;

        logger.info("swapchain: final stats:", .{});
        logger.info("  frames: {d}", .{self.metrics.frame_index});
        logger.info("  dropped: {d}", .{self.metrics.dropped_frames});
        logger.info("  avg frame: {d:.2}ms", .{avg_frame_time / 1_000_000.0});
        logger.info("  stability: {d:.2}%", .{self.metrics.frame_time_stability * 100});

        self.allocator.destroy(self);
    }

    pub fn acquireNextImage(self: *Swapchain, image_available_semaphore: c.VkSemaphore) !struct { image_index: u32, should_recreate: bool } {
        if (self.metrics.is_minimized) {
            return .{ .image_index = 0, .should_recreate = false };
        }

        const start_time = std.time.nanoTimestamp();

        var image_index: u32 = undefined;
        const result = c.vkAcquireNextImageKHR(
            self.device,
            self.handle,
            std.math.maxInt(u64),
            image_available_semaphore,
            null,
            &image_index,
        );

        switch (result) {
            c.VK_SUCCESS => {},
            c.VK_SUBOPTIMAL_KHR => return .{ .image_index = image_index, .should_recreate = true },
            c.VK_ERROR_OUT_OF_DATE_KHR => return .{ .image_index = 0, .should_recreate = true },
            else => return error.ImageAcquisitionFailed,
        }

        self.metrics.acquire_time_ns = @as(u64, @intCast(std.time.nanoTimestamp() - start_time));
        return .{ .image_index = image_index, .should_recreate = false };
    }

    pub fn presentImage(self: *Swapchain, image_index: u32, wait_semaphores: []const c.VkSemaphore) !bool {
        if (self.metrics.is_minimized) {
            return false;
        }

        const start_time = std.time.nanoTimestamp();

        const present_info = c.VkPresentInfoKHR{
            .sType = c.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = null,
            .waitSemaphoreCount = @intCast(wait_semaphores.len),
            .pWaitSemaphores = wait_semaphores.ptr,
            .swapchainCount = 1,
            .pSwapchains = &self.handle,
            .pImageIndices = &image_index,
            .pResults = null,
        };

        const result = c.vkQueuePresentKHR(self.present_queue, &present_info);

        const end_time = std.time.nanoTimestamp();
        self.metrics.present_time_ns = @as(u64, @intCast(end_time - start_time));
        self.metrics.frame_time_ns = @as(u64, @intCast(end_time));
        self.metrics.frame_index += 1;

        switch (result) {
            c.VK_SUCCESS => return false,
            c.VK_SUBOPTIMAL_KHR => return true,
            c.VK_ERROR_OUT_OF_DATE_KHR => return true,
            else => {
                logger.err("swapchain: present failed with {any}", .{result});
                return error.PresentationFailed;
            },
        }
    }

    fn chooseBestSurfaceFormat(
        available_formats: []const c.VkSurfaceFormatKHR,
        preferred_formats: []const c.VkFormat,
        preferred_color_space: c.VkColorSpaceKHR,
        enable_hdr: bool,
    ) !c.VkSurfaceFormatKHR {
        if (enable_hdr) {
            const hdr_formats = [_]c.VkFormat{
                c.VK_FORMAT_A2B10G10R10_UNORM_PACK32,
                c.VK_FORMAT_R16G16B16A16_SFLOAT,
            };

            for (available_formats) |format| {
                for (hdr_formats) |hdr_format| {
                    if (format.format == hdr_format and
                        format.colorSpace == c.VK_COLOR_SPACE_HDR10_ST2084_EXT)
                    {
                        return format;
                    }
                }
            }
        }

        for (available_formats) |format| {
            for (preferred_formats) |preferred| {
                if (format.format == preferred and
                    format.colorSpace == preferred_color_space)
                {
                    return format;
                }
            }
        }

        return available_formats[0];
    }

    fn chooseBestPresentMode(
        available_modes: []const c.VkPresentModeKHR,
        enable_vsync: bool,
        enable_triple_buffering: bool,
        enable_vrr: bool,
        enable_low_latency: bool,
    ) !c.VkPresentModeKHR {
        logger.debug("vulkan: choosing present mode:", .{});
        logger.debug("  vsync: {}", .{enable_vsync});
        logger.debug("  triple buffering: {}", .{enable_triple_buffering});
        logger.debug("  vrr: {}", .{enable_vrr});
        logger.debug("  low latency: {}", .{enable_low_latency});
        logger.debug("  available modes:", .{});

        var has_mailbox = false;
        for (available_modes) |mode| {
            const mode_str = switch (mode) {
                c.VK_PRESENT_MODE_IMMEDIATE_KHR => "IMMEDIATE",
                c.VK_PRESENT_MODE_MAILBOX_KHR => "MAILBOX",
                c.VK_PRESENT_MODE_FIFO_KHR => "FIFO",
                c.VK_PRESENT_MODE_FIFO_RELAXED_KHR => "FIFO_RELAXED",
                else => "UNKNOWN",
            };
            logger.debug("    - {any} ({s})", .{ mode, mode_str });
            if (mode == c.VK_PRESENT_MODE_MAILBOX_KHR) {
                has_mailbox = true;
            }
        }

        // if triple buffering is requested but mailbox mode isn't available (common on AMD),
        // log a warning since we'll have to fall back to another mode
        if (enable_triple_buffering and !has_mailbox) {
            logger.warn("vulkan: triple buffering requested but mailbox mode not supported (common on AMD GPUs)", .{});
            logger.warn("vulkan: falling back to standard vsync or immediate mode", .{});
        }

        // if vsync is off, try immediate mode first, then mailbox if triple buffering
        if (!enable_vsync) {
            // if triple buffering is enabled and mailbox is available, prefer that over immediate
            if (enable_triple_buffering and has_mailbox) {
                for (available_modes) |mode| {
                    if (mode == c.VK_PRESENT_MODE_MAILBOX_KHR) {
                        logger.debug("vulkan: selected mailbox present mode", .{});
                        return mode;
                    }
                }
            }

            // otherwise try immediate mode
            for (available_modes) |mode| {
                if (mode == c.VK_PRESENT_MODE_IMMEDIATE_KHR) {
                    logger.debug("vulkan: selected immediate present mode for vsync off", .{});
                    return mode;
                }
            }
        }

        // if vsync is on but VRR is enabled, try relaxed FIFO
        if (enable_vsync and enable_vrr) {
            for (available_modes) |mode| {
                if (mode == c.VK_PRESENT_MODE_FIFO_RELAXED_KHR) {
                    logger.debug("vulkan: selected FIFO relaxed present mode for VRR", .{});
                    return mode;
                }
            }
        }

        // For vsync on, or if no other modes are available, use FIFO
        for (available_modes) |mode| {
            if (mode == c.VK_PRESENT_MODE_FIFO_KHR) {
                logger.debug("vulkan: selected FIFO present mode for vsync", .{});
                return mode;
            }
        }

        logger.err("vulkan: no suitable present mode found, this shouldn't happen", .{});
        return error.UnsupportedPresentMode;
    }

    fn getSwapchainImages(vk_device: c.VkDevice, swapchain: c.VkSwapchainKHR, alloc: std.mem.Allocator) ![]c.VkImage {
        var image_count: u32 = undefined;
        try checkVkResult(c.vkGetSwapchainImagesKHR(vk_device, swapchain, &image_count, null));

        const images = try alloc.alloc(c.VkImage, image_count);
        errdefer alloc.free(images);

        try checkVkResult(c.vkGetSwapchainImagesKHR(vk_device, swapchain, &image_count, images.ptr));
        return images;
    }

    fn createImageViews(
        vk_device: c.VkDevice,
        images: []const c.VkImage,
        format: c.VkFormat,
        alloc: std.mem.Allocator,
    ) ![]c.VkImageView {
        const image_views = try alloc.alloc(c.VkImageView, images.len);
        errdefer alloc.free(image_views);

        for (images, 0..) |image, i| {
            const view_info = c.VkImageViewCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = image,
                .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
                .format = format,
                .components = .{
                    .r = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                    .g = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                    .b = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                    .a = c.VK_COMPONENT_SWIZZLE_IDENTITY,
                },
                .subresourceRange = .{
                    .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
                .flags = 0,
                .pNext = null,
            };

            try checkVkResult(c.vkCreateImageView(vk_device, &view_info, null, &image_views[i]));
        }

        return image_views;
    }

    fn createFramebuffers(
        vk_device: c.VkDevice,
        render_pass: c.VkRenderPass,
        image_views: []const c.VkImageView,
        extent: c.VkExtent2D,
        alloc: std.mem.Allocator,
    ) ![]c.VkFramebuffer {
        const framebuffers = try alloc.alloc(c.VkFramebuffer, image_views.len);
        errdefer alloc.free(framebuffers);

        for (image_views, 0..) |view, i| {
            const framebuffer_info = c.VkFramebufferCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = render_pass,
                .attachmentCount = 1,
                .pAttachments = &view,
                .width = extent.width,
                .height = extent.height,
                .layers = 1,
                .flags = 0,
                .pNext = null,
            };

            try checkVkResult(c.vkCreateFramebuffer(vk_device, &framebuffer_info, null, &framebuffers[i]));
        }

        return framebuffers;
    }

    fn chooseSwapExtent(
        capabilities: c.VkSurfaceCapabilitiesKHR,
        width: u32,
        height: u32,
    ) c.VkExtent2D {
        if (capabilities.currentExtent.width != std.math.maxInt(u32)) {
            return capabilities.currentExtent;
        }

        return c.VkExtent2D{
            .width = std.math.clamp(
                width,
                capabilities.minImageExtent.width,
                capabilities.maxImageExtent.width,
            ),
            .height = std.math.clamp(
                height,
                capabilities.minImageExtent.height,
                capabilities.maxImageExtent.height,
            ),
        };
    }

    pub fn handleWindowState(self: *Swapchain, width: u32, height: u32) void {
        if (width == 0 or height == 0) {
            self.metrics.is_minimized = true;
            return;
        }

        self.metrics.is_minimized = false;

        self.metrics.current_width = width;
        self.metrics.current_height = height;
    }
};
