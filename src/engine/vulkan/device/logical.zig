const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const logger = @import("../../core/logger.zig");
const physical = @import("physical.zig");

const MAX_QUEUE_COUNT: u32 = 8;
const DEFAULT_QUEUE_PRIORITY: f32 = 1.0;
const DEDICATED_COMPUTE_PRIORITY: f32 = 0.9;
const DEDICATED_TRANSFER_PRIORITY: f32 = 0.8;

const DeviceFeatureChain = struct {
    vulkan_12: c.VkPhysicalDeviceVulkan12Features,
    vulkan_13: c.VkPhysicalDeviceVulkan13Features,
    robustness2: c.VkPhysicalDeviceRobustness2FeaturesEXT,
    memory: c.VkPhysicalDeviceMemoryPriorityFeaturesEXT,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: Config) !*DeviceFeatureChain {
        const self = try allocator.create(DeviceFeatureChain);
        errdefer allocator.destroy(self);

        self.* = .{
            .allocator = allocator,
            .vulkan_12 = .{
                .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                .pNext = null,
                .descriptorIndexing = c.VK_FALSE,
                .descriptorBindingUniformBufferUpdateAfterBind = c.VK_FALSE,
                .descriptorBindingStorageBufferUpdateAfterBind = c.VK_FALSE,
                .descriptorBindingSampledImageUpdateAfterBind = c.VK_FALSE,
                .descriptorBindingStorageImageUpdateAfterBind = c.VK_FALSE,
                .descriptorBindingUpdateUnusedWhilePending = c.VK_FALSE,
                .timelineSemaphore = c.VK_FALSE,
                .bufferDeviceAddress = c.VK_FALSE,
                .hostQueryReset = c.VK_FALSE,
                .shaderSampledImageArrayNonUniformIndexing = c.VK_FALSE,
                .shaderStorageBufferArrayNonUniformIndexing = c.VK_FALSE,
            },
            .vulkan_13 = .{
                .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                .pNext = null,
                .dynamicRendering = c.VK_FALSE,
                .synchronization2 = c.VK_FALSE,
                .maintenance4 = c.VK_FALSE,
            },
            .robustness2 = .{
                .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
                .pNext = null,
                .robustBufferAccess2 = c.VK_FALSE,
                .robustImageAccess2 = c.VK_FALSE,
                .nullDescriptor = c.VK_FALSE,
            },
            .memory = .{
                .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT,
                .pNext = null,
                .memoryPriority = c.VK_FALSE,
            },
        };

        // Only enable features based on config
        if (config.enable_descriptor_indexing) {
            self.vulkan_12.descriptorIndexing = c.VK_TRUE;
            self.vulkan_12.descriptorBindingUniformBufferUpdateAfterBind = c.VK_TRUE;
            self.vulkan_12.descriptorBindingStorageBufferUpdateAfterBind = c.VK_TRUE;
            self.vulkan_12.descriptorBindingSampledImageUpdateAfterBind = c.VK_TRUE;
            self.vulkan_12.descriptorBindingStorageImageUpdateAfterBind = c.VK_TRUE;
            self.vulkan_12.descriptorBindingUpdateUnusedWhilePending = c.VK_TRUE;
        }

        if (config.enable_timeline_semaphores) {
            self.vulkan_12.timelineSemaphore = c.VK_TRUE;
        }

        if (config.enable_buffer_device_address) {
            self.vulkan_12.bufferDeviceAddress = c.VK_TRUE;
        }

        if (config.enable_host_query_reset) {
            self.vulkan_12.hostQueryReset = c.VK_TRUE;
        }

        if (config.enable_dynamic_rendering) {
            self.vulkan_13.dynamicRendering = c.VK_TRUE;
        }

        if (config.enable_synchronization2) {
            self.vulkan_13.synchronization2 = c.VK_TRUE;
        }

        if (config.enable_maintenance4) {
            self.vulkan_13.maintenance4 = c.VK_TRUE;
        }

        if (config.enable_robustness) {
            self.robustness2.robustBufferAccess2 = c.VK_TRUE;
            self.robustness2.robustImageAccess2 = c.VK_TRUE;
        }

        if (config.enable_null_descriptors) {
            self.robustness2.nullDescriptor = c.VK_TRUE;
        }

        if (config.enable_memory_priority) {
            self.memory.memoryPriority = c.VK_TRUE;
        }

        // Set up the feature chain
        self.vulkan_13.pNext = &self.vulkan_12;
        self.vulkan_12.pNext = &self.robustness2;
        self.robustness2.pNext = &self.memory;

        return self;
    }

    pub fn deinit(self: *DeviceFeatureChain) void {
        self.allocator.destroy(self);
    }
};

const QueueConfig = struct {
    family_index: u32,
    priorities: []const f32,
    flags: c.VkDeviceQueueCreateFlags = 0,
    protected: bool = false,
};

const QueueHandle = struct {
    handle: c.VkQueue,
    family: u32,
    index: u32,
    flags: c.VkQueueFlags,
    priority: f32,
    timestamp_valid_bits: u32,
    min_image_transfer_granularity: c.VkExtent3D,

    pub fn supportsTimestamps(self: QueueHandle) bool {
        return self.timestamp_valid_bits > 0;
    }

    pub fn supportsGraphics(self: QueueHandle) bool {
        return (self.flags & c.VK_QUEUE_GRAPHICS_BIT) != 0;
    }

    pub fn supportsCompute(self: QueueHandle) bool {
        return (self.flags & c.VK_QUEUE_COMPUTE_BIT) != 0;
    }

    pub fn supportsTransfer(self: QueueHandle) bool {
        return (self.flags & c.VK_QUEUE_TRANSFER_BIT) != 0;
    }

    pub fn supportsSparseBinding(self: QueueHandle) bool {
        return (self.flags & c.VK_QUEUE_SPARSE_BINDING_BIT) != 0;
    }

    pub fn supportsProtected(self: QueueHandle) bool {
        return (self.flags & c.VK_QUEUE_PROTECTED_BIT) != 0;
    }
};

pub const Config = struct {
    features: ?c.VkPhysicalDeviceFeatures = null,
    extension_names: []const [*:0]const u8 = &[_][*:0]const u8{},
    queue_configs: []const QueueConfig = &[_]QueueConfig{},
    enable_robustness: bool = true,
    enable_dynamic_rendering: bool = true,
    enable_timeline_semaphores: bool = true,
    enable_synchronization2: bool = true,
    enable_buffer_device_address: bool = true,
    enable_memory_priority: bool = true,
    enable_memory_budget: bool = true,
    enable_descriptor_indexing: bool = true,
    enable_maintenance4: bool = true,
    enable_null_descriptors: bool = true,
    enable_shader_draw_parameters: bool = true,
    enable_host_query_reset: bool = true,
};

const Queues = struct {
    graphics: QueueHandle,
    compute: QueueHandle,
    transfer: QueueHandle,
    present: QueueHandle,

    fn init(
        device: c.VkDevice,
        indices: physical.QueueFamilyIndices,
        priorities: []const f32,
        phys_device: c.VkPhysicalDevice,
    ) !Queues {
        var graphics_queue: c.VkQueue = undefined;
        var compute_queue: c.VkQueue = undefined;
        var transfer_queue: c.VkQueue = undefined;
        var present_queue: c.VkQueue = undefined;

        inline for ([_]struct { name: []const u8, family: ?u32, handle: *c.VkQueue }{
            .{ .name = "graphics", .family = indices.graphics_family, .handle = &graphics_queue },
            .{ .name = "compute", .family = indices.compute_family, .handle = &compute_queue },
            .{ .name = "transfer", .family = indices.transfer_family, .handle = &transfer_queue },
            .{ .name = "present", .family = indices.present_family, .handle = &present_queue },
        }) |queue| {
            if (queue.family) |family| {
                c.vkGetDeviceQueue(device, family, 0, queue.handle);
                if (queue.handle.* == null) {
                    logger.err("device.logical: failed to get {s} queue handle", .{queue.name});
                    return error.QueueCreationFailed;
                }
            }
        }

        var queue_props: [MAX_QUEUE_COUNT]c.VkQueueFamilyProperties = undefined;
        var queue_count: u32 = MAX_QUEUE_COUNT;
        c.vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &queue_count, &queue_props);

        if (queue_count == 0) {
            logger.err("device.logical: no queue families available", .{});
            return error.NoQueueFamiliesAvailable;
        }

        return Queues{
            .graphics = .{
                .handle = graphics_queue,
                .family = indices.graphics_family.?,
                .index = 0,
                .flags = queue_props[indices.graphics_family.?].queueFlags,
                .priority = priorities[0],
                .timestamp_valid_bits = queue_props[indices.graphics_family.?].timestampValidBits,
                .min_image_transfer_granularity = queue_props[indices.graphics_family.?].minImageTransferGranularity,
            },
            .compute = .{
                .handle = compute_queue,
                .family = indices.compute_family.?,
                .index = 0,
                .flags = queue_props[indices.compute_family.?].queueFlags,
                .priority = if (indices.compute_family.? == indices.graphics_family.?)
                    priorities[0]
                else
                    DEDICATED_COMPUTE_PRIORITY,
                .timestamp_valid_bits = queue_props[indices.compute_family.?].timestampValidBits,
                .min_image_transfer_granularity = queue_props[indices.compute_family.?].minImageTransferGranularity,
            },
            .transfer = .{
                .handle = transfer_queue,
                .family = indices.transfer_family.?,
                .index = 0,
                .flags = queue_props[indices.transfer_family.?].queueFlags,
                .priority = if (indices.transfer_family.? == indices.graphics_family.?)
                    priorities[0]
                else
                    DEDICATED_TRANSFER_PRIORITY,
                .timestamp_valid_bits = queue_props[indices.transfer_family.?].timestampValidBits,
                .min_image_transfer_granularity = queue_props[indices.transfer_family.?].minImageTransferGranularity,
            },
            .present = .{
                .handle = present_queue,
                .family = indices.present_family.?,
                .index = 0,
                .flags = queue_props[indices.present_family.?].queueFlags,
                .priority = priorities[0],
                .timestamp_valid_bits = queue_props[indices.present_family.?].timestampValidBits,
                .min_image_transfer_granularity = queue_props[indices.present_family.?].minImageTransferGranularity,
            },
        };
    }

    fn logInfo(self: *const Queues) void {
        logger.info("device.logical: queue Configuration:", .{});
        inline for (.{
            .{ .name = "Graphics", .queue = self.graphics },
            .{ .name = "Compute", .queue = self.compute },
            .{ .name = "Transfer", .queue = self.transfer },
            .{ .name = "Present", .queue = self.present },
        }) |info| {
            logger.info("  {s} Queue:", .{info.name});
            logger.info("    Family: {d}", .{info.queue.family});
            logger.info("    Flags: 0x{X:0>8}", .{info.queue.flags});
            logger.info("    Priority: {d:.2}", .{info.queue.priority});
            logger.info("    Timestamp Valid Bits: {d}", .{info.queue.timestamp_valid_bits});
            logger.info("    Min Image Transfer Granularity: ({d}, {d}, {d})", .{
                info.queue.min_image_transfer_granularity.width,
                info.queue.min_image_transfer_granularity.height,
                info.queue.min_image_transfer_granularity.depth,
            });
        }
    }
};

pub const Device = struct {
    handle: c.VkDevice,
    queues: Queues,
    physical_device: *physical.PhysicalDevice,
    allocator: std.mem.Allocator,
    features: c.VkPhysicalDeviceFeatures,
    enabled_extensions: std.StringHashMap([*:0]const u8),
    api_version: u32,
    graphics_queue: c.VkQueue,
    present_queue: c.VkQueue,
    feature_chain: *DeviceFeatureChain,

    pub fn getPhysicalDeviceHandle(self: *const Device) c.VkPhysicalDevice {
        return self.physical_device.handle;
    }

    pub fn init(
        phys_device: *physical.PhysicalDevice,
        config: Config,
        allocator: std.mem.Allocator,
    ) !Device {
        var unique_families = std.AutoHashMap(u32, void).init(allocator);
        defer unique_families.deinit();

        inline for (.{
            .{ .name = "graphics", .family = phys_device.queue_families.graphics_family },
            .{ .name = "compute", .family = phys_device.queue_families.compute_family },
            .{ .name = "transfer", .family = phys_device.queue_families.transfer_family },
            .{ .name = "present", .family = phys_device.queue_families.present_family },
        }) |queue| {
            if (queue.family) |family| {
                try unique_families.put(family, {});
            } else {
                logger.err("device.logical: missing required {s} queue family", .{queue.name});
                return error.MissingRequiredQueueFamily;
            }
        }

        if (unique_families.count() > MAX_QUEUE_COUNT) {
            logger.err("device.logical: too many unique queue families: {d} (max {d})", .{ unique_families.count(), MAX_QUEUE_COUNT });
            return error.TooManyQueueFamilies;
        }

        const queue_priorities = [_]f32{DEFAULT_QUEUE_PRIORITY} ** MAX_QUEUE_COUNT;
        var queue_create_infos = std.ArrayList(c.VkDeviceQueueCreateInfo).init(allocator);
        defer queue_create_infos.deinit();

        var family_it = unique_families.keyIterator();
        while (family_it.next()) |family| {
            try queue_create_infos.append(.{
                .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = family.*,
                .queueCount = 1,
                .pQueuePriorities = &queue_priorities,
                .flags = 0,
                .pNext = null,
            });
        }

        const feature_chain = try DeviceFeatureChain.init(allocator, config);
        errdefer feature_chain.deinit();

        const enabled_features = if (config.features) |features| features else phys_device.features;

        var enabled_extensions = std.StringHashMap([*:0]const u8).init(allocator);
        errdefer enabled_extensions.deinit();

        for (physical.DEVICE_EXTENSIONS) |ext| {
            if (ext.required) {
                const name = std.mem.span(ext.name);
                if (!try validateExtension(phys_device.handle, name, allocator)) {
                    logger.err("device.logical: required extension not supported: {s}", .{name});
                    return error.RequiredExtensionNotSupported;
                }
                try enabled_extensions.put(name, ext.name);
            }
        }

        for (config.extension_names) |ext_name| {
            const name = std.mem.span(ext_name);
            if (!enabled_extensions.contains(name)) {
                if (try validateExtension(phys_device.handle, name, allocator)) {
                    try enabled_extensions.put(name, ext_name);
                } else {
                    logger.warn("device.logical: optional extension not supported: {s}", .{name});
                }
            }
        }

        var extension_names = try allocator.alloc([*:0]const u8, enabled_extensions.count());
        defer allocator.free(extension_names);

        var i: usize = 0;
        var ext_it = enabled_extensions.valueIterator();
        while (ext_it.next()) |ext| : (i += 1) {
            extension_names[i] = ext.*;
        }

        const device_create_info = c.VkDeviceCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = &feature_chain.vulkan_13,
            .queueCreateInfoCount = @intCast(queue_create_infos.items.len),
            .pQueueCreateInfos = queue_create_infos.items.ptr,
            .pEnabledFeatures = &enabled_features,
            .enabledExtensionCount = @intCast(extension_names.len),
            .ppEnabledExtensionNames = extension_names.ptr,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .flags = 0,
        };

        var device: c.VkDevice = undefined;
        const result = c.vkCreateDevice(phys_device.handle, &device_create_info, null, &device);
        if (result != c.VK_SUCCESS) {
            feature_chain.deinit();
            logger.err("device.logical: failed to create logical device: {s}", .{getVulkanResultString(result)});
            return error.DeviceCreationFailed;
        }

        const queues = Queues.init(device, phys_device.queue_families, &queue_priorities, phys_device.handle) catch |err| {
            feature_chain.deinit();
            c.vkDestroyDevice(device, null);
            return err;
        };

        const result_device = Device{
            .handle = device,
            .queues = queues,
            .physical_device = phys_device,
            .allocator = allocator,
            .features = enabled_features,
            .enabled_extensions = enabled_extensions,
            .api_version = phys_device.properties.apiVersion,
            .graphics_queue = queues.graphics.handle,
            .present_queue = queues.present.handle,
            .feature_chain = feature_chain,
        };

        result_device.logDeviceInfo();

        return result_device;
    }

    fn validateExtension(
        device: c.VkPhysicalDevice,
        name: []const u8,
        allocator: std.mem.Allocator,
    ) !bool {
        var count: u32 = 0;
        _ = c.vkEnumerateDeviceExtensionProperties(device, null, &count, null);

        const properties = try allocator.alloc(c.VkExtensionProperties, count);
        defer allocator.free(properties);

        _ = c.vkEnumerateDeviceExtensionProperties(device, null, &count, properties.ptr);

        for (properties) |prop| {
            const ext_name = std.mem.span(@as([*:0]const u8, @ptrCast(&prop.extensionName)));
            if (std.mem.eql(u8, name, ext_name)) {
                return true;
            }
        }
        return false;
    }

    fn getVulkanResultString(result: c.VkResult) []const u8 {
        return switch (result) {
            c.VK_SUCCESS => "VK_SUCCESS",
            c.VK_ERROR_OUT_OF_HOST_MEMORY => "VK_ERROR_OUT_OF_HOST_MEMORY",
            c.VK_ERROR_OUT_OF_DEVICE_MEMORY => "VK_ERROR_OUT_OF_DEVICE_MEMORY",
            c.VK_ERROR_INITIALIZATION_FAILED => "VK_ERROR_INITIALIZATION_FAILED",
            c.VK_ERROR_EXTENSION_NOT_PRESENT => "VK_ERROR_EXTENSION_NOT_PRESENT",
            c.VK_ERROR_FEATURE_NOT_PRESENT => "VK_ERROR_FEATURE_NOT_PRESENT",
            c.VK_ERROR_TOO_MANY_OBJECTS => "VK_ERROR_TOO_MANY_OBJECTS",
            c.VK_ERROR_DEVICE_LOST => "VK_ERROR_DEVICE_LOST",
            else => "Unknown error",
        };
    }

    fn logDeviceInfo(self: *const Device) void {
        logger.info("device.logical: logical device created:", .{});
        logger.info("  API Version: {d}.{d}.{d}", .{
            c.VK_VERSION_MAJOR(self.api_version),
            c.VK_VERSION_MINOR(self.api_version),
            c.VK_VERSION_PATCH(self.api_version),
        });

        logger.info("  Enabled Features:", .{});
        const features = &self.features;
        inline for (.{
            "robustBufferAccess",
            "fullDrawIndexUint32",
            "imageCubeArray",
            "independentBlend",
        }) |field| {
            if (@field(features, field) == c.VK_TRUE) {
                logger.info("    - {s}", .{field});
            }
        }

        logger.info("  Enabled Extensions:", .{});
        var ext_it = self.enabled_extensions.keyIterator();
        while (ext_it.next()) |ext| {
            logger.info("    - {s}", .{ext.*});
        }

        self.queues.logInfo();
    }

    pub fn deinit(self: *Device) void {
        if (c.vkDeviceWaitIdle(self.handle) != c.VK_SUCCESS) {
            logger.err("device.logical: failed to wait for device idle during cleanup", .{});
        }

        self.feature_chain.deinit();
        self.enabled_extensions.deinit();
        self.allocator.destroy(self.physical_device);
        c.vkDestroyDevice(self.handle, null);
    }

    pub fn waitIdle(self: *Device) !void {
        const result = c.vkDeviceWaitIdle(self.handle);
        if (result != c.VK_SUCCESS) {
            logger.err("device.logical: failed to wait for device idle: {s}", .{getVulkanResultString(result)});
            return error.DeviceWaitIdleFailed;
        }
    }

    pub fn getGraphicsQueue(self: *const Device) QueueHandle {
        const queue = self.queues.graphics;
        if (!queue.supportsGraphics()) {
            logger.warn("device.logical: accessing graphics queue without graphics capability", .{});
        }
        return queue;
    }

    pub fn getComputeQueue(self: *const Device) QueueHandle {
        const queue = self.queues.compute;
        if (!queue.supportsCompute()) {
            logger.warn("device.logical: accessing compute queue without compute capability", .{});
        }
        return queue;
    }

    pub fn getTransferQueue(self: *const Device) QueueHandle {
        const queue = self.queues.transfer;
        if (!queue.supportsTransfer()) {
            logger.warn("device.logical: accessing transfer queue without transfer capability", .{});
        }
        return queue;
    }

    pub fn getPresentQueue(self: *const Device) QueueHandle {
        return self.queues.present;
    }

    pub fn hasExtension(self: *const Device, name: []const u8) bool {
        return self.enabled_extensions.contains(name);
    }

    pub fn hasFeature(self: *const Device, comptime field: []const u8) bool {
        return @field(self.features, field) == c.VK_TRUE;
    }

    pub fn supportsApiVersion(self: *const Device, major: u32, minor: u32) bool {
        const required = c.VK_MAKE_VERSION(major, minor, 0);
        return self.api_version >= required;
    }

    pub fn hasMemoryType(
        self: *const Device,
        type_bits: u32,
        properties: c.VkMemoryPropertyFlags,
    ) bool {
        const memory_props = self.physical_device.memory_properties;
        for (0..memory_props.memoryTypeCount) |i| {
            if (type_bits & (1 << @as(u5, @intCast(i))) != 0 and
                (memory_props.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return true;
            }
        }
        return false;
    }

    pub fn getQueueFamilyIndices(self: *const Device) physical.QueueFamilyIndices {
        return self.physical_device.queue_families;
    }
};

pub const Error = error{
    DeviceCreationFailed,
    DeviceWaitIdleFailed,
    QueueCreationFailed,
    NoQueueFamiliesAvailable,
    MissingRequiredQueueFamily,
    TooManyQueueFamilies,
    RequiredExtensionNotSupported,
};
