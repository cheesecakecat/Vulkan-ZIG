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
    vulkan_12: c.VkPhysicalDeviceVulkan12Features = .{
        .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = null,
        .descriptorIndexing = c.VK_TRUE,
        .descriptorBindingUniformBufferUpdateAfterBind = c.VK_TRUE,
        .descriptorBindingStorageBufferUpdateAfterBind = c.VK_TRUE,
        .descriptorBindingSampledImageUpdateAfterBind = c.VK_TRUE,
        .descriptorBindingStorageImageUpdateAfterBind = c.VK_TRUE,
        .descriptorBindingUpdateUnusedWhilePending = c.VK_TRUE,
        .timelineSemaphore = c.VK_TRUE,
        .bufferDeviceAddress = c.VK_TRUE,
        .hostQueryReset = c.VK_TRUE,
        .shaderSampledImageArrayNonUniformIndexing = c.VK_TRUE,
        .shaderStorageBufferArrayNonUniformIndexing = c.VK_TRUE,
    },
    vulkan_13: c.VkPhysicalDeviceVulkan13Features = .{
        .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = null,
        .dynamicRendering = c.VK_TRUE,
        .synchronization2 = c.VK_TRUE,
        .maintenance4 = c.VK_TRUE,
    },
    robustness2: c.VkPhysicalDeviceRobustness2FeaturesEXT = .{
        .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
        .pNext = null,
        .robustBufferAccess2 = c.VK_TRUE,
        .robustImageAccess2 = c.VK_TRUE,
        .nullDescriptor = c.VK_TRUE,
    },
    memory: c.VkPhysicalDeviceMemoryPriorityFeaturesEXT = .{
        .sType = c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT,
        .pNext = null,
        .memoryPriority = c.VK_TRUE,
    },
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
                    logger.err("Failed to get {s} queue handle", .{queue.name});
                    return error.QueueCreationFailed;
                }
            }
        }

        var queue_props: [MAX_QUEUE_COUNT]c.VkQueueFamilyProperties = undefined;
        var queue_count: u32 = MAX_QUEUE_COUNT;
        c.vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &queue_count, &queue_props);

        if (queue_count == 0) {
            logger.err("No queue families available", .{});
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
        logger.info("Queue Configuration:", .{});
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
                logger.err("Missing required {s} queue family", .{queue.name});
                return error.MissingRequiredQueueFamily;
            }
        }

        if (unique_families.count() > MAX_QUEUE_COUNT) {
            logger.err("Too many unique queue families: {d} (max {d})", .{ unique_families.count(), MAX_QUEUE_COUNT });
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

        var feature_chain = DeviceFeatureChain{};
        feature_chain.vulkan_13.pNext = &feature_chain.vulkan_12;
        feature_chain.vulkan_12.pNext = &feature_chain.robustness2;
        feature_chain.robustness2.pNext = &feature_chain.memory;

        const enabled_features = if (config.features) |features| features else phys_device.features;

        var enabled_extensions = std.StringHashMap([*:0]const u8).init(allocator);
        errdefer enabled_extensions.deinit();

        for (physical.DEVICE_EXTENSIONS) |ext| {
            if (ext.required) {
                const name = std.mem.span(ext.name);
                if (!try validateExtension(phys_device.handle, name, allocator)) {
                    logger.err("Required extension not supported: {s}", .{name});
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
                    logger.warn("Optional extension not supported: {s}", .{name});
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
            logger.err("Failed to create logical device: {s}", .{getVulkanResultString(result)});
            return error.DeviceCreationFailed;
        }

        const queues = Queues.init(device, phys_device.queue_families, &queue_priorities, phys_device.handle) catch |err| {
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
        logger.info("Logical Device Created:", .{});
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
            logger.err("Failed to wait for device idle during cleanup", .{});
        }

        self.enabled_extensions.deinit();
        self.physical_device.deinit(self.allocator);
        c.vkDestroyDevice(self.handle, null);
    }

    pub fn waitIdle(self: *Device) !void {
        const result = c.vkDeviceWaitIdle(self.handle);
        if (result != c.VK_SUCCESS) {
            logger.err("Failed to wait for device idle: {s}", .{getVulkanResultString(result)});
            return error.DeviceWaitIdleFailed;
        }
    }

    pub fn getGraphicsQueue(self: *const Device) QueueHandle {
        const queue = self.queues.graphics;
        if (!queue.supportsGraphics()) {
            logger.warn("Accessing graphics queue without graphics capability", .{});
        }
        return queue;
    }

    pub fn getComputeQueue(self: *const Device) QueueHandle {
        const queue = self.queues.compute;
        if (!queue.supportsCompute()) {
            logger.warn("Accessing compute queue without compute capability", .{});
        }
        return queue;
    }

    pub fn getTransferQueue(self: *const Device) QueueHandle {
        const queue = self.queues.transfer;
        if (!queue.supportsTransfer()) {
            logger.warn("Accessing transfer queue without transfer capability", .{});
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
