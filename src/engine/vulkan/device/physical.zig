const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const logger = @import("../../core/logger.zig");

const DEVICE_TYPE_WEIGHTS = struct {
    const DISCRETE_GPU: i32 = 1000;
    const INTEGRATED_GPU: i32 = 500;
    const VIRTUAL_GPU: i32 = 250;
    const CPU: i32 = 100;
    const OTHER: i32 = 50;
};

const REQUIRED_FEATURES = c.VkPhysicalDeviceFeatures{
    .samplerAnisotropy = c.VK_TRUE,
    .fillModeNonSolid = c.VK_TRUE,
    .wideLines = c.VK_TRUE,
    .depthClamp = c.VK_TRUE,
    .depthBiasClamp = c.VK_TRUE,
    .depthBounds = c.VK_TRUE,
    .alphaToOne = c.VK_FALSE,
    .multiViewport = c.VK_TRUE,
    .geometryShader = c.VK_FALSE,
    .tessellationShader = c.VK_FALSE,
    .sampleRateShading = c.VK_TRUE,
    .dualSrcBlend = c.VK_TRUE,
    .logicOp = c.VK_TRUE,
    .fragmentStoresAndAtomics = c.VK_TRUE,
    .vertexPipelineStoresAndAtomics = c.VK_TRUE,
    .shaderStorageImageExtendedFormats = c.VK_TRUE,
};

const REQUIRED_EXTENSIONS = [_][*:0]const u8{
    c.VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

const OPTIMAL_LIMITS = struct {
    const MIN_MEMORY_GB: u64 = 0;
    const MIN_UNIFORM_BUFFER_RANGE: u64 = 16384;
    const MIN_STORAGE_BUFFER_RANGE: u64 = 1 << 20;
    const MIN_PUSH_CONSTANTS_SIZE: u32 = 128;
    const MIN_DESCRIPTOR_SETS: u32 = 4;
    const MIN_SAMPLER_ALLOC_COUNT: u32 = 256;
    const MIN_DISCRETE_QUEUE_PRIORITY: f32 = 0.5;
};

pub const DeviceExtension = struct {
    name: [*:0]const u8,
    required: bool,
    alternatives: []const [*:0]const u8 = &[_][*:0]const u8{},
};

pub const DEVICE_EXTENSIONS = [_]DeviceExtension{
    .{ .name = c.VK_KHR_SWAPCHAIN_EXTENSION_NAME, .required = true },
    .{
        .name = "VK_KHR_maintenance1",
        .required = false,
    },
    .{
        .name = "VK_KHR_shader_draw_parameters",
        .required = false,
    },
    .{
        .name = "VK_EXT_memory_budget",
        .required = false,
    },
    .{
        .name = "VK_EXT_memory_priority",
        .required = false,
        .alternatives = &[_][*:0]const u8{"VK_AMD_memory_overallocation_behavior"},
    },
};

const OPTIMAL_FEATURES = c.VkPhysicalDeviceFeatures{
    .samplerAnisotropy = c.VK_TRUE,
    .fillModeNonSolid = c.VK_TRUE,
    .wideLines = c.VK_TRUE,
    .depthClamp = c.VK_TRUE,
    .depthBiasClamp = c.VK_TRUE,
    .depthBounds = c.VK_FALSE,
    .alphaToOne = c.VK_FALSE,
    .multiViewport = c.VK_FALSE,
    .geometryShader = c.VK_FALSE,
    .tessellationShader = c.VK_FALSE,
    .sampleRateShading = c.VK_TRUE,
    .dualSrcBlend = c.VK_FALSE,
    .logicOp = c.VK_FALSE,
    .fragmentStoresAndAtomics = c.VK_FALSE,
    .vertexPipelineStoresAndAtomics = c.VK_FALSE,
    .shaderStorageImageExtendedFormats = c.VK_FALSE,
    .shaderUniformBufferArrayDynamicIndexing = c.VK_TRUE,
    .shaderStorageBufferArrayDynamicIndexing = c.VK_FALSE,
    .shaderClipDistance = c.VK_TRUE,
    .shaderCullDistance = c.VK_FALSE,
    .textureCompressionBC = c.VK_FALSE,
    .occlusionQueryPrecise = c.VK_FALSE,
    .pipelineStatisticsQuery = c.VK_FALSE,
};

const VENDOR_ID = struct {
    const NVIDIA: u32 = 0x10DE;
    const AMD: u32 = 0x1002;
    const INTEL: u32 = 0x8086;
    const ARM: u32 = 0x13B5;
    const QUALCOMM: u32 = 0x5143;
    const APPLE: u32 = 0x106B;
};

const DeviceScore = struct {
    device: c.VkPhysicalDevice,
    properties: c.VkPhysicalDeviceProperties,
    features: c.VkPhysicalDeviceFeatures,
    memory: c.VkPhysicalDeviceMemoryProperties,
    score: i32,
    supported_extensions: std.StringHashMap(void),
    vendor_name: []const u8,
    allocator: std.mem.Allocator,

    pub fn init(device: c.VkPhysicalDevice, allocator: std.mem.Allocator) !DeviceScore {
        var self = DeviceScore{
            .device = device,
            .properties = undefined,
            .features = undefined,
            .memory = undefined,
            .score = 0,
            .supported_extensions = std.StringHashMap(void).init(allocator),
            .vendor_name = undefined,
            .allocator = allocator,
        };

        c.vkGetPhysicalDeviceProperties(device, &self.properties);
        c.vkGetPhysicalDeviceFeatures(device, &self.features);
        c.vkGetPhysicalDeviceMemoryProperties(device, &self.memory);

        self.vendor_name = try getVendorName(device, allocator);
        errdefer allocator.free(self.vendor_name);

        try self.enumerateExtensions();
        self.calculateScore();

        return self;
    }

    pub fn deinit(self: *DeviceScore) void {
        var it = self.supported_extensions.keyIterator();
        while (it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.supported_extensions.deinit();
        self.allocator.free(self.vendor_name);
    }

    fn getVendorName(device: c.VkPhysicalDevice, allocator: std.mem.Allocator) ![]const u8 {
        var props: c.VkPhysicalDeviceProperties = undefined;
        c.vkGetPhysicalDeviceProperties(device, &props);

        return switch (props.vendorID) {
            VENDOR_ID.NVIDIA => try allocator.dupe(u8, "NVIDIA"),
            VENDOR_ID.AMD => try allocator.dupe(u8, "AMD"),
            VENDOR_ID.INTEL => try allocator.dupe(u8, "Intel"),
            VENDOR_ID.ARM => try allocator.dupe(u8, "ARM"),
            VENDOR_ID.QUALCOMM => try allocator.dupe(u8, "Qualcomm"),
            VENDOR_ID.APPLE => try allocator.dupe(u8, "Apple"),
            else => try std.fmt.allocPrint(allocator, "Unknown (0x{X:0>4})", .{props.vendorID}),
        };
    }

    fn enumerateExtensions(self: *DeviceScore) !void {
        var extension_count: u32 = 0;
        _ = c.vkEnumerateDeviceExtensionProperties(self.device, null, &extension_count, null);

        const extensions = try self.supported_extensions.allocator.alloc(
            c.VkExtensionProperties,
            extension_count,
        );
        defer self.supported_extensions.allocator.free(extensions);

        _ = c.vkEnumerateDeviceExtensionProperties(
            self.device,
            null,
            &extension_count,
            extensions.ptr,
        );

        for (extensions) |ext| {
            const raw_name = std.mem.span(@as([*:0]const u8, @ptrCast(&ext.extensionName)));

            if (self.supported_extensions.contains(raw_name)) continue;
            const name = try self.supported_extensions.allocator.dupe(u8, raw_name);
            errdefer self.supported_extensions.allocator.free(name);
            try self.supported_extensions.put(name, {});
        }
    }

    pub fn calculateScore(self: *DeviceScore) void {
        self.score = switch (self.properties.deviceType) {
            c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => DEVICE_TYPE_WEIGHTS.DISCRETE_GPU,
            c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => DEVICE_TYPE_WEIGHTS.INTEGRATED_GPU,
            c.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => DEVICE_TYPE_WEIGHTS.VIRTUAL_GPU,
            c.VK_PHYSICAL_DEVICE_TYPE_CPU => DEVICE_TYPE_WEIGHTS.CPU,
            else => DEVICE_TYPE_WEIGHTS.OTHER,
        };

        switch (self.properties.vendorID) {
            VENDOR_ID.NVIDIA => {
                self.score += 200;
            },
            VENDOR_ID.AMD => {
                self.score += 150;
            },
            VENDOR_ID.INTEL => {
                if (self.properties.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                    self.score += 50;
                }
            },
            else => {},
        }

        var device_local_memory: u64 = 0;
        var host_visible_memory: u64 = 0;
        for (0..self.memory.memoryHeapCount) |i| {
            if (self.memory.memoryHeaps[i].flags & c.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT != 0) {
                device_local_memory += self.memory.memoryHeaps[i].size;
            }

            for (0..self.memory.memoryTypeCount) |j| {
                if (self.memory.memoryTypes[j].heapIndex == i and
                    self.memory.memoryTypes[j].propertyFlags & c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0)
                {
                    host_visible_memory += self.memory.memoryHeaps[i].size;
                }
            }
        }

        const device_memory_gb = @divFloor(device_local_memory, 1024 * 1024 * 1024);
        if (device_memory_gb >= OPTIMAL_LIMITS.MIN_MEMORY_GB) {
            self.score += @intCast(device_memory_gb * 100);
        }

        inline for (@typeInfo(c.VkPhysicalDeviceFeatures).@"struct".fields) |field| {
            const optimal = @field(OPTIMAL_FEATURES, field.name);
            const available = @field(self.features, field.name);
            if (optimal == c.VK_TRUE and available == c.VK_TRUE) {
                self.score += 25;
            }
        }

        for (DEVICE_EXTENSIONS) |ext| {
            if (self.supported_extensions.contains(std.mem.span(ext.name))) {
                self.score += if (ext.required) 100 else 50;
            }
        }

        const limits = self.properties.limits;
        if (limits.maxUniformBufferRange >= OPTIMAL_LIMITS.MIN_UNIFORM_BUFFER_RANGE) self.score += 50;
        if (limits.maxStorageBufferRange >= OPTIMAL_LIMITS.MIN_STORAGE_BUFFER_RANGE) self.score += 50;
        if (limits.maxPushConstantsSize >= OPTIMAL_LIMITS.MIN_PUSH_CONSTANTS_SIZE) self.score += 50;
        if (limits.maxBoundDescriptorSets >= OPTIMAL_LIMITS.MIN_DESCRIPTOR_SETS) self.score += 50;
        if (limits.maxSamplerAllocationCount >= OPTIMAL_LIMITS.MIN_SAMPLER_ALLOC_COUNT) self.score += 50;

        if (limits.timestampComputeAndGraphics == c.VK_TRUE) self.score += 25;
        if (limits.maxImageDimension2D >= 16384) self.score += 25;
    }

    pub fn meetsRequirements(self: *const DeviceScore) bool {
        inline for (@typeInfo(c.VkPhysicalDeviceFeatures).@"struct".fields) |field| {
            const required = @field(REQUIRED_FEATURES, field.name);
            const available = @field(self.features, field.name);
            if (required == c.VK_TRUE and available != c.VK_TRUE) {
                logger.warn("device.physical: missing required feature: {s}", .{field.name});
                return false;
            }
        }

        for (DEVICE_EXTENSIONS) |ext| {
            if (!ext.required) continue;

            const name = std.mem.span(ext.name);
            if (!self.supported_extensions.contains(name)) {
                var has_alternative = false;
                for (ext.alternatives) |alt| {
                    if (self.supported_extensions.contains(std.mem.span(alt))) {
                        has_alternative = true;
                        break;
                    }
                }
                if (!has_alternative) {
                    logger.warn("device.physical: missing required extension: {s}", .{name});
                    return false;
                }
            }
        }

        var device_local_memory: u64 = 0;
        for (0..self.memory.memoryHeapCount) |i| {
            if (self.memory.memoryHeaps[i].flags & c.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT != 0) {
                device_local_memory += self.memory.memoryHeaps[i].size;
            }
        }

        const device_memory_gb = @divFloor(device_local_memory, 1024 * 1024 * 1024);
        if (device_memory_gb < OPTIMAL_LIMITS.MIN_MEMORY_GB) {
            logger.warn("device.physical: insufficient device local memory: {d}GB (minimum {d}GB)", .{ device_memory_gb, OPTIMAL_LIMITS.MIN_MEMORY_GB });
            return false;
        }

        const limits = self.properties.limits;
        if (limits.maxUniformBufferRange < OPTIMAL_LIMITS.MIN_UNIFORM_BUFFER_RANGE) {
            logger.warn("device.physical: insufficient uniform buffer range", .{});
            return false;
        }
        if (limits.maxStorageBufferRange < OPTIMAL_LIMITS.MIN_STORAGE_BUFFER_RANGE) {
            logger.warn("device: insufficient storage buffer range", .{});
            return false;
        }
        if (limits.maxPushConstantsSize < OPTIMAL_LIMITS.MIN_PUSH_CONSTANTS_SIZE) {
            logger.warn("device: insufficient push constant size", .{});
            return false;
        }

        return true;
    }

    pub fn logSpecs(self: *const DeviceScore) void {
        logger.info("device.physical: physical Device: {s} ({s})", .{
            self.properties.deviceName,
            self.vendor_name,
        });

        logger.info("  Type: {s}", .{switch (self.properties.deviceType) {
            c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => "Discrete GPU",
            c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => "Integrated GPU",
            c.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU => "Virtual GPU",
            c.VK_PHYSICAL_DEVICE_TYPE_CPU => "CPU",
            else => "Other",
        }});

        logger.info("  API Version: {}.{}.{}", .{
            c.VK_VERSION_MAJOR(self.properties.apiVersion),
            c.VK_VERSION_MINOR(self.properties.apiVersion),
            c.VK_VERSION_PATCH(self.properties.apiVersion),
        });

        logger.info("  Driver Version: {}.{}.{}", .{
            c.VK_VERSION_MAJOR(self.properties.driverVersion),
            c.VK_VERSION_MINOR(self.properties.driverVersion),
            c.VK_VERSION_PATCH(self.properties.driverVersion),
        });

        var device_local_memory: u64 = 0;
        var host_visible_memory: u64 = 0;
        for (0..self.memory.memoryHeapCount) |i| {
            if (self.memory.memoryHeaps[i].flags & c.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT != 0) {
                device_local_memory += self.memory.memoryHeaps[i].size;
            }

            for (0..self.memory.memoryTypeCount) |j| {
                if (self.memory.memoryTypes[j].heapIndex == i and
                    self.memory.memoryTypes[j].propertyFlags & c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT != 0)
                {
                    host_visible_memory += self.memory.memoryHeaps[i].size;
                    break;
                }
            }
        }

        logger.info("  Device Local Memory: {d:.2} GB", .{@as(f64, @floatFromInt(device_local_memory)) / (1024 * 1024 * 1024)});
        logger.info("  Host Visible Memory: {d:.2} GB", .{@as(f64, @floatFromInt(host_visible_memory)) / (1024 * 1024 * 1024)});

        const limits = self.properties.limits;
        logger.info("  Device Limits:", .{});
        logger.info("    Max Image Size 2D: {d}", .{limits.maxImageDimension2D});
        logger.info("    Max Uniform Buffer Range: {d} bytes", .{limits.maxUniformBufferRange});
        logger.info("    Max Storage Buffer Range: {d} bytes", .{limits.maxStorageBufferRange});
        logger.info("    Max Push Constants Size: {d} bytes", .{limits.maxPushConstantsSize});
        logger.info("    Max Memory Allocation Count: {d}", .{limits.maxMemoryAllocationCount});
        logger.info("    Max Sampler Allocation Count: {d}", .{limits.maxSamplerAllocationCount});
        logger.info("    Max Bound Descriptor Sets: {d}", .{limits.maxBoundDescriptorSets});

        logger.info("  Supported Extensions:", .{});
        var it = self.supported_extensions.keyIterator();
        while (it.next()) |key| {
            logger.info("    - {s}", .{key.*});
        }

        logger.info("  Final Score: {d}", .{self.score});
    }
};

pub const PhysicalDevice = struct {
    handle: c.VkPhysicalDevice,
    properties: c.VkPhysicalDeviceProperties,
    features: c.VkPhysicalDeviceFeatures,
    memory_properties: c.VkPhysicalDeviceMemoryProperties,
    queue_families: QueueFamilyIndices,

    pub fn selectBest(
        instance: c.VkInstance,
        surface: c.VkSurfaceKHR,
        allocator: std.mem.Allocator,
    ) !*PhysicalDevice {
        var device_count: u32 = 0;
        if (c.vkEnumeratePhysicalDevices(instance, &device_count, null) != c.VK_SUCCESS) {
            return error.PhysicalDeviceEnumerationFailed;
        }

        if (device_count == 0) {
            logger.err("device.physical: no physical devices found", .{});
            return error.NoPhysicalDevicesFound;
        }

        const devices = try allocator.alloc(c.VkPhysicalDevice, device_count);
        defer allocator.free(devices);

        if (c.vkEnumeratePhysicalDevices(instance, &device_count, devices.ptr) != c.VK_SUCCESS) {
            return error.PhysicalDeviceEnumerationFailed;
        }

        logger.info("device.physical: found {d} physical device(s)", .{device_count});

        var device_scores = try allocator.alloc(DeviceScore, device_count);
        defer allocator.free(device_scores);
        defer {
            for (device_scores) |*score| {
                score.deinit();
            }
        }

        var best_score: i32 = -1;
        var best_device: ?*DeviceScore = null;

        for (devices, 0..) |device, i| {
            device_scores[i] = try DeviceScore.init(device, allocator);

            if (!device_scores[i].meetsRequirements()) {
                logger.warn("device.physical: device '{s}' does not meet requirements", .{device_scores[i].properties.deviceName});
                continue;
            }

            device_scores[i].logSpecs();

            if (device_scores[i].score > best_score) {
                best_score = device_scores[i].score;
                best_device = &device_scores[i];
            }
        }

        if (best_device == null) {
            logger.err("device.physical: no suitable physical device found", .{});
            return error.NoSuitablePhysicalDevice;
        }

        logger.info("device.physical: selected physical device: {s}", .{best_device.?.properties.deviceName});

        const result = try allocator.create(PhysicalDevice);
        errdefer allocator.destroy(result);

        result.* = PhysicalDevice{
            .handle = best_device.?.device,
            .properties = best_device.?.properties,
            .features = best_device.?.features,
            .memory_properties = best_device.?.memory,
            .queue_families = try QueueFamilyIndices.find(best_device.?.device, surface, allocator),
        };

        return result;
    }
};

pub const QueueFamilyIndices = struct {
    graphics_family: ?u32 = null,
    present_family: ?u32 = null,
    compute_family: ?u32 = null,
    transfer_family: ?u32 = null,

    pub fn isComplete(self: QueueFamilyIndices) bool {
        return self.graphics_family != null and
            self.present_family != null and
            self.compute_family != null and
            self.transfer_family != null;
    }

    pub fn find(
        device: c.VkPhysicalDevice,
        surface: c.VkSurfaceKHR,
        allocator: std.mem.Allocator,
    ) !QueueFamilyIndices {
        var indices = QueueFamilyIndices{};

        var queue_family_count: u32 = 0;
        c.vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, null);

        const queue_families = try allocator.alloc(c.VkQueueFamilyProperties, queue_family_count);
        defer allocator.free(queue_families);

        c.vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.ptr);

        for (queue_families, 0..) |family, i| {
            const idx: u32 = @intCast(i);

            const supports_graphics = family.queueFlags & c.VK_QUEUE_GRAPHICS_BIT != 0;
            const supports_compute = family.queueFlags & c.VK_QUEUE_COMPUTE_BIT != 0;
            const supports_transfer = family.queueFlags & c.VK_QUEUE_TRANSFER_BIT != 0;

            var present_support: c.VkBool32 = c.VK_FALSE;
            _ = c.vkGetPhysicalDeviceSurfaceSupportKHR(device, idx, surface, &present_support);

            if (supports_graphics and supports_compute and supports_transfer and present_support == c.VK_TRUE) {
                indices = .{
                    .graphics_family = idx,
                    .present_family = idx,
                    .compute_family = idx,
                    .transfer_family = idx,
                };
                logger.info("device.physical: found unified queue family at index {d}", .{idx});
                return indices;
            }
        }

        for (queue_families, 0..) |family, i| {
            const idx: u32 = @intCast(i);

            if (indices.graphics_family == null and family.queueFlags & c.VK_QUEUE_GRAPHICS_BIT != 0) {
                indices.graphics_family = idx;
                logger.info("device.physical: found graphics queue family at index {d}", .{idx});
            }

            if (indices.compute_family == null and family.queueFlags & c.VK_QUEUE_COMPUTE_BIT != 0) {
                if (family.queueFlags & c.VK_QUEUE_GRAPHICS_BIT == 0) {
                    indices.compute_family = idx;
                    logger.info("device.physical: found dedicated compute queue family at index {d}", .{idx});
                }
            }

            if (indices.transfer_family == null and family.queueFlags & c.VK_QUEUE_TRANSFER_BIT != 0) {
                if (family.queueFlags & (c.VK_QUEUE_GRAPHICS_BIT | c.VK_QUEUE_COMPUTE_BIT) == 0) {
                    indices.transfer_family = idx;
                    logger.info("device.physical: found dedicated transfer queue family at index {d}", .{idx});
                }
            }

            var present_support: c.VkBool32 = c.VK_FALSE;
            _ = c.vkGetPhysicalDeviceSurfaceSupportKHR(device, idx, surface, &present_support);

            if (indices.present_family == null and present_support == c.VK_TRUE) {
                indices.present_family = idx;
                logger.info("device.physical: found present queue family at index {d}", .{idx});
            }
        }

        if (indices.compute_family == null) {
            indices.compute_family = indices.graphics_family;
            logger.info("device.physical: using graphics queue for compute operations", .{});
        }

        if (indices.transfer_family == null) {
            indices.transfer_family = indices.graphics_family;
            logger.info("device.physical: using graphics queue for transfer operations", .{});
        }

        if (!indices.isComplete()) {
            logger.err("device.physical: failed to find required queue families", .{});
            return error.NoSuitableQueueFamilies;
        }

        return indices;
    }
};

pub const Error = error{
    PhysicalDeviceEnumerationFailed,
    NoPhysicalDevicesFound,
    NoSuitablePhysicalDevice,
    NoSuitableQueueFamilies,
};
