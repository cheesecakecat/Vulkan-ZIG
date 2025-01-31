const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const logger = @import("../core/logger.zig");

pub const QueueFamilyIndices = struct {
    graphics_family: ?u32 = null,
    present_family: ?u32 = null,

    pub fn isComplete(self: QueueFamilyIndices) bool {
        return self.graphics_family != null and self.present_family != null;
    }
};

pub const Device = struct {
    physical: c.VkPhysicalDevice,
    logical: c.VkDevice,
    graphics_queue: c.VkQueue,
    present_queue: c.VkQueue,
    queue_indices: QueueFamilyIndices,
    allocator: std.mem.Allocator,

    pub fn init(
        instance: c.VkInstance,
        surface: c.VkSurfaceKHR,
        allocator: std.mem.Allocator,
    ) !Device {
        const physical = try pickPhysicalDevice(instance, surface, allocator);
        const queue_indices = try findQueueFamilies(physical, surface, allocator);
        const device = try createLogicalDevice(physical, queue_indices);

        var graphics_queue: c.VkQueue = undefined;
        var present_queue: c.VkQueue = undefined;

        c.vkGetDeviceQueue(device, queue_indices.graphics_family.?, 0, &graphics_queue);
        c.vkGetDeviceQueue(device, queue_indices.present_family.?, 0, &present_queue);

        return Device{
            .physical = physical,
            .logical = device,
            .graphics_queue = graphics_queue,
            .present_queue = present_queue,
            .queue_indices = queue_indices,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Device) void {
        c.vkDestroyDevice(self.logical, null);
    }

    fn pickPhysicalDevice(
        instance: c.VkInstance,
        surface: c.VkSurfaceKHR,
        allocator: std.mem.Allocator,
    ) !c.VkPhysicalDevice {
        var device_count: u32 = 0;
        if (c.vkEnumeratePhysicalDevices(instance, &device_count, null) != c.VK_SUCCESS) {
            return error.PhysicalDeviceEnumerationFailed;
        }

        if (device_count == 0) {
            return error.NoPhysicalDevicesFound;
        }

        const devices = try allocator.alloc(c.VkPhysicalDevice, device_count);
        defer allocator.free(devices);
        _ = c.vkEnumeratePhysicalDevices(instance, &device_count, devices.ptr);

        for (devices) |device| {
            var properties: c.VkPhysicalDeviceProperties = undefined;
            c.vkGetPhysicalDeviceProperties(device, &properties);

            if (properties.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU and
                try isDeviceSuitable(device, surface, allocator))
            {
                return device;
            }
        }

        for (devices) |device| {
            if (try isDeviceSuitable(device, surface, allocator)) {
                return device;
            }
        }

        return error.NoSuitablePhysicalDevice;
    }

    fn findQueueFamilies(
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
            if (family.queueFlags & c.VK_QUEUE_GRAPHICS_BIT != 0) {
                indices.graphics_family = @intCast(i);
            }

            var present_support: c.VkBool32 = c.VK_FALSE;
            _ = c.vkGetPhysicalDeviceSurfaceSupportKHR(device, @intCast(i), surface, &present_support);

            if (present_support == c.VK_TRUE) {
                indices.present_family = @intCast(i);
            }

            if (indices.isComplete()) break;
        }

        return indices;
    }

    fn createLogicalDevice(
        physical_device: c.VkPhysicalDevice,
        queue_indices: QueueFamilyIndices,
    ) !c.VkDevice {
        const queue_priority = [_]f32{1.0};
        var queue_create_infos: [2]c.VkDeviceQueueCreateInfo = undefined;
        var queue_create_info_count: u32 = 1;

        queue_create_infos[0] = .{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = queue_indices.graphics_family.?,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
            .flags = 0,
            .pNext = null,
        };

        if (queue_indices.present_family.? != queue_indices.graphics_family.?) {
            queue_create_infos[1] = .{
                .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = queue_indices.present_family.?,
                .queueCount = 1,
                .pQueuePriorities = &queue_priority,
                .flags = 0,
                .pNext = null,
            };
            queue_create_info_count += 1;
        }

        const device_features = c.VkPhysicalDeviceFeatures{
            .samplerAnisotropy = c.VK_TRUE,
            .fillModeNonSolid = c.VK_TRUE,
            .wideLines = c.VK_TRUE,
            .geometryShader = c.VK_FALSE,
            .tessellationShader = c.VK_FALSE,
        };

        const device_extensions = [_][*:0]const u8{
            "VK_KHR_swapchain",
        };

        const device_create_info = c.VkDeviceCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = queue_create_info_count,
            .pQueueCreateInfos = &queue_create_infos,
            .pEnabledFeatures = &device_features,
            .enabledExtensionCount = device_extensions.len,
            .ppEnabledExtensionNames = &device_extensions,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .flags = 0,
            .pNext = null,
        };

        var device: c.VkDevice = undefined;
        if (c.vkCreateDevice(physical_device, &device_create_info, null, &device) != c.VK_SUCCESS) {
            return error.DeviceCreationFailed;
        }

        return device;
    }

    fn isDeviceSuitable(
        device: c.VkPhysicalDevice,
        surface: c.VkSurfaceKHR,
        allocator: std.mem.Allocator,
    ) !bool {
        const indices = try findQueueFamilies(device, surface, allocator);
        if (!indices.isComplete()) return false;

        const extensions_supported = try checkDeviceExtensionSupport(device, allocator);
        if (!extensions_supported) return false;

        var features: c.VkPhysicalDeviceFeatures = undefined;
        c.vkGetPhysicalDeviceFeatures(device, &features);
        if (features.samplerAnisotropy != c.VK_TRUE) return false;

        return true;
    }

    fn checkDeviceExtensionSupport(
        device: c.VkPhysicalDevice,
        allocator: std.mem.Allocator,
    ) !bool {
        var extension_count: u32 = 0;
        _ = c.vkEnumerateDeviceExtensionProperties(device, null, &extension_count, null);

        const available_extensions = try allocator.alloc(c.VkExtensionProperties, extension_count);
        defer allocator.free(available_extensions);
        _ = c.vkEnumerateDeviceExtensionProperties(device, null, &extension_count, available_extensions.ptr);

        const required_extensions = [_][*:0]const u8{
            c.VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        };

        outer: for (required_extensions) |required| {
            for (available_extensions) |extension| {
                const available_name = @as([*:0]const u8, @ptrCast(&extension.extensionName));
                if (std.mem.eql(u8, std.mem.span(required), std.mem.span(available_name))) {
                    continue :outer;
                }
            }
            return false;
        }

        return true;
    }

    pub const Error = error{
        NoGraphicsDevice,
        NoSuitableDevice,
        DeviceCreationFailed,
        ExtensionNotSupported,
    };
};
