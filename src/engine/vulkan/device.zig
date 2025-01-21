const std = @import("std");
const core = @import("core.zig");

const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});

pub const QueueFamilyIndices = struct {
    graphics_family: ?u32 = null,
    present_family: ?u32 = null,

    pub fn isComplete(self: QueueFamilyIndices) bool {
        return self.graphics_family != null and self.present_family != null;
    }
};

pub fn pickPhysicalDevice(instance: c.VkInstance, surface: c.VkSurfaceKHR) !c.VkPhysicalDevice {
    var device_count: u32 = 0;
    _ = c.vkEnumeratePhysicalDevices(instance, &device_count, null);

    if (device_count == 0) {
        return error.NoGPUFound;
    }

    const devices = try std.heap.c_allocator.alloc(c.VkPhysicalDevice, device_count);
    defer std.heap.c_allocator.free(devices);
    _ = c.vkEnumeratePhysicalDevices(instance, &device_count, devices.ptr);

    for (devices) |device| {
        if (try isDeviceSuitable(device, surface)) {
            return device;
        }
    }

    return error.NoSuitableGPU;
}

pub fn createLogicalDevice(
    physical_device: c.VkPhysicalDevice,
    surface: c.VkSurfaceKHR,
    allocator: std.mem.Allocator,
) !struct { device: c.VkDevice, graphics_queue: c.VkQueue, present_queue: c.VkQueue } {
    const indices = try findQueueFamilies(physical_device, surface);

    var unique_queue_families = std.ArrayList(u32).init(allocator);
    defer unique_queue_families.deinit();

    try unique_queue_families.append(indices.graphics_family.?);
    if (indices.graphics_family.? != indices.present_family.?) {
        try unique_queue_families.append(indices.present_family.?);
    }

    var queue_create_infos = std.ArrayList(c.VkDeviceQueueCreateInfo).init(allocator);
    defer queue_create_infos.deinit();

    const queue_priority = [_]f32{1.0};
    for (unique_queue_families.items) |queue_family| {
        try queue_create_infos.append(.{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueFamilyIndex = queue_family,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
        });
    }

    const device_features = std.mem.zeroes(c.VkPhysicalDeviceFeatures);
    const create_info = c.VkDeviceCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .queueCreateInfoCount = @intCast(queue_create_infos.items.len),
        .pQueueCreateInfos = queue_create_infos.items.ptr,
        .enabledLayerCount = @intCast(core.validation_layers.len),
        .ppEnabledLayerNames = &core.validation_layers,
        .enabledExtensionCount = @intCast(core.required_device_extensions.len),
        .ppEnabledExtensionNames = &core.required_device_extensions,
        .pEnabledFeatures = &device_features,
    };

    var device: c.VkDevice = undefined;
    if (c.vkCreateDevice(physical_device, &create_info, null, &device) != c.VK_SUCCESS) {
        return error.DeviceCreationFailed;
    }

    var graphics_queue: c.VkQueue = undefined;
    var present_queue: c.VkQueue = undefined;
    c.vkGetDeviceQueue(device, indices.graphics_family.?, 0, &graphics_queue);
    c.vkGetDeviceQueue(device, indices.present_family.?, 0, &present_queue);

    return .{
        .device = device,
        .graphics_queue = graphics_queue,
        .present_queue = present_queue,
    };
}

pub fn findQueueFamilies(device: c.VkPhysicalDevice, surface: c.VkSurfaceKHR) !QueueFamilyIndices {
    var indices = QueueFamilyIndices{};

    var queue_family_count: u32 = 0;
    c.vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, null);

    const queue_families = try std.heap.c_allocator.alloc(c.VkQueueFamilyProperties, queue_family_count);
    defer std.heap.c_allocator.free(queue_families);
    c.vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.ptr);

    for (queue_families, 0..) |queue_family, i| {
        if (queue_family.queueFlags & c.VK_QUEUE_GRAPHICS_BIT != 0) {
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

fn isDeviceSuitable(device: c.VkPhysicalDevice, surface: c.VkSurfaceKHR) !bool {
    const indices = try findQueueFamilies(device, surface);
    const extensions_supported = try checkDeviceExtensionSupport(device);

    var swapchain_adequate = false;
    if (extensions_supported) {
        var swapchain_support = try querySwapChainSupport(device, surface);
        defer {
            swapchain_support.deinit();
        }
        swapchain_adequate = swapchain_support.formats.len > 0 and swapchain_support.present_modes.len > 0;
    }

    return indices.isComplete() and extensions_supported and swapchain_adequate;
}

fn checkDeviceExtensionSupport(device: c.VkPhysicalDevice) !bool {
    var extension_count: u32 = undefined;
    _ = c.vkEnumerateDeviceExtensionProperties(device, null, &extension_count, null);

    const available_extensions = try std.heap.c_allocator.alloc(c.VkExtensionProperties, extension_count);
    defer std.heap.c_allocator.free(available_extensions);
    _ = c.vkEnumerateDeviceExtensionProperties(device, null, &extension_count, available_extensions.ptr);

    const required_extensions = core.required_device_extensions;
    var required_extension_count = required_extensions.len;

    for (required_extensions) |required_extension| {
        for (available_extensions) |extension| {
            const available_extension_name = std.mem.sliceTo(&extension.extensionName, 0);
            if (std.mem.eql(u8, std.mem.sliceTo(required_extension, 0), available_extension_name)) {
                required_extension_count -= 1;
                break;
            }
        }
    }

    return required_extension_count == 0;
}

pub const SwapChainSupportDetails = struct {
    capabilities: c.VkSurfaceCapabilitiesKHR,
    formats: []c.VkSurfaceFormatKHR,
    present_modes: []c.VkPresentModeKHR,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *SwapChainSupportDetails) void {
        self.allocator.free(self.formats);
        self.allocator.free(self.present_modes);
    }
};

pub fn querySwapChainSupport(device: c.VkPhysicalDevice, surface: c.VkSurfaceKHR) !SwapChainSupportDetails {
    var details = SwapChainSupportDetails{
        .capabilities = undefined,
        .formats = undefined,
        .present_modes = undefined,
        .allocator = std.heap.c_allocator,
    };

    _ = c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    var format_count: u32 = undefined;
    _ = c.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, null);
    details.formats = try details.allocator.alloc(c.VkSurfaceFormatKHR, format_count);
    _ = c.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, details.formats.ptr);

    var present_mode_count: u32 = undefined;
    _ = c.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, null);
    details.present_modes = try details.allocator.alloc(c.VkPresentModeKHR, present_mode_count);
    _ = c.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, details.present_modes.ptr);

    return details;
}
