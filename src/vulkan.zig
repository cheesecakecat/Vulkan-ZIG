const std = @import("std");
const glfw = @import("mach-glfw");

const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});

pub const required_device_extensions = [_][*:0]const u8{
    "VK_KHR_swapchain",
};

pub const validation_layers = if (@import("builtin").mode == .Debug) [_][*:0]const u8{
    "VK_LAYER_KHRONOS_validation",
} else [_][*:0]const u8{};

pub fn createInstance() !c.VkInstance {
    const app_info = c.VkApplicationInfo{
        .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = null,
        .pApplicationName = "Vulkan Triangle",
        .applicationVersion = c.VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = c.VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = c.VK_API_VERSION_1_3,
    };

    const extensions = try getRequiredExtensions();
    defer extensions.deinit();

    if (validation_layers.len > 0) {
        try checkValidationLayerSupport();
    }

    const create_info = c.VkInstanceCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = @intCast(validation_layers.len),
        .ppEnabledLayerNames = &validation_layers,
        .enabledExtensionCount = @intCast(extensions.items.len),
        .ppEnabledExtensionNames = extensions.items.ptr,
    };

    var instance: c.VkInstance = undefined;
    const result = c.vkCreateInstance(&create_info, null, &instance);
    if (result != c.VK_SUCCESS) {
        return error.InstanceCreationFailed;
    }
    return instance;
}

fn getRequiredExtensions() !std.ArrayList([*:0]const u8) {
    var extensions = std.ArrayList([*:0]const u8).init(std.heap.c_allocator);
    errdefer extensions.deinit();

    const glfw_extensions = glfw.getRequiredInstanceExtensions() orelse return error.GLFWExtensionsNotFound;
    for (glfw_extensions) |ext| {
        try extensions.append(ext);
    }

    if (validation_layers.len > 0) {
        try extensions.append("VK_EXT_debug_utils");
    }

    return extensions;
}

fn checkValidationLayerSupport() !void {
    var layer_count: u32 = undefined;
    _ = c.vkEnumerateInstanceLayerProperties(&layer_count, null);

    const available_layers = try std.heap.c_allocator.alloc(c.VkLayerProperties, layer_count);
    defer std.heap.c_allocator.free(available_layers);

    _ = c.vkEnumerateInstanceLayerProperties(&layer_count, available_layers.ptr);

    for (validation_layers) |layer_name| {
        var layer_found = false;

        for (available_layers) |layer_properties| {
            const available_name = std.mem.sliceTo(&layer_properties.layerName, 0);
            if (std.mem.eql(u8, std.mem.sliceTo(layer_name, 0), available_name)) {
                layer_found = true;
                break;
            }
        }

        if (!layer_found) {
            return error.ValidationLayerNotFound;
        }
    }
}

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

pub const QueueFamilyIndices = struct {
    graphics_family: ?u32 = null,
    present_family: ?u32 = null,

    pub fn isComplete(self: QueueFamilyIndices) bool {
        return self.graphics_family != null and self.present_family != null;
    }
};

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

fn checkDeviceExtensionSupport(device: c.VkPhysicalDevice) !bool {
    var extension_count: u32 = undefined;
    _ = c.vkEnumerateDeviceExtensionProperties(device, null, &extension_count, null);

    const available_extensions = try std.heap.c_allocator.alloc(c.VkExtensionProperties, extension_count);
    defer std.heap.c_allocator.free(available_extensions);
    _ = c.vkEnumerateDeviceExtensionProperties(device, null, &extension_count, available_extensions.ptr);

    const required_extensions = required_device_extensions;
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
