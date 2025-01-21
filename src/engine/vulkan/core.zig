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
