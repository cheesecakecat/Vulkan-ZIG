const std = @import("std");
const device = @import("device.zig");
const Window = @import("../window.zig").Window;

const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});

pub const SwapChain = struct {
    device: c.VkDevice,
    handle: c.VkSwapchainKHR,
    images: []c.VkImage,
    image_views: []c.VkImageView,
    framebuffers: []c.VkFramebuffer,
    format: c.VkFormat,
    extent: c.VkExtent2D,
    allocator: std.mem.Allocator,

    pub fn chooseSwapSurfaceFormat(available_formats: []const c.VkSurfaceFormatKHR) c.VkSurfaceFormatKHR {
        for (available_formats) |format| {
            if (format.format == c.VK_FORMAT_B8G8R8A8_SRGB and format.colorSpace == c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return format;
            }
        }
        return available_formats[0];
    }

    pub fn init(
        physical_device: c.VkPhysicalDevice,
        logical_device: c.VkDevice,
        surface: c.VkSurfaceKHR,
        window: *Window,
        allocator: std.mem.Allocator,
        render_pass: c.VkRenderPass,
    ) !SwapChain {
        var swapchain_support = try device.querySwapChainSupport(physical_device, surface);
        defer swapchain_support.deinit();

        const surface_format = chooseSwapSurfaceFormat(swapchain_support.formats);
        const present_mode = chooseSwapPresentMode(swapchain_support.present_modes);
        const extent = chooseSwapExtent(window, swapchain_support.capabilities);

        var image_count = swapchain_support.capabilities.minImageCount + 1;
        if (swapchain_support.capabilities.maxImageCount > 0) {
            image_count = @min(image_count, swapchain_support.capabilities.maxImageCount);
        }

        const indices = try device.findQueueFamilies(physical_device, surface);
        const queue_family_indices = [_]u32{ indices.graphics_family.?, indices.present_family.? };
        const sharing_mode = if (indices.graphics_family.? != indices.present_family.?)
            @as(c_uint, c.VK_SHARING_MODE_CONCURRENT)
        else
            @as(c_uint, c.VK_SHARING_MODE_EXCLUSIVE);

        const create_info = c.VkSwapchainCreateInfoKHR{
            .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = surface,
            .minImageCount = image_count,
            .imageFormat = surface_format.format,
            .imageColorSpace = surface_format.colorSpace,
            .imageExtent = extent,
            .imageArrayLayers = 1,
            .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = sharing_mode,
            .queueFamilyIndexCount = if (sharing_mode == c.VK_SHARING_MODE_CONCURRENT) 2 else 0,
            .pQueueFamilyIndices = if (sharing_mode == c.VK_SHARING_MODE_CONCURRENT) &queue_family_indices else undefined,
            .preTransform = swapchain_support.capabilities.currentTransform,
            .compositeAlpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = present_mode,
            .clipped = c.VK_TRUE,
            .oldSwapchain = null,
        };

        var swapchain: c.VkSwapchainKHR = undefined;
        if (c.vkCreateSwapchainKHR(logical_device, &create_info, null, &swapchain) != c.VK_SUCCESS) {
            return error.SwapChainCreationFailed;
        }

        var actual_image_count: u32 = undefined;
        _ = c.vkGetSwapchainImagesKHR(logical_device, swapchain, &actual_image_count, null);
        const images = try allocator.alloc(c.VkImage, actual_image_count);
        _ = c.vkGetSwapchainImagesKHR(logical_device, swapchain, &actual_image_count, images.ptr);

        const image_views = try createImageViews(logical_device, images, surface_format.format, allocator);
        const framebuffers = try createFramebuffers(logical_device, image_views, render_pass, extent, allocator);

        return SwapChain{
            .device = logical_device,
            .handle = swapchain,
            .images = images,
            .image_views = image_views,
            .framebuffers = framebuffers,
            .format = surface_format.format,
            .extent = extent,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SwapChain) void {
        for (self.framebuffers) |framebuffer| {
            c.vkDestroyFramebuffer(self.device, framebuffer, null);
        }
        self.allocator.free(self.framebuffers);

        for (self.image_views) |view| {
            c.vkDestroyImageView(self.device, view, null);
        }
        self.allocator.free(self.image_views);

        c.vkDestroySwapchainKHR(self.device, self.handle, null);
        self.allocator.free(self.images);
    }
};

fn chooseSwapPresentMode(available_present_modes: []const c.VkPresentModeKHR) c.VkPresentModeKHR {
    for (available_present_modes) |present_mode| {
        if (present_mode == c.VK_PRESENT_MODE_MAILBOX_KHR) {
            return present_mode;
        }
    }
    return c.VK_PRESENT_MODE_FIFO_KHR;
}

fn chooseSwapExtent(window: *Window, capabilities: c.VkSurfaceCapabilitiesKHR) c.VkExtent2D {
    if (capabilities.currentExtent.width != std.math.maxInt(u32)) {
        return capabilities.currentExtent;
    }

    const size = window.getFramebufferSize();
    var actual_extent = c.VkExtent2D{
        .width = size.width,
        .height = size.height,
    };

    actual_extent.width = std.math.clamp(
        actual_extent.width,
        capabilities.minImageExtent.width,
        capabilities.maxImageExtent.width,
    );
    actual_extent.height = std.math.clamp(
        actual_extent.height,
        capabilities.minImageExtent.height,
        capabilities.maxImageExtent.height,
    );

    return actual_extent;
}

fn createImageViews(
    vk_device: c.VkDevice,
    images: []c.VkImage,
    format: c.VkFormat,
    allocator: std.mem.Allocator,
) ![]c.VkImageView {
    const image_views = try allocator.alloc(c.VkImageView, images.len);
    errdefer allocator.free(image_views);

    for (images, 0..) |image, i| {
        const create_info = c.VkImageViewCreateInfo{
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
        };

        if (c.vkCreateImageView(vk_device, &create_info, null, &image_views[i]) != c.VK_SUCCESS) {
            return error.ImageViewCreationFailed;
        }
    }

    return image_views;
}

fn createFramebuffers(
    vk_device: c.VkDevice,
    image_views: []c.VkImageView,
    render_pass: c.VkRenderPass,
    extent: c.VkExtent2D,
    allocator: std.mem.Allocator,
) ![]c.VkFramebuffer {
    const framebuffers = try allocator.alloc(c.VkFramebuffer, image_views.len);
    errdefer allocator.free(framebuffers);

    for (image_views, 0..) |image_view, i| {
        const attachments = [_]c.VkImageView{image_view};
        const framebuffer_info = c.VkFramebufferCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = render_pass,
            .attachmentCount = 1,
            .pAttachments = &attachments,
            .width = extent.width,
            .height = extent.height,
            .layers = 1,
        };

        if (c.vkCreateFramebuffer(vk_device, &framebuffer_info, null, &framebuffers[i]) != c.VK_SUCCESS) {
            return error.FramebufferCreationFailed;
        }
    }

    return framebuffers;
}
