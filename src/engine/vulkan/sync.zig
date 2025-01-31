const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const logger = @import("../core/logger.zig");

pub const SyncObjects = struct {
    image_available_semaphores: []c.VkSemaphore,
    render_finished_semaphores: []c.VkSemaphore,
    in_flight_fences: []c.VkFence,
    device: c.VkDevice,
    allocator: std.mem.Allocator,
    max_frames_in_flight: u32,

    pub fn init(
        device: c.VkDevice,
        max_frames: u32,
        alloc: std.mem.Allocator,
    ) !SyncObjects {
        const image_available_semaphores = try alloc.alloc(c.VkSemaphore, max_frames);
        errdefer alloc.free(image_available_semaphores);

        const render_finished_semaphores = try alloc.alloc(c.VkSemaphore, max_frames);
        errdefer alloc.free(render_finished_semaphores);

        const in_flight_fences = try alloc.alloc(c.VkFence, max_frames);
        errdefer alloc.free(in_flight_fences);

        const semaphore_info = c.VkSemaphoreCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .flags = 0,
            .pNext = null,
        };

        const fence_info = c.VkFenceCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = c.VK_FENCE_CREATE_SIGNALED_BIT,
            .pNext = null,
        };

        var i: u32 = 0;
        while (i < max_frames) : (i += 1) {
            if (c.vkCreateSemaphore(device, &semaphore_info, null, &image_available_semaphores[i]) != c.VK_SUCCESS or
                c.vkCreateSemaphore(device, &semaphore_info, null, &render_finished_semaphores[i]) != c.VK_SUCCESS or
                c.vkCreateFence(device, &fence_info, null, &in_flight_fences[i]) != c.VK_SUCCESS)
            {
                return error.SyncObjectCreationFailed;
            }
        }

        return SyncObjects{
            .image_available_semaphores = image_available_semaphores,
            .render_finished_semaphores = render_finished_semaphores,
            .in_flight_fences = in_flight_fences,
            .device = device,
            .allocator = alloc,
            .max_frames_in_flight = max_frames,
        };
    }

    pub fn deinit(self: *SyncObjects) void {
        var i: u32 = 0;
        while (i < self.max_frames_in_flight) : (i += 1) {
            c.vkDestroySemaphore(self.device, self.image_available_semaphores[i], null);
            c.vkDestroySemaphore(self.device, self.render_finished_semaphores[i], null);
            c.vkDestroyFence(self.device, self.in_flight_fences[i], null);
        }

        self.allocator.free(self.image_available_semaphores);
        self.allocator.free(self.render_finished_semaphores);
        self.allocator.free(self.in_flight_fences);
    }

    pub fn waitForFence(self: *SyncObjects, frame_index: u32) !void {
        _ = c.vkWaitForFences(
            self.device,
            1,
            &self.in_flight_fences[frame_index],
            c.VK_TRUE,
            std.math.maxInt(u64),
        );
    }

    pub fn resetFence(self: *SyncObjects, frame_index: u32) void {
        _ = c.vkResetFences(self.device, 1, &self.in_flight_fences[frame_index]);
    }

    pub const Error = error{
        SyncObjectCreationFailed,
        FenceWaitFailed,
    };
};
