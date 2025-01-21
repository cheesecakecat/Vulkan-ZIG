const std = @import("std");

const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});

pub const MAX_FRAMES_IN_FLIGHT = 2;

pub const SyncObjects = struct {
    device: c.VkDevice,
    image_available_semaphores: []c.VkSemaphore,
    render_finished_semaphores: []c.VkSemaphore,
    in_flight_fences: []c.VkFence,
    allocator: std.mem.Allocator,

    pub fn init(device: c.VkDevice, allocator: std.mem.Allocator) !SyncObjects {
        const image_available_semaphores = try allocator.alloc(c.VkSemaphore, MAX_FRAMES_IN_FLIGHT);
        errdefer allocator.free(image_available_semaphores);

        const render_finished_semaphores = try allocator.alloc(c.VkSemaphore, MAX_FRAMES_IN_FLIGHT);
        errdefer allocator.free(render_finished_semaphores);

        const in_flight_fences = try allocator.alloc(c.VkFence, MAX_FRAMES_IN_FLIGHT);
        errdefer allocator.free(in_flight_fences);

        const semaphore_info = c.VkSemaphoreCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
        };

        const fence_info = c.VkFenceCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = null,
            .flags = c.VK_FENCE_CREATE_SIGNALED_BIT,
        };

        var i: usize = 0;
        while (i < MAX_FRAMES_IN_FLIGHT) : (i += 1) {
            if (c.vkCreateSemaphore(device, &semaphore_info, null, &image_available_semaphores[i]) != c.VK_SUCCESS or
                c.vkCreateSemaphore(device, &semaphore_info, null, &render_finished_semaphores[i]) != c.VK_SUCCESS or
                c.vkCreateFence(device, &fence_info, null, &in_flight_fences[i]) != c.VK_SUCCESS)
            {
                return error.SyncObjectCreationFailed;
            }
        }

        return SyncObjects{
            .device = device,
            .image_available_semaphores = image_available_semaphores,
            .render_finished_semaphores = render_finished_semaphores,
            .in_flight_fences = in_flight_fences,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SyncObjects) void {
        var i: usize = 0;
        while (i < MAX_FRAMES_IN_FLIGHT) : (i += 1) {
            c.vkDestroyFence(self.device, self.in_flight_fences[i], null);
            c.vkDestroySemaphore(self.device, self.render_finished_semaphores[i], null);
            c.vkDestroySemaphore(self.device, self.image_available_semaphores[i], null);
        }

        self.allocator.free(self.in_flight_fences);
        self.allocator.free(self.render_finished_semaphores);
        self.allocator.free(self.image_available_semaphores);
    }

    pub fn waitForFence(self: *SyncObjects, frame_index: usize) !void {
        _ = c.vkWaitForFences(self.device, 1, &self.in_flight_fences[frame_index], c.VK_TRUE, std.math.maxInt(u64));
    }

    pub fn resetFence(self: *SyncObjects, frame_index: usize) !void {
        _ = c.vkResetFences(self.device, 1, &self.in_flight_fences[frame_index]);
    }
};
