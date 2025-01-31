const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const logger = @import("../core/logger.zig");

pub const CommandPool = struct {
    handle: c.VkCommandPool,
    device: c.VkDevice,

    pub fn init(device: c.VkDevice, queue_family_index: u32) !CommandPool {
        const pool_info = c.VkCommandPoolCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queue_family_index,
            .pNext = null,
        };

        var pool: c.VkCommandPool = undefined;
        if (c.vkCreateCommandPool(device, &pool_info, null, &pool) != c.VK_SUCCESS) {
            return error.CommandPoolCreationFailed;
        }

        return CommandPool{
            .handle = pool,
            .device = device,
        };
    }

    pub fn deinit(self: *CommandPool) void {
        c.vkDestroyCommandPool(self.device, self.handle, null);
    }

    pub fn reset(self: *CommandPool) !void {
        if (c.vkResetCommandPool(self.device, self.handle, 0) != c.VK_SUCCESS) {
            return error.CommandPoolResetFailed;
        }
    }
};

pub const CommandBuffer = struct {
    handle: c.VkCommandBuffer,
    pool: *CommandPool,

    pub fn init(pool: *CommandPool, level: c.VkCommandBufferLevel) !CommandBuffer {
        const alloc_info = c.VkCommandBufferAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool.handle,
            .level = level,
            .commandBufferCount = 1,
            .pNext = null,
        };

        var buffer: c.VkCommandBuffer = undefined;
        if (c.vkAllocateCommandBuffers(pool.device, &alloc_info, &buffer) != c.VK_SUCCESS) {
            return error.CommandBufferAllocationFailed;
        }

        return CommandBuffer{
            .handle = buffer,
            .pool = pool,
        };
    }

    pub fn deinit(self: *CommandBuffer) void {
        c.vkFreeCommandBuffers(self.pool.device, self.pool.handle, 1, &self.handle);
    }

    pub fn begin(self: *CommandBuffer, flags: c.VkCommandBufferUsageFlags) !void {
        const begin_info = c.VkCommandBufferBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = flags,
            .pInheritanceInfo = null,
            .pNext = null,
        };

        if (c.vkBeginCommandBuffer(self.handle, &begin_info) != c.VK_SUCCESS) {
            return error.CommandBufferBeginFailed;
        }
    }

    pub fn end(self: *CommandBuffer) !void {
        if (c.vkEndCommandBuffer(self.handle) != c.VK_SUCCESS) {
            return error.CommandBufferEndFailed;
        }
    }

    pub fn reset(self: *CommandBuffer) !void {
        if (c.vkResetCommandBuffer(self.handle, 0) != c.VK_SUCCESS) {
            return error.CommandBufferResetFailed;
        }
    }

    pub fn submit(
        self: *CommandBuffer,
        queue: c.VkQueue,
        wait_semaphores: []const c.VkSemaphore,
        wait_stages: []const c.VkPipelineStageFlags,
        signal_semaphores: []const c.VkSemaphore,
        fence: ?c.VkFence,
    ) !void {
        const submit_info = c.VkSubmitInfo{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = @intCast(wait_semaphores.len),
            .pWaitSemaphores = wait_semaphores.ptr,
            .pWaitDstStageMask = wait_stages.ptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &self.handle,
            .signalSemaphoreCount = @intCast(signal_semaphores.len),
            .pSignalSemaphores = signal_semaphores.ptr,
            .pNext = null,
        };

        if (c.vkQueueSubmit(queue, 1, &submit_info, if (fence) |f| f else null) != c.VK_SUCCESS) {
            return error.CommandBufferSubmitFailed;
        }
    }

    pub const Error = error{
        CommandBufferAllocationFailed,
        CommandBufferBeginFailed,
        CommandBufferEndFailed,
        CommandBufferResetFailed,
        CommandBufferSubmitFailed,
    };
};

pub const CommandBufferBuilder = struct {
    buffer: *CommandBuffer,

    pub fn init(buffer: *CommandBuffer) !CommandBufferBuilder {
        try buffer.begin(0);
        return CommandBufferBuilder{
            .buffer = buffer,
        };
    }

    pub fn end(self: *CommandBufferBuilder) !void {
        try self.buffer.end();
    }

    pub fn bindPipeline(self: *CommandBufferBuilder, pipeline: c.VkPipeline, bind_point: c.VkPipelineBindPoint) void {
        c.vkCmdBindPipeline(self.buffer.handle, bind_point, pipeline);
    }

    pub fn setViewport(self: *CommandBufferBuilder, viewport: c.VkViewport) void {
        c.vkCmdSetViewport(self.buffer.handle, 0, 1, &viewport);
    }

    pub fn setScissor(self: *CommandBufferBuilder, scissor: c.VkRect2D) void {
        c.vkCmdSetScissor(self.buffer.handle, 0, 1, &scissor);
    }

    pub fn bindVertexBuffers(
        self: *CommandBufferBuilder,
        first_binding: u32,
        buffers: []const c.VkBuffer,
        offsets: []const c.VkDeviceSize,
    ) void {
        c.vkCmdBindVertexBuffers(
            self.buffer.handle,
            first_binding,
            @intCast(buffers.len),
            buffers.ptr,
            offsets.ptr,
        );
    }

    pub fn bindIndexBuffer(
        self: *CommandBufferBuilder,
        buffer: c.VkBuffer,
        offset: c.VkDeviceSize,
        index_type: c.VkIndexType,
    ) void {
        c.vkCmdBindIndexBuffer(self.buffer.handle, buffer, offset, index_type);
    }

    pub fn drawIndexed(
        self: *CommandBufferBuilder,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) void {
        c.vkCmdDrawIndexed(
            self.buffer.handle,
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        );
    }

    pub fn beginRenderPass(
        self: *CommandBufferBuilder,
        render_pass: c.VkRenderPass,
        framebuffer: c.VkFramebuffer,
        render_area: c.VkRect2D,
        clear_values: []const c.VkClearValue,
    ) void {
        const render_pass_info = c.VkRenderPassBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = render_pass,
            .framebuffer = framebuffer,
            .renderArea = render_area,
            .clearValueCount = @intCast(clear_values.len),
            .pClearValues = clear_values.ptr,
            .pNext = null,
        };

        c.vkCmdBeginRenderPass(
            self.buffer.handle,
            &render_pass_info,
            c.VK_SUBPASS_CONTENTS_INLINE,
        );
    }

    pub fn endRenderPass(self: *CommandBufferBuilder) void {
        c.vkCmdEndRenderPass(self.buffer.handle);
    }

    pub fn pushConstants(
        self: *CommandBufferBuilder,
        layout: c.VkPipelineLayout,
        stage_flags: c.VkShaderStageFlags,
        offset: u32,
        size: u32,
        values: *const anyopaque,
    ) void {
        c.vkCmdPushConstants(
            self.buffer.handle,
            layout,
            stage_flags,
            offset,
            size,
            values,
        );
    }
};
