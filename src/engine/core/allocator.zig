const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const logger = @import("logger.zig");

pub const MemoryType = enum {
    GpuOnly,

    CpuToGpu,

    GpuToCpu,
};

const MemoryBlock = struct {
    memory: c.VkDeviceMemory,
    size: c.VkDeviceSize,
    offset: c.VkDeviceSize,
    mapped_ptr: ?*anyopaque,
    is_free: bool,
    alignment: c.VkDeviceSize,
};

const MemoryPool = struct {
    device: c.VkDevice,
    memory_type_index: u32,
    blocks: std.ArrayList(MemoryBlock),
    total_size: c.VkDeviceSize,
    used_size: c.VkDeviceSize,
    min_block_size: c.VkDeviceSize,
    allocator: std.mem.Allocator,

    const Self = @This();

    pub fn init(
        device: c.VkDevice,
        memory_type_index: u32,
        min_block_size: c.VkDeviceSize,
        allocator: std.mem.Allocator,
    ) !Self {
        return Self{
            .device = device,
            .memory_type_index = memory_type_index,
            .blocks = std.ArrayList(MemoryBlock).init(allocator),
            .total_size = 0,
            .used_size = 0,
            .min_block_size = min_block_size,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.blocks.items) |block| {
            if (block.mapped_ptr != null) {
                c.vkUnmapMemory(self.device, block.memory);
            }
            c.vkFreeMemory(self.device, block.memory, null);
        }
        self.blocks.deinit();
    }

    pub fn allocate(
        self: *Self,
        size: c.VkDeviceSize,
        alignment: c.VkDeviceSize,
        should_map: bool,
    ) !MemoryBlock {
        for (self.blocks.items) |*block| {
            if (block.is_free and block.size >= size and block.alignment >= alignment) {
                block.is_free = false;
                self.used_size += block.size;
                return block.*;
            }
        }

        const block_size = @max(size, self.min_block_size);

        const alloc_info = c.VkMemoryAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = block_size,
            .memoryTypeIndex = self.memory_type_index,
            .pNext = null,
        };

        var memory: c.VkDeviceMemory = undefined;
        if (c.vkAllocateMemory(self.device, &alloc_info, null, &memory) != c.VK_SUCCESS) {
            return error.OutOfDeviceMemory;
        }

        var mapped_ptr: ?*anyopaque = null;
        if (should_map) {
            if (c.vkMapMemory(self.device, memory, 0, block_size, 0, &mapped_ptr) != c.VK_SUCCESS) {
                c.vkFreeMemory(self.device, memory, null);
                return error.MemoryMapFailed;
            }
        }

        const block = MemoryBlock{
            .memory = memory,
            .size = block_size,
            .offset = 0,
            .mapped_ptr = mapped_ptr,
            .is_free = false,
            .alignment = alignment,
        };

        try self.blocks.append(block);
        self.total_size += block_size;
        self.used_size += block_size;

        return block;
    }

    pub fn free(self: *Self, block: *MemoryBlock) void {
        block.is_free = true;
        self.used_size -= block.size;
    }
};

pub const GraphicsAllocator = struct {
    device: c.VkDevice,
    physical_device: c.VkPhysicalDevice,
    pools: std.AutoHashMap(MemoryType, MemoryPool),
    allocator: std.mem.Allocator,
    graphics_queue_family: u32,

    const Self = @This();

    pub fn init(
        device: c.VkDevice,
        physical_device: c.VkPhysicalDevice,
        graphics_queue_family: u32,
        allocator: std.mem.Allocator,
    ) !*Self {
        const self = try allocator.create(Self);

        self.* = .{
            .device = device,
            .physical_device = physical_device,
            .pools = std.AutoHashMap(MemoryType, MemoryPool).init(allocator),
            .allocator = allocator,
            .graphics_queue_family = graphics_queue_family,
        };

        var mem_properties: c.VkPhysicalDeviceMemoryProperties = undefined;
        c.vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

        const gpu_only_index = try self.findMemoryType(
            mem_properties,
            c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            std.math.maxInt(u32),
        );

        const cpu_to_gpu_index = try self.findMemoryType(
            mem_properties,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            std.math.maxInt(u32),
        );

        const gpu_to_cpu_index = try self.findMemoryType(
            mem_properties,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            std.math.maxInt(u32),
        );

        try self.pools.put(.GpuOnly, try MemoryPool.init(
            device,
            gpu_only_index,
            16 * 1024 * 1024,
            allocator,
        ));

        try self.pools.put(.CpuToGpu, try MemoryPool.init(
            device,
            cpu_to_gpu_index,
            8 * 1024 * 1024,
            allocator,
        ));

        try self.pools.put(.GpuToCpu, try MemoryPool.init(
            device,
            gpu_to_cpu_index,
            4 * 1024 * 1024,
            allocator,
        ));

        return self;
    }

    pub fn deinit(self: *Self) void {
        var iterator = self.pools.valueIterator();
        while (iterator.next()) |pool| {
            pool.deinit();
        }
        self.pools.deinit();
        self.allocator.destroy(self);
    }

    pub fn allocate(
        self: *Self,
        memory_type: MemoryType,
        size: c.VkDeviceSize,
        alignment: c.VkDeviceSize,
    ) !MemoryBlock {
        const should_map = memory_type != .GpuOnly;

        if (self.pools.getPtr(memory_type)) |pool| {
            return pool.allocate(size, alignment, should_map);
        }

        return error.InvalidMemoryType;
    }

    pub fn free(self: *Self, memory_type: MemoryType, block: *MemoryBlock) void {
        if (self.pools.getPtr(memory_type)) |pool| {
            pool.free(block);
        }
    }

    pub fn findMemoryType(
        self: *Self,
        properties: c.VkPhysicalDeviceMemoryProperties,
        required_properties: c.VkMemoryPropertyFlags,
        memory_type_bits: u32,
    ) !u32 {
        _ = self;
        var i: u32 = 0;
        while (i < properties.memoryTypeCount) : (i += 1) {
            const memory_type = properties.memoryTypes[i];
            const bit_mask = @as(u32, 1) << @intCast(i);
            if (((memory_type_bits & bit_mask) != 0) and
                (memory_type.propertyFlags & required_properties == required_properties))
            {
                return i;
            }
        }
        return error.NoSuitableMemoryType;
    }

    pub fn createCommandPool(self: *Self) !c.VkCommandPool {
        const pool_info = c.VkCommandPoolCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = self.graphics_queue_family,
            .pNext = null,
        };

        var command_pool: c.VkCommandPool = undefined;
        if (c.vkCreateCommandPool(self.device, &pool_info, null, &command_pool) != c.VK_SUCCESS) {
            return error.CommandPoolCreationFailed;
        }

        return command_pool;
    }

    pub fn getStats(self: *Self) struct {
        total_size: c.VkDeviceSize,
        used_size: c.VkDeviceSize,
        allocation_count: usize,
    } {
        var stats = .{
            .total_size = 0,
            .used_size = 0,
            .allocation_count = 0,
        };

        var iterator = self.pools.valueIterator();
        while (iterator.next()) |pool| {
            stats.total_size += pool.total_size;
            stats.used_size += pool.used_size;
            stats.allocation_count += pool.blocks.items.len;
        }

        return stats;
    }
};
