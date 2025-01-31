const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const math = @import("../core/math.zig");
const logger = @import("../core/logger.zig");
const resources = @import("resources.zig");
const builtin = @import("std").builtin;
const physical = @import("device/physical.zig");

fn calculateModelMatrix(position: [2]f32, size: [2]f32, rotation: f32) [4][4]f32 {
    const pos = math.Vec2.fromArray(position);
    const scale = math.Vec2.fromArray(size);

    // Create translation matrix
    var transform = math.Mat4.translate(pos.x(), pos.y(), 0);

    // Create rotation matrix
    const rot = math.Mat4.rotate(rotation, 0, 0, 1);

    // Create scale matrix
    const scl = math.Mat4.scale(scale.x(), scale.y(), 1);

    // Combine matrices: transform * rot * scale
    const result = transform.mul(rot).mul(scl);

    return result.toArray2D();
}

pub const Vertex = extern struct {
    position: [2]f32 align(16),
    uv: [2]f32 align(16),
    color: [4]f32 align(16),

    pub inline fn getBindingDescription() c.VkVertexInputBindingDescription {
        return .{
            .binding = 0,
            .stride = @sizeOf(Vertex),
            .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX,
        };
    }

    pub inline fn getAttributeDescriptions() [3]c.VkVertexInputAttributeDescription {
        return .{
            .{
                .binding = 0,
                .location = 0,
                .format = c.VK_FORMAT_R32G32_SFLOAT,
                .offset = @offsetOf(Vertex, "position"),
            },
            .{
                .binding = 0,
                .location = 1,
                .format = c.VK_FORMAT_R32G32_SFLOAT,
                .offset = @offsetOf(Vertex, "uv"),
            },
            .{
                .binding = 0,
                .location = 2,
                .format = c.VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = @offsetOf(Vertex, "color"),
            },
        };
    }
};

pub const SpriteInstance = extern struct {
    model: [4][4]f32 align(16),
    color_and_uv: [4]f32 align(16),
    metadata: packed struct {
        layer: f32,
        texture_id: u32,
        flags: u32,
        _padding: u32 = 0,
    } align(16),

    pub inline fn getBindingDescription() c.VkVertexInputBindingDescription {
        return .{
            .binding = 1,
            .stride = @sizeOf(SpriteInstance),
            .inputRate = c.VK_VERTEX_INPUT_RATE_INSTANCE,
        };
    }

    pub inline fn getAttributeDescriptions() [6]c.VkVertexInputAttributeDescription {
        return .{
            .{
                .binding = 1,
                .location = 3,
                .format = c.VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = @offsetOf(SpriteInstance, "model") + 0,
            },
            .{
                .binding = 1,
                .location = 4,
                .format = c.VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = @offsetOf(SpriteInstance, "model") + 16,
            },
            .{
                .binding = 1,
                .location = 5,
                .format = c.VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = @offsetOf(SpriteInstance, "model") + 32,
            },
            .{
                .binding = 1,
                .location = 6,
                .format = c.VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = @offsetOf(SpriteInstance, "model") + 48,
            },
            .{
                .binding = 1,
                .location = 7,
                .format = c.VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = @offsetOf(SpriteInstance, "color_and_uv"),
            },
            .{
                .binding = 1,
                .location = 8,
                .format = c.VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = @offsetOf(SpriteInstance, "metadata"),
            },
        };
    }
};

const SortKey = packed struct {
    layer: u16,
    texture_id: u8,
    blend_mode: u4,
    flags: u4,
};

fn compareSortKeys(_: void, a: SortKey, b: SortKey) bool {
    const self_val = @as(u32, @bitCast(a));
    const other_val = @as(u32, @bitCast(b));
    return self_val < other_val;
}

const CommandBufferPool = struct {
    pool: c.VkCommandPool,
    buffers: []c.VkCommandBuffer,
    device: c.VkDevice,
    allocator: std.mem.Allocator,
    current: usize,

    pub fn init(device: c.VkDevice, queue_family: u32, max_buffers: u32, alloc: std.mem.Allocator) !CommandBufferPool {
        var pool: c.VkCommandPool = undefined;
        const pool_info = c.VkCommandPoolCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = 0,
            .queueFamilyIndex = queue_family,
            .pNext = null,
        };

        if (c.vkCreateCommandPool(device, &pool_info, null, &pool) != c.VK_SUCCESS) {
            return error.CommandPoolCreationFailed;
        }
        errdefer c.vkDestroyCommandPool(device, pool, null);

        const buffers = try alloc.alloc(c.VkCommandBuffer, max_buffers);
        errdefer alloc.free(buffers);

        const alloc_info = c.VkCommandBufferAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool,
            .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = max_buffers,
            .pNext = null,
        };

        if (c.vkAllocateCommandBuffers(device, &alloc_info, buffers.ptr) != c.VK_SUCCESS) {
            return error.CommandBufferAllocationFailed;
        }

        return CommandBufferPool{
            .pool = pool,
            .buffers = buffers,
            .device = device,
            .allocator = alloc,
            .current = 0,
        };
    }

    pub fn deinit(self: *CommandBufferPool) void {
        c.vkDestroyCommandPool(self.device, self.pool, null);
        self.allocator.free(self.buffers);
    }

    pub fn getNextBuffer(self: *CommandBufferPool) !c.VkCommandBuffer {
        if (self.current >= self.buffers.len) {
            return error.NoCommandBuffersAvailable;
        }
        const buffer = self.buffers[self.current];
        self.current += 1;
        return buffer;
    }

    pub fn reset(self: *CommandBufferPool) void {
        _ = c.vkResetCommandPool(self.device, self.pool, c.VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
        self.current = 0;
    }
};

const SpriteTransform = extern struct {
    position: [4]f32 align(16),
    scale: [4]f32 align(16),
    rotation: [4]f32 align(16),
    layer: [4]f32 align(16),
};

const COMPUTE_WORKGROUP_SIZE = 256;
const MAX_SPRITES_PER_DISPATCH = COMPUTE_WORKGROUP_SIZE * 64;

const MEMORY_POOL_BLOCK_SIZE = 64;
const MEMORY_POOL_ALIGNMENT = 16;
const MEMORY_POOL_MIN_BLOCKS = 8;
const MEMORY_POOL_MAX_BLOCKS = 1024;

const MemoryBlock = struct {
    data: []align(MEMORY_POOL_ALIGNMENT) u8,
    next: ?*MemoryBlock,
    used: bool,
    size: usize,
};

const MemoryChunk = struct {
    blocks: []MemoryBlock,
    next: ?*MemoryChunk,
    used_blocks: u32,
};

const SpriteMemoryPool = struct {
    transforms: []align(MEMORY_POOL_ALIGNMENT) SpriteTransform,
    vertices: []align(MEMORY_POOL_ALIGNMENT) Vertex,
    indices: []align(MEMORY_POOL_ALIGNMENT) u32,
    instances: []align(MEMORY_POOL_ALIGNMENT) SpriteInstance,

    transform_chunks: std.ArrayList(*MemoryChunk),
    vertex_chunks: std.ArrayList(*MemoryChunk),
    instance_chunks: std.ArrayList(*MemoryChunk),

    free_transforms: ?*MemoryBlock,
    free_vertices: ?*MemoryBlock,
    free_instances: ?*MemoryBlock,

    allocator: std.mem.Allocator,

    pub fn init(max_sprites: u32, alloc: std.mem.Allocator) !SpriteMemoryPool {
        const transforms = try alloc.alignedAlloc(SpriteTransform, MEMORY_POOL_ALIGNMENT, max_sprites);
        errdefer alloc.free(transforms);

        const vertices = try alloc.alignedAlloc(Vertex, MEMORY_POOL_ALIGNMENT, max_sprites * 4);
        errdefer alloc.free(vertices);

        const indices = try alloc.alignedAlloc(u32, MEMORY_POOL_ALIGNMENT, max_sprites * 6);
        errdefer alloc.free(indices);

        const instances = try alloc.alignedAlloc(SpriteInstance, MEMORY_POOL_ALIGNMENT, max_sprites);
        errdefer alloc.free(instances);

        var transform_chunks = std.ArrayList(*MemoryChunk).init(alloc);
        errdefer transform_chunks.deinit();

        var vertex_chunks = std.ArrayList(*MemoryChunk).init(alloc);
        errdefer vertex_chunks.deinit();

        var instance_chunks = std.ArrayList(*MemoryChunk).init(alloc);
        errdefer instance_chunks.deinit();

        const initial_blocks = @max(MEMORY_POOL_MIN_BLOCKS, max_sprites / MEMORY_POOL_BLOCK_SIZE);
        try transform_chunks.append(try createMemoryChunk(alloc, @sizeOf(SpriteTransform), initial_blocks));
        try vertex_chunks.append(try createMemoryChunk(alloc, @sizeOf(Vertex) * 4, initial_blocks));
        try instance_chunks.append(try createMemoryChunk(alloc, @sizeOf(SpriteInstance), initial_blocks));

        return SpriteMemoryPool{
            .transforms = transforms,
            .vertices = vertices,
            .indices = indices,
            .instances = instances,
            .transform_chunks = transform_chunks,
            .vertex_chunks = vertex_chunks,
            .instance_chunks = instance_chunks,
            .free_transforms = null,
            .free_vertices = null,
            .free_instances = null,
            .allocator = alloc,
        };
    }

    pub fn deinit(self: *SpriteMemoryPool) void {
        for (self.transform_chunks.items) |chunk| {
            freeMemoryChunk(self.allocator, chunk);
        }
        for (self.vertex_chunks.items) |chunk| {
            freeMemoryChunk(self.allocator, chunk);
        }
        for (self.instance_chunks.items) |chunk| {
            freeMemoryChunk(self.allocator, chunk);
        }

        self.transform_chunks.deinit();
        self.vertex_chunks.deinit();
        self.instance_chunks.deinit();

        self.allocator.free(self.transforms);
        self.allocator.free(self.vertices);
        self.allocator.free(self.indices);
        self.allocator.free(self.instances);
    }

    fn createMemoryChunk(allocator: std.mem.Allocator, block_size: usize, num_blocks: u32) !*MemoryChunk {
        const chunk = try allocator.create(MemoryChunk);
        errdefer allocator.destroy(chunk);

        const aligned_size = std.mem.alignForward(usize, block_size, MEMORY_POOL_ALIGNMENT);
        const blocks = try allocator.alloc(MemoryBlock, num_blocks);
        errdefer allocator.free(blocks);

        const total_size = aligned_size * num_blocks;
        const data = try allocator.alignedAlloc(u8, MEMORY_POOL_ALIGNMENT, total_size);
        errdefer allocator.free(data);

        for (blocks, 0..) |*block, i| {
            const start = i * aligned_size;
            const slice = data[start .. start + aligned_size];
            block.* = .{
                .data = @alignCast(slice),
                .next = if (i < blocks.len - 1) &blocks[i + 1] else null,
                .used = false,
                .size = block_size,
            };
        }

        chunk.* = .{
            .blocks = blocks,
            .next = null,
            .used_blocks = 0,
        };

        return chunk;
    }

    fn freeMemoryChunk(allocator: std.mem.Allocator, chunk: *MemoryChunk) void {
        if (chunk.blocks.len > 0) {
            allocator.free(chunk.blocks[0].data[0 .. chunk.blocks[0].size * chunk.blocks.len]);
        }
        allocator.free(chunk.blocks);
        allocator.destroy(chunk);
    }

    pub fn allocTransform(self: *SpriteMemoryPool) ?*SpriteTransform {
        if (self.free_transforms) |block| {
            self.free_transforms = block.next;
            block.used = true;
            return @ptrCast(@alignCast(block.data.ptr));
        }

        for (self.transform_chunks.items) |chunk| {
            if (chunk.used_blocks < chunk.blocks.len) {
                const block = &chunk.blocks[chunk.used_blocks];
                chunk.used_blocks += 1;
                block.used = true;
                return @ptrCast(@alignCast(block.data.ptr));
            }
        }

        if (self.transform_chunks.items.len * MEMORY_POOL_BLOCK_SIZE < MEMORY_POOL_MAX_BLOCKS) {
            const new_chunk = createMemoryChunk(
                self.allocator,
                @sizeOf(SpriteTransform),
                MEMORY_POOL_BLOCK_SIZE,
            ) catch return null;

            self.transform_chunks.append(new_chunk) catch {
                freeMemoryChunk(self.allocator, new_chunk);
                return null;
            };

            const block = &new_chunk.blocks[0];
            new_chunk.used_blocks = 1;
            block.used = true;
            return @ptrCast(@alignCast(block.data.ptr));
        }

        return null;
    }

    pub fn allocVertex(self: *SpriteMemoryPool) ?*Vertex {
        if (self.free_vertices) |block| {
            self.free_vertices = block.next;
            block.used = true;
            return @ptrCast(@alignCast(block.data.ptr));
        }

        for (self.vertex_chunks.items) |chunk| {
            if (chunk.used_blocks < chunk.blocks.len) {
                const block = &chunk.blocks[chunk.used_blocks];
                chunk.used_blocks += 1;
                block.used = true;
                return @ptrCast(@alignCast(block.data.ptr));
            }
        }

        if (self.vertex_chunks.items.len * MEMORY_POOL_BLOCK_SIZE < MEMORY_POOL_MAX_BLOCKS) {
            const new_chunk = createMemoryChunk(
                self.allocator,
                @sizeOf(Vertex) * 4,
                MEMORY_POOL_BLOCK_SIZE,
            ) catch return null;

            self.vertex_chunks.append(new_chunk) catch {
                freeMemoryChunk(self.allocator, new_chunk);
                return null;
            };

            const block = &new_chunk.blocks[0];
            new_chunk.used_blocks = 1;
            block.used = true;
            return @ptrCast(@alignCast(block.data.ptr));
        }

        return null;
    }

    pub fn allocInstance(self: *SpriteMemoryPool) ?*SpriteInstance {
        if (self.free_instances) |block| {
            self.free_instances = block.next;
            block.used = true;
            return @ptrCast(@alignCast(block.data.ptr));
        }

        for (self.instance_chunks.items) |chunk| {
            if (chunk.used_blocks < chunk.blocks.len) {
                const block = &chunk.blocks[chunk.used_blocks];
                chunk.used_blocks += 1;
                block.used = true;
                return @ptrCast(@alignCast(block.data.ptr));
            }
        }

        if (self.instance_chunks.items.len * MEMORY_POOL_BLOCK_SIZE < MEMORY_POOL_MAX_BLOCKS) {
            const new_chunk = createMemoryChunk(
                self.allocator,
                @sizeOf(SpriteInstance),
                MEMORY_POOL_BLOCK_SIZE,
            ) catch return null;

            self.instance_chunks.append(new_chunk) catch {
                freeMemoryChunk(self.allocator, new_chunk);
                return null;
            };

            const block = &new_chunk.blocks[0];
            new_chunk.used_blocks = 1;
            block.used = true;
            return @ptrCast(@alignCast(block.data.ptr));
        }

        return null;
    }

    pub fn freeTransform(self: *SpriteMemoryPool, ptr: *SpriteTransform) void {
        const block_ptr = @as([*]u8, @ptrCast(ptr));

        for (self.transform_chunks.items) |chunk| {
            for (chunk.blocks) |*block| {
                if (block.data.ptr == block_ptr) {
                    block.used = false;
                    block.next = self.free_transforms;
                    self.free_transforms = block;
                    return;
                }
            }
        }
    }

    pub fn freeVertex(self: *SpriteMemoryPool, ptr: *Vertex) void {
        const block_ptr = @as([*]u8, @ptrCast(ptr));

        for (self.vertex_chunks.items) |chunk| {
            for (chunk.blocks) |*block| {
                if (block.data.ptr == block_ptr) {
                    block.used = false;
                    block.next = self.free_vertices;
                    self.free_vertices = block;
                    return;
                }
            }
        }
    }

    pub fn freeInstance(self: *SpriteMemoryPool, ptr: *SpriteInstance) void {
        const block_ptr = @as([*]u8, @ptrCast(ptr));

        for (self.instance_chunks.items) |chunk| {
            for (chunk.blocks) |*block| {
                if (block.data.ptr == block_ptr) {
                    block.used = false;
                    block.next = self.free_instances;
                    self.free_instances = block;
                    return;
                }
            }
        }
    }
};

const ComputePipeline = struct {
    pipeline: c.VkPipeline,
    pipeline_layout: c.VkPipelineLayout,
    descriptor_set_layout: c.VkDescriptorSetLayout,
    descriptor_pool: c.VkDescriptorPool,
    descriptor_sets: []c.VkDescriptorSet,

    pub fn init(
        device: c.VkDevice,
        shader_module: c.VkShaderModule,
        max_sets: u32,
        alloc: std.mem.Allocator,
    ) !ComputePipeline {
        const binding_flags = c.VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
            c.VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;

        const bindings = [_]c.VkDescriptorSetLayoutBinding{
            .{
                .binding = 0,
                .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = null,
            },
            .{
                .binding = 1,
                .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = null,
            },
            .{
                .binding = 2,
                .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = null,
            },
        };

        const binding_flags_array = [_]c.VkDescriptorBindingFlags{
            binding_flags, binding_flags, binding_flags,
        };

        const flags_info = c.VkDescriptorSetLayoutBindingFlagsCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
            .bindingCount = binding_flags_array.len,
            .pBindingFlags = &binding_flags_array,
            .pNext = null,
        };

        const layout_info = c.VkDescriptorSetLayoutCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .flags = c.VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
            .bindingCount = bindings.len,
            .pBindings = &bindings,
            .pNext = &flags_info,
        };

        var descriptor_set_layout: c.VkDescriptorSetLayout = undefined;
        if (c.vkCreateDescriptorSetLayout(
            device,
            &layout_info,
            null,
            &descriptor_set_layout,
        ) != c.VK_SUCCESS) {
            return error.DescriptorSetLayoutCreationFailed;
        }
        errdefer c.vkDestroyDescriptorSetLayout(device, descriptor_set_layout, null);

        const push_constant_range = c.VkPushConstantRange{
            .stageFlags = c.VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = @sizeOf(ComputePushConstants),
        };

        const pipeline_layout_info = c.VkPipelineLayoutCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &descriptor_set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant_range,
            .pNext = null,
            .flags = 0,
        };

        var pipeline_layout: c.VkPipelineLayout = undefined;
        if (c.vkCreatePipelineLayout(
            device,
            &pipeline_layout_info,
            null,
            &pipeline_layout,
        ) != c.VK_SUCCESS) {
            return error.PipelineLayoutCreationFailed;
        }
        errdefer c.vkDestroyPipelineLayout(device, pipeline_layout, null);

        const pipeline_info = c.VkComputePipelineCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = .{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = c.VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader_module,
                .pName = "main",
                .pNext = null,
                .flags = 0,
                .pSpecializationInfo = null,
            },
            .layout = pipeline_layout,
            .pNext = null,
            .flags = 0,
            .basePipelineHandle = null,
            .basePipelineIndex = -1,
        };

        var pipeline: c.VkPipeline = undefined;
        if (c.vkCreateComputePipelines(
            device,
            null,
            1,
            &pipeline_info,
            null,
            &pipeline,
        ) != c.VK_SUCCESS) {
            return error.ComputePipelineCreationFailed;
        }
        errdefer c.vkDestroyPipeline(device, pipeline, null);

        const pool_sizes = [_]c.VkDescriptorPoolSize{
            .{
                .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = max_sets * 3,
            },
        };

        const pool_info = c.VkDescriptorPoolCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags = c.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = max_sets,
            .poolSizeCount = pool_sizes.len,
            .pPoolSizes = &pool_sizes,
            .pNext = null,
        };

        var descriptor_pool: c.VkDescriptorPool = undefined;
        if (c.vkCreateDescriptorPool(
            device,
            &pool_info,
            null,
            &descriptor_pool,
        ) != c.VK_SUCCESS) {
            return error.DescriptorPoolCreationFailed;
        }
        errdefer c.vkDestroyDescriptorPool(device, descriptor_pool, null);

        const layouts = try alloc.alloc(c.VkDescriptorSetLayout, max_sets);
        defer alloc.free(layouts);
        @memset(layouts, descriptor_set_layout);

        const alloc_info = c.VkDescriptorSetAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = max_sets,
            .pSetLayouts = layouts.ptr,
            .pNext = null,
        };

        const descriptor_sets = try alloc.alloc(c.VkDescriptorSet, max_sets);
        errdefer alloc.free(descriptor_sets);

        if (c.vkAllocateDescriptorSets(
            device,
            &alloc_info,
            descriptor_sets.ptr,
        ) != c.VK_SUCCESS) {
            return error.DescriptorSetAllocationFailed;
        }

        return ComputePipeline{
            .pipeline = pipeline,
            .pipeline_layout = pipeline_layout,
            .descriptor_set_layout = descriptor_set_layout,
            .descriptor_pool = descriptor_pool,
            .descriptor_sets = descriptor_sets,
        };
    }

    pub fn deinit(self: *ComputePipeline, device: c.VkDevice, alloc: std.mem.Allocator) void {
        c.vkDestroyPipeline(device, self.pipeline, null);
        c.vkDestroyPipelineLayout(device, self.pipeline_layout, null);
        c.vkDestroyDescriptorSetLayout(device, self.descriptor_set_layout, null);
        c.vkDestroyDescriptorPool(device, self.descriptor_pool, null);
        alloc.free(self.descriptor_sets);
    }
};

const ComputePushConstants = extern struct {
    view_proj: [16]f32 align(16),
    frustum_planes: [6][4]f32 align(16),
    sprite_count: u32,
    padding: [3]u32,
};

const BatchKey = packed struct {
    texture_id: u32,
    blend_mode: u4,
    flags: u4,
    _padding: u24 = 0,

    pub fn hash(self: BatchKey) u32 {
        var h: u32 = 2166136261;
        const bytes = @as([*]const u8, @ptrCast(&self))[0..@sizeOf(BatchKey)];
        for (bytes) |b| {
            h = (h ^ b) *% 16777619;
        }
        return h;
    }

    pub fn eql(self: BatchKey, other: BatchKey) bool {
        return @as(u32, @bitCast(self)) == @as(u32, @bitCast(other));
    }
};

const BatchInfo = struct {
    start_index: u32,
    sprite_count: u32,
    layer_min: f32,
    layer_max: f32,
};

const BatchTable = std.AutoHashMap(BatchKey, BatchInfo);

pub const SpriteBatch = struct {
    vertices: []Vertex,
    indices: []u32,
    instances: []SpriteInstance,
    vertex_buffer: resources.Buffer,
    index_buffer: resources.Buffer,
    instance_buffer: resources.Buffer,

    cmd_pool: CommandBufferPool,
    current_cmd: ?c.VkCommandBuffer,

    device: c.VkDevice,
    allocator: std.mem.Allocator,
    sprite_count: u32,
    instance_count: u32,
    max_sprites: u32,
    has_compute: bool,

    mapped_vertices: ?[*]Vertex,
    mapped_indices: ?[*]u32,
    mapped_instances: ?[*]SpriteInstance,

    sort_keys: []SortKey,
    draw_calls: std.ArrayList(DrawCall),

    compute_pipeline: ComputePipeline,
    memory_pool: SpriteMemoryPool,
    transform_buffer: resources.Buffer,
    compute_cmd: ?c.VkCommandBuffer,

    batch_table: BatchTable,
    batch_keys: []BatchKey,

    const VERTEX_BUFFER_ALIGNMENT = 256;
    const INDEX_BUFFER_ALIGNMENT = 64;
    const INSTANCE_BUFFER_ALIGNMENT = 256;
    const DEFAULT_MAX_SPRITES = 10000;
    const VERTICES_PER_SPRITE = 4;
    const INDICES_PER_SPRITE = 6;
    const MAX_COMMAND_BUFFERS = 8;

    const DrawCall = struct {
        first_index: u32,
        index_count: u32,
        instance_count: u32,
        texture_id: u32,
        blend_mode: u4,
    };

    pub fn init(
        device: c.VkDevice,
        physical_device: *physical.PhysicalDevice,
        queue_family: u32,
        compute_shader: ?c.VkShaderModule,
        max_sprites: u32,
        alloc: std.mem.Allocator,
    ) !*SpriteBatch {
        logger.info("spritebatch: initializing with capacity for {d} sprites", .{max_sprites});

        const actual_max_sprites = if (max_sprites == 0) DEFAULT_MAX_SPRITES else max_sprites;

        const vertex_size = @sizeOf(Vertex) * actual_max_sprites * VERTICES_PER_SPRITE;
        const index_size = @sizeOf(u32) * actual_max_sprites * INDICES_PER_SPRITE;
        const instance_size = @sizeOf(SpriteInstance) * actual_max_sprites;

        const vertices = try alloc.alloc(Vertex, actual_max_sprites * VERTICES_PER_SPRITE);
        errdefer alloc.free(vertices);

        const indices = try alloc.alloc(u32, actual_max_sprites * INDICES_PER_SPRITE);
        errdefer alloc.free(indices);

        const instances = try alloc.alloc(SpriteInstance, actual_max_sprites);
        errdefer alloc.free(instances);

        if (vertex_size > 256 * 1024 * 1024) {
            logger.warn("spritebatch: large vertex buffer ({d} MB), consider reducing sprite count", .{
                vertex_size / (1024 * 1024),
            });
        }

        if (instance_size > 128 * 1024 * 1024) {
            logger.warn("spritebatch: large instance buffer ({d} MB), consider reducing sprite count", .{
                instance_size / (1024 * 1024),
            });
        }

        var vertex_buffer = try resources.Buffer.init(
            device,
            physical_device.handle,
            vertex_size,
            c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        errdefer vertex_buffer.deinit();

        var index_buffer = try resources.Buffer.init(
            device,
            physical_device.handle,
            index_size,
            c.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        errdefer index_buffer.deinit();

        var instance_buffer = try resources.Buffer.init(
            device,
            physical_device.handle,
            instance_size,
            c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        errdefer instance_buffer.deinit();

        var draw_calls = std.ArrayList(DrawCall).init(alloc);
        errdefer draw_calls.deinit();

        var i: u32 = 0;
        while (i < actual_max_sprites) : (i += 1) {
            const base = i * VERTICES_PER_SPRITE;
            const idx = i * INDICES_PER_SPRITE;
            indices[idx + 0] = base + 0;
            indices[idx + 1] = base + 2;
            indices[idx + 2] = base + 1;
            indices[idx + 3] = base + 0;
            indices[idx + 4] = base + 3;
            indices[idx + 5] = base + 2;
        }

        var cmd_pool = try CommandBufferPool.init(device, queue_family, MAX_COMMAND_BUFFERS, alloc);
        errdefer cmd_pool.deinit();

        try vertex_buffer.map();
        try index_buffer.map();
        try instance_buffer.map();

        const self = try alloc.create(SpriteBatch);
        errdefer alloc.destroy(self);

        self.* = .{
            .vertices = vertices,
            .indices = indices,
            .instances = instances,
            .vertex_buffer = vertex_buffer,
            .index_buffer = index_buffer,
            .instance_buffer = instance_buffer,
            .cmd_pool = cmd_pool,
            .current_cmd = null,
            .device = device,
            .allocator = alloc,
            .sprite_count = 0,
            .instance_count = 0,
            .max_sprites = actual_max_sprites,
            .has_compute = false,
            .mapped_vertices = @ptrCast(@alignCast(vertex_buffer.mapped_data.?)),
            .mapped_indices = @ptrCast(@alignCast(index_buffer.mapped_data.?)),
            .mapped_instances = @ptrCast(@alignCast(instance_buffer.mapped_data.?)),
            .sort_keys = try alloc.alloc(SortKey, actual_max_sprites),
            .draw_calls = draw_calls,
            .compute_pipeline = undefined,
            .memory_pool = undefined,
            .transform_buffer = undefined,
            .compute_cmd = null,
            .batch_table = BatchTable.init(alloc),
            .batch_keys = try alloc.alloc(BatchKey, actual_max_sprites),
        };

        @memcpy(
            @as([*]u8, @ptrCast(@alignCast(self.mapped_indices.?)))[0..index_size],
            @as([*]const u8, @ptrCast(@alignCast(indices.ptr)))[0..index_size],
        );

        if (compute_shader != null) {
            logger.info("spritebatch: initializing compute pipeline for hardware acceleration", .{});
            var compute_pipeline = try ComputePipeline.init(
                device,
                compute_shader.?,
                MAX_COMMAND_BUFFERS,
                alloc,
            );
            errdefer compute_pipeline.deinit(device, alloc);

            var memory_pool = try SpriteMemoryPool.init(max_sprites, alloc);
            errdefer memory_pool.deinit();

            const transform_size = @sizeOf(SpriteTransform) * max_sprites;
            var transform_buffer = try resources.Buffer.init(
                device,
                physical_device.handle,
                transform_size,
                c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            );
            errdefer transform_buffer.deinit();

            self.compute_pipeline = compute_pipeline;
            self.memory_pool = memory_pool;
            self.transform_buffer = transform_buffer;
            self.has_compute = true;
        }
        self.compute_cmd = null;

        logger.info("spritebatch: initialized successfully\n  Vertex buffer: {d} KB ({d} vertices)\n  Index buffer: {d} KB ({d} indices)\n  Instance buffer: {d} KB ({d} instances)", .{
            vertex_size / 1024,
            actual_max_sprites * VERTICES_PER_SPRITE,
            index_size / 1024,
            actual_max_sprites * INDICES_PER_SPRITE,
            instance_size / 1024,
            actual_max_sprites,
        });
        return self;
    }

    pub fn deinit(self: *SpriteBatch) void {
        logger.info("spritebatch: shutting down", .{});
        self.draw_calls.deinit();
        self.vertex_buffer.unmap();
        self.index_buffer.unmap();
        self.instance_buffer.unmap();
        self.vertex_buffer.deinit();
        self.index_buffer.deinit();
        self.instance_buffer.deinit();
        self.cmd_pool.deinit();

        if (self.has_compute) {
            self.compute_pipeline.deinit(self.device, self.allocator);
            self.memory_pool.deinit();
            self.transform_buffer.deinit();
        }

        self.allocator.free(self.vertices);
        self.allocator.free(self.indices);
        self.allocator.free(self.instances);
        self.allocator.free(self.sort_keys);
        self.batch_table.deinit();
        self.allocator.free(self.batch_keys);
        self.allocator.destroy(self);
    }

    pub fn begin(self: *SpriteBatch) !void {
        self.sprite_count = 0;
        self.instance_count = 0;
        self.draw_calls.clearRetainingCapacity();
        self.current_cmd = self.cmd_pool.getNextBuffer() catch |err| {
            logger.err("spritebatch: command buffer allocation failed\nPossible solutions:\n1. Increase MAX_COMMAND_BUFFERS (current: {d})\n2. Ensure proper GPU synchronization\n3. Reduce draw call frequency", .{MAX_COMMAND_BUFFERS});
            return err;
        };

        const begin_info = c.VkCommandBufferBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null,
            .pNext = null,
        };

        if (c.vkBeginCommandBuffer(self.current_cmd.?, &begin_info) != c.VK_SUCCESS) {
            return error.CommandBufferBeginFailed;
        }
    }

    pub fn end(self: *SpriteBatch) !void {
        if (self.sprite_count == 0) {
            self.cmd_pool.reset();
            return;
        }

        if (self.draw_calls.items.len > 100) {
            logger.warn("spritebatch: high draw call count ({d}), consider:\n1. Using texture atlasing\n2. Enabling instancing\n3. Implementing batching", .{self.draw_calls.items.len});
        }

        std.sort.pdq(SortKey, self.sort_keys[0..self.sprite_count], {}, compareSortKeys);

        self.batch_table.clearRetainingCapacity();

        var i: u32 = 0;
        while (i < self.sprite_count) : (i += 1) {
            const key = BatchKey{
                .texture_id = self.sort_keys[i].texture_id,
                .blend_mode = self.sort_keys[i].blend_mode,
                .flags = self.sort_keys[i].flags,
                ._padding = 0,
            };

            const layer = @as(f32, @floatFromInt(self.sort_keys[i].layer)) / 65535.0;

            if (self.batch_table.getPtr(key)) |batch| {
                batch.sprite_count += 1;
                batch.layer_min = @min(batch.layer_min, layer);
                batch.layer_max = @max(batch.layer_max, layer);
            } else {
                try self.batch_table.put(key, .{
                    .start_index = i,
                    .sprite_count = 1,
                    .layer_min = layer,
                    .layer_max = layer,
                });
            }
        }

        var it = self.batch_table.iterator();
        var batch_count: u32 = 0;
        while (it.next()) |entry| {
            self.batch_keys[batch_count] = entry.key_ptr.*;
            batch_count += 1;
        }

        const SortContext = struct {
            table: *const BatchTable,
            pub fn lessThan(ctx: @This(), a: BatchKey, b: BatchKey) bool {
                const a_info = ctx.table.get(a).?;
                const b_info = ctx.table.get(b).?;
                return a_info.layer_min < b_info.layer_min;
            }
        };

        std.sort.pdq(BatchKey, self.batch_keys[0..batch_count], SortContext{ .table = &self.batch_table }, SortContext.lessThan);

        for (self.batch_keys[0..batch_count]) |key| {
            const info = self.batch_table.get(key).?;
            try self.draw_calls.append(.{
                .first_index = info.start_index * INDICES_PER_SPRITE,
                .index_count = info.sprite_count * INDICES_PER_SPRITE,
                .instance_count = 1,
                .texture_id = key.texture_id,
                .blend_mode = key.blend_mode,
            });
        }

        const vertex_size = @sizeOf(Vertex) * self.sprite_count * VERTICES_PER_SPRITE;
        const instance_size = @sizeOf(SpriteInstance) * self.instance_count;

        @memcpy(
            @as([*]u8, @ptrCast(@alignCast(self.mapped_vertices.?)))[0..vertex_size],
            @as([*]const u8, @ptrCast(@alignCast(self.vertices.ptr)))[0..vertex_size],
        );

        @memcpy(
            @as([*]u8, @ptrCast(@alignCast(self.mapped_instances.?)))[0..instance_size],
            @as([*]const u8, @ptrCast(@alignCast(self.instances.ptr)))[0..instance_size],
        );

        if (c.vkEndCommandBuffer(self.current_cmd.?) != c.VK_SUCCESS) {
            logger.err("spritebatch: failed to end command buffer recording", .{});
            return error.CommandBufferEndFailed;
        }

        self.cmd_pool.reset();

        const batch_efficiency = @as(f32, @floatFromInt(self.sprite_count)) / @as(f32, @floatFromInt(self.draw_calls.items.len));

        if (self.sprite_count > 10 and batch_efficiency < 5.0) {
            logger.warn("spritebatch: low batch efficiency ({d:.1} sprites/batch), check texture/material sorting", .{batch_efficiency});
        } else if (self.draw_calls.items.len > 1000) {
            logger.warn("spritebatch: excessive draw calls ({d}), consider batching or atlasing", .{self.draw_calls.items.len});
        }
    }

    pub fn draw(
        self: *SpriteBatch,
        position: [2]f32,
        size: [2]f32,
        rotation: f32,
        color: [4]f32,
        texture_id: u32,
        uv_rect: [4]f32,
        layer: f32,
        flags: u32,
    ) !void {
        if (self.sprite_count >= self.max_sprites) {
            logger.err("spritebatch: exceeded maximum sprite count\nPossible solutions:\n1. Increase capacity (current: {d})\n2. Split into multiple batches\n3. Enable instancing for similar sprites", .{self.max_sprites});
            return error.BatchFull;
        }

        if (size[0] * size[1] > 1000 * 1000) {
            logger.warn("spritebatch: large sprite detected ({d}x{d}), consider texture optimization", .{
                @as(u32, @intFromFloat(size[0])),
                @as(u32, @intFromFloat(size[1])),
            });
        }

        self.sort_keys[self.sprite_count] = .{
            .layer = @intFromFloat(layer * 65535.0),
            .texture_id = @intCast(texture_id & 0xFF),
            .blend_mode = @intCast((flags >> 28) & 0xF),
            .flags = @intCast((flags >> 24) & 0xF),
        };

        const half_width = size[0] * 0.5;
        const half_height = size[1] * 0.5;

        const cos_r = @cos(rotation);
        const sin_r = @sin(rotation);

        const transform_x = [2]f32{ cos_r * half_width, sin_r * half_width };
        const transform_y = [2]f32{ sin_r * half_height, -cos_r * half_height };

        const vertex_offset = self.sprite_count * VERTICES_PER_SPRITE;

        self.vertices[vertex_offset + 0] = .{
            .position = .{
                position[0] - transform_x[0] + transform_y[0],
                position[1] - transform_x[1] + transform_y[1],
            },
            .uv = .{ uv_rect[0], uv_rect[1] },
            .color = color,
        };

        self.vertices[vertex_offset + 1] = .{
            .position = .{
                position[0] + transform_x[0] + transform_y[0],
                position[1] + transform_x[1] + transform_y[1],
            },
            .uv = .{ uv_rect[2], uv_rect[1] },
            .color = color,
        };

        self.vertices[vertex_offset + 2] = .{
            .position = .{
                position[0] + transform_x[0] - transform_y[0],
                position[1] + transform_x[1] - transform_y[1],
            },
            .uv = .{ uv_rect[2], uv_rect[3] },
            .color = color,
        };

        self.vertices[vertex_offset + 3] = .{
            .position = .{
                position[0] - transform_x[0] - transform_y[0],
                position[1] - transform_x[1] - transform_y[1],
            },
            .uv = .{ uv_rect[0], uv_rect[3] },
            .color = color,
        };

        if (flags & SPRITE_FLAG_USE_INSTANCING != 0) {
            self.instances[self.instance_count] = .{
                .model = calculateModelMatrix(position, size, rotation),
                .color_and_uv = .{ color[0], color[1], color[2], color[3] },
                .metadata = .{
                    .layer = layer,
                    .texture_id = texture_id & 0xFF,
                    .flags = flags,
                    ._padding = 0,
                },
            };
            self.instance_count += 1;
        }

        self.sprite_count += 1;
    }

    pub const SPRITE_FLAG_USE_INSTANCING = 1 << 0;
    pub const SPRITE_FLAG_FLIP_X = 1 << 1;
    pub const SPRITE_FLAG_FLIP_Y = 1 << 2;
    pub const SPRITE_FLAG_NO_DEPTH = 1 << 3;

    pub const Error = error{
        BatchFull,
        CommandBufferBeginFailed,
        CommandBufferEndFailed,
        NoCommandBuffersAvailable,
    };
};
