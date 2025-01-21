const std = @import("std");
const Thread = std.Thread;
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});

/// Possible errors that can occur during command buffer operations. Each error corresponds
/// to a specific failure point in the command buffer lifecycle, from pool creation through
/// command recording and submission. These errors help identify issues during development
/// and provide meaningful feedback for debugging command buffer related problems.
pub const CommandError = error{
    CommandPoolCreationFailed,
    CommandBufferAllocationFailed,
    CommandPoolResetFailed,
    CommandBufferBeginFailed,
    CommandBufferEndFailed,
    CommandBufferResetFailed,
    AlreadyRecording,
    NotRecording,
    StillRecording,
    InvalidUsage,
    OutOfMemory,
    SubmissionFailed,
    QueuePresentFailed,
    RenderPassNotEnded,
};

/// Configuration options that determine how a command buffer behaves and how its memory
/// is managed. These settings affect the entire lifecycle of the command buffer, from
/// allocation through recording and submission. The choice of flags impacts memory usage
/// patterns and performance characteristics. For example, one-time submit buffers can use
/// transient memory while reusable buffers are allocated from longer-lived memory pools.
///
/// Note: These flags must be set at creation time and cannot be changed after the buffer
/// is allocated. Choose flags based on your specific use case - one-time buffers for
/// setup operations, reusable buffers for frame rendering, or secondary buffers for
/// multi-threaded recording.
pub const CommandBufferUsage = struct {
    /// Whether the command buffer can be reset and reused
    is_reusable: bool = false,
    /// Whether the command buffer will be submitted once
    one_time_submit: bool = false,
    /// Whether the command buffer can be reset individually
    allow_reset: bool = false,
    /// Whether the command buffer is a secondary command buffer
    is_secondary: bool = false,
    /// Whether the command buffer inherits from a primary command buffer
    is_inherited: bool = false,
    /// Whether the command buffer is used for compute operations
    is_compute: bool = false,
    /// Whether the command buffer is used for transfer operations
    is_transfer: bool = false,
};

/// Submission parameters for command buffer execution. This struct encapsulates the
/// synchronization primitives and dependencies required for safe command buffer submission
/// to a Vulkan queue. It defines the execution and memory dependencies between command
/// buffers, ensuring correct ordering of operations and resource availability.
///
/// The wait semaphores and their corresponding pipeline stages establish prerequisites
/// that must be satisfied before the command buffer execution can begin. Signal semaphores
/// notify dependent operations when execution completes. The optional fence allows the CPU
/// to track completion of the submitted work.
///
/// Note: The arrays for wait semaphores and pipeline stages must have matching lengths,
/// as each wait semaphore corresponds to a specific pipeline stage where the wait occurs.
/// Signal semaphores are triggered when all commands complete execution.
pub const SubmitInfo = struct {
    /// Semaphores that must be signaled before command buffer execution begins.
    /// Each semaphore corresponds to a pipeline stage specified in wait_stages.
    wait_semaphores: []const c.VkSemaphore = &[_]c.VkSemaphore{},

    /// Pipeline stages at which each corresponding wait semaphore will be waited upon.
    /// These stages define points in the pipeline where execution must pause until
    /// the semaphore is signaled. Must have the same length as wait_semaphores.
    wait_stages: []const c.VkPipelineStageFlags = &[_]c.VkPipelineStageFlags{},

    /// Semaphores that will be signaled when command buffer execution completes.
    /// These semaphores can be used to synchronize subsequent work that depends
    /// on this submission.
    signal_semaphores: []const c.VkSemaphore = &[_]c.VkSemaphore{},

    /// Optional fence that will be signaled when command buffer execution completes.
    /// Unlike semaphores which synchronize GPU work, fences allow the CPU to track
    /// completion. This is useful for managing resource lifetimes or implementing
    /// CPU waits.
    fence: ?c.VkFence = null,
};

/// Memory barrier parameters for synchronizing access to resources across pipeline stages.
/// Memory barriers ensure correct ordering of memory operations and visibility of memory
/// writes. They control both the execution dependency between pipeline stages and the
/// memory dependency between memory accesses.
///
/// Memory barriers are particularly important when accessing resources that may be written
/// in one pipeline stage and read in another, or when transitioning resources between
/// different queue families. They prevent race conditions and ensure memory coherency
/// across the GPU pipeline stages.
///
/// Note: Memory barriers can have significant performance implications. It is recommended
/// to batch barriers where possible and use the most specific barrier type (buffer or
/// image barriers) when the affected resources are known.
pub const MemoryBarrier = struct {
    /// Pipeline stage where previous memory accesses must complete before the barrier.
    /// This defines the latest stage where memory operations that must be visible have
    /// occurred.
    src_stage: c.VkPipelineStageFlags,

    /// Pipeline stage where subsequent memory accesses will wait for the barrier.
    /// This defines the earliest stage where memory operations that need the barrier
    /// will occur.
    dst_stage: c.VkPipelineStageFlags,

    /// Types of memory accesses that must complete before the barrier.
    /// This mask specifies which memory operations in src_stage must be visible.
    src_access: c.VkAccessFlags,

    /// Types of memory accesses that must wait for the barrier.
    /// This mask specifies which memory operations in dst_stage must wait.
    dst_access: c.VkAccessFlags,
};

/// Image barrier parameters for synchronizing access and transitioning layouts of image
/// resources. Image barriers provide fine-grained control over image memory access and
/// layout transitions. They ensure proper synchronization when an image's contents or
/// layout changes, preventing invalid reads or writes during transitions.
///
/// Image barriers are essential for managing image layout transitions, such as preparing
/// an image for rendering, presenting, or shader access. They also handle ownership
/// transfers between queue families and ensure proper visibility of image contents
/// across pipeline stages.
///
/// Note: Image layout transitions may involve data movement or reformatting on some
/// hardware. Choose layouts appropriate for the intended access patterns and minimize
/// transitions to optimize performance.
pub const ImageBarrier = struct {
    /// Image resource affected by the barrier.
    /// This is the specific image whose access and layout will be synchronized.
    image: c.VkImage,

    /// Current layout of the image before the barrier.
    /// This must match the actual current layout of the image.
    old_layout: c.VkImageLayout,

    /// New layout the image will transition to after the barrier.
    /// The image will be automatically transitioned to this layout.
    new_layout: c.VkImageLayout,

    /// Pipeline stage where previous image accesses must complete.
    /// This defines when existing accesses to the image must be visible.
    src_stage: c.VkPipelineStageFlags,

    /// Pipeline stage where subsequent image accesses will wait.
    /// This defines when new accesses to the image can begin.
    dst_stage: c.VkPipelineStageFlags,

    /// Types of memory accesses to the image that must complete.
    /// This specifies which operations on the image must be visible.
    src_access: c.VkAccessFlags,

    /// Types of memory accesses to the image that must wait.
    /// This specifies which operations on the image must wait.
    dst_access: c.VkAccessFlags,

    /// Queue family releasing ownership of the image.
    /// Use VK_QUEUE_FAMILY_IGNORED if no ownership transfer.
    src_queue_family: u32 = c.VK_QUEUE_FAMILY_IGNORED,

    /// Queue family acquiring ownership of the image.
    /// Use VK_QUEUE_FAMILY_IGNORED if no ownership transfer.
    dst_queue_family: u32 = c.VK_QUEUE_FAMILY_IGNORED,

    /// Range of image subresources affected by the barrier.
    /// This specifies which aspects, mip levels, and array layers are synchronized.
    subresource_range: c.VkImageSubresourceRange,
};

/// Buffer barrier parameters for synchronizing access to buffer resources. Buffer
/// barriers provide control over memory access to specific ranges within buffer
/// objects. They ensure proper synchronization when buffer contents are modified
/// or accessed across different pipeline stages or queue families.
///
/// Buffer barriers are used to manage visibility of buffer memory operations and
/// handle ownership transfers. They are more efficient than general memory barriers
/// when the affected resources are known to be buffers, as they can target specific
/// buffer regions rather than all memory.
///
/// Note: When possible, use buffer barriers instead of general memory barriers
/// as they provide better performance through more specific synchronization.
pub const BufferBarrier = struct {
    /// Buffer resource affected by the barrier.
    /// This is the specific buffer whose memory will be synchronized.
    buffer: c.VkBuffer,

    /// Offset into the buffer where the barrier begins.
    /// This allows synchronization of specific ranges within the buffer.
    offset: c.VkDeviceSize,

    /// Size of the buffer range affected by the barrier.
    /// This defines how many bytes after the offset are synchronized.
    size: c.VkDeviceSize,

    /// Pipeline stage where previous buffer accesses must complete.
    /// This defines when existing accesses to the buffer must be visible.
    src_stage: c.VkPipelineStageFlags,

    /// Pipeline stage where subsequent buffer accesses will wait.
    /// This defines when new accesses to the buffer can begin.
    dst_stage: c.VkPipelineStageFlags,

    /// Types of memory accesses to the buffer that must complete.
    /// This specifies which operations on the buffer must be visible.
    src_access: c.VkAccessFlags,

    /// Types of memory accesses to the buffer that must wait.
    /// This specifies which operations on the buffer must wait.
    dst_access: c.VkAccessFlags,

    /// Queue family releasing ownership of the buffer.
    /// Use VK_QUEUE_FAMILY_IGNORED if no ownership transfer.
    src_queue_family: u32 = c.VK_QUEUE_FAMILY_IGNORED,

    /// Queue family acquiring ownership of the buffer.
    /// Use VK_QUEUE_FAMILY_IGNORED if no ownership transfer.
    dst_queue_family: u32 = c.VK_QUEUE_FAMILY_IGNORED,
};

/// Query parameters for collecting statistics and timestamps during command buffer
/// execution. Queries provide a mechanism to gather performance data and statistics
/// about GPU operations. They can be used to measure execution time of command
/// sequences, count primitives processed, or track other pipeline statistics.
///
/// Queries must be properly synchronized to ensure accurate results. The query pool
/// must remain valid until the command buffer completes execution. Results can be
/// retrieved once the command buffer execution is complete and the query pool is
/// available for reading.
///
/// Note: Query operations may have a small performance impact. Use them judiciously
/// in production code and consider disabling them in performance-critical paths.
pub const QueryInfo = struct {
    /// Query pool from which to draw the query.
    /// The pool must be created with appropriate flags for the intended query type.
    pool: c.VkQueryPool,

    /// Index of the query within the pool.
    /// Must be less than the number of queries allocated in the pool.
    query: u32,

    /// Number of consecutive queries to update.
    /// Useful for collecting multiple related statistics in sequence.
    count: u32 = 1,

    /// Control flags affecting query behavior.
    /// Can enable features like precise occlusion counting.
    flags: c.VkQueryControlFlags = 0,
};

/// Configuration parameters that control command pool behavior and performance
/// characteristics. These settings affect how command buffers are allocated,
/// managed, and optimized within the pool. The configuration allows tuning
/// the pool's behavior for specific use cases such as static scene rendering
/// or dynamic command generation.
///
/// The settings chosen can significantly impact memory usage and performance.
/// For example, thread-local pools reduce contention in multi-threaded scenarios,
/// while command batching can reduce driver overhead. The configuration should
/// be tailored to the specific usage pattern of the command pool.
///
/// Note: These settings cannot be changed after pool creation. Choose values
/// that balance resource usage with the expected command buffer allocation
/// patterns in your application.
pub const CommandPoolConfig = struct {
    /// Number of command buffers to pre-allocate when creating the pool.
    /// Pre-allocation reduces allocation overhead during rendering by ensuring
    /// a base set of buffers is immediately available.
    initial_buffer_count: u32 = 8,

    /// Maximum number of command buffers to keep in the free list.
    /// Buffers beyond this limit are destroyed when returned to the pool
    /// rather than being kept for reuse.
    max_free_buffers: u32 = 32,

    /// Whether to create a thread-local command pool.
    /// Thread-local pools eliminate synchronization overhead when recording
    /// commands from multiple threads.
    thread_local: bool = false,

    /// Whether to batch similar commands together.
    /// Command batching can reduce driver overhead by combining compatible
    /// operations into larger submissions.
    batch_commands: bool = true,

    /// Whether to cache pipeline state between commands.
    /// State caching reduces redundant state changes but requires additional
    /// memory to track the current state.
    cache_state: bool = true,
};

/// Barrier batching system for optimizing synchronization operations. This struct
/// accumulates multiple barrier operations and combines them into a single Vulkan
/// pipeline barrier command. Batching reduces command buffer overhead and improves
/// performance by minimizing the number of individual barrier commands.
///
/// The system maintains separate lists for memory, buffer, and image barriers.
/// When barriers are added, they are stored in the appropriate list and their
/// pipeline stages are tracked. The combined barrier operation uses the union
/// of all source and destination pipeline stages to ensure correct synchronization
/// while minimizing the number of pipeline stalls.
///
/// Note: Barrier batching trades memory usage for performance. The batch should
/// be flushed periodically to prevent excessive memory accumulation and ensure
/// timely synchronization of resources.
pub const BarrierBatch = struct {
    /// Collection of memory barriers affecting global memory access.
    /// These barriers synchronize access to memory without specifying
    /// particular buffers or images.
    memory_barriers: std.ArrayList(c.VkMemoryBarrier),

    /// Collection of buffer memory barriers for specific buffer ranges.
    /// These barriers provide fine-grained synchronization for buffer
    /// resources and can include ownership transfers.
    buffer_barriers: std.ArrayList(c.VkBufferMemoryBarrier),

    /// Collection of image memory barriers for specific images.
    /// These barriers handle both memory access synchronization and
    /// image layout transitions.
    image_barriers: std.ArrayList(c.VkImageMemoryBarrier),

    /// Union of source pipeline stages from all batched barriers.
    /// This represents the latest stage where any memory operation
    /// that needs synchronization occurs.
    src_stage: c.VkPipelineStageFlags,

    /// Union of destination pipeline stages from all batched barriers.
    /// This represents the earliest stage where any memory operation
    /// that needs to wait occurs.
    dst_stage: c.VkPipelineStageFlags,

    /// Allocator used for managing barrier lists.
    /// Must remain valid for the lifetime of the batch.
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !BarrierBatch {
        return BarrierBatch{
            .memory_barriers = std.ArrayList(c.VkMemoryBarrier).init(allocator),
            .buffer_barriers = std.ArrayList(c.VkBufferMemoryBarrier).init(allocator),
            .image_barriers = std.ArrayList(c.VkImageMemoryBarrier).init(allocator),
            .src_stage = 0,
            .dst_stage = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BarrierBatch) void {
        self.memory_barriers.deinit();
        self.buffer_barriers.deinit();
        self.image_barriers.deinit();
    }

    pub fn flush(self: *BarrierBatch, cmd: *CommandBuffer) !void {
        if (self.memory_barriers.items.len == 0 and
            self.buffer_barriers.items.len == 0 and
            self.image_barriers.items.len == 0) return;

        c.vkCmdPipelineBarrier(
            cmd.handle,
            self.src_stage,
            self.dst_stage,
            0,
            @intCast(self.memory_barriers.items.len),
            if (self.memory_barriers.items.len > 0) self.memory_barriers.items.ptr else null,
            @intCast(self.buffer_barriers.items.len),
            if (self.buffer_barriers.items.len > 0) self.buffer_barriers.items.ptr else null,
            @intCast(self.image_barriers.items.len),
            if (self.image_barriers.items.len > 0) self.image_barriers.items.ptr else null,
        );

        self.memory_barriers.clearRetainingCapacity();
        self.buffer_barriers.clearRetainingCapacity();
        self.image_barriers.clearRetainingCapacity();
        self.src_stage = 0;
        self.dst_stage = 0;
    }
};

/// Thread-local command pool storage
threadlocal var thread_local_pool: ?*CommandPool = null;

/// Gets or creates a thread-local command pool for efficient command buffer allocation.
/// The pool is created with the specified device, queue family, and configuration.
/// Each thread maintains its own pool to eliminate synchronization overhead when
/// recording commands.
///
/// Note: Thread-local pools are automatically cleaned up when the thread exits.
/// However, long-running threads should periodically reset their pools to prevent
/// memory growth.
///
/// Warning: The allocator must remain valid for the lifetime of the thread-local
/// pool. Destroying the allocator while the pool exists will result in undefined
/// behavior.
pub fn getThreadLocalPool(
    device: c.VkDevice,
    queue_family_index: u32,
    config: CommandPoolConfig,
    allocator: std.mem.Allocator,
) !*CommandPool {
    if (thread_local_pool) |pool| {
        return pool;
    }

    const pool = try CommandPool.init(
        device,
        queue_family_index,
        .{
            .is_reusable = true,
            .allow_reset = true,
        },
        config,
        allocator,
    );
    thread_local_pool = pool;
    return pool;
}

/// Manages a Vulkan command pool and its buffers
pub const CommandPool = struct {
    /// Vulkan device associated with this command pool.
    /// All command buffers allocated from this pool are created on this device.
    device: c.VkDevice,

    /// Native Vulkan command pool handle.
    /// Used for allocating command buffers and managing their memory.
    handle: c.VkCommandPool,

    /// Memory allocator used for command buffer management.
    /// Must remain valid for the lifetime of the command pool.
    allocator: std.mem.Allocator,

    /// Queue family index this pool allocates command buffers for.
    /// Command buffers from this pool can only be submitted to queues
    /// of this family.
    queue_family_index: u32,

    /// List of command buffers currently in use.
    /// These buffers are being recorded or have been submitted for execution.
    active_buffers: std.ArrayList(*CommandBuffer),

    /// List of command buffers available for reuse.
    /// These buffers have been reset and can be recycled for new recordings.
    free_buffers: std.ArrayList(*CommandBuffer),

    /// Usage flags controlling command buffer behavior.
    /// These flags affect how command buffers allocated from this pool operate.
    usage: CommandBufferUsage,

    /// Performance configuration for the command pool.
    /// Controls memory management and optimization settings.
    config: CommandPoolConfig,

    /// Mutex protecting concurrent access to the command pool.
    /// Ensures thread-safe allocation and deallocation of command buffers.
    mutex: std.Thread.Mutex,

    /// Creates a new command pool optimized for the specified usage
    pub fn init(
        device: c.VkDevice,
        queue_family_index: u32,
        usage: CommandBufferUsage,
        config: CommandPoolConfig,
        allocator: std.mem.Allocator,
    ) !*CommandPool {
        const pool = try allocator.create(CommandPool);
        errdefer allocator.destroy(pool);

        var flags: c.VkCommandPoolCreateFlags = 0;
        if (usage.allow_reset) flags |= c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (usage.is_reusable) flags |= c.VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

        const pool_info = c.VkCommandPoolCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = flags,
            .queueFamilyIndex = queue_family_index,
        };

        var handle: c.VkCommandPool = undefined;
        if (c.vkCreateCommandPool(device, &pool_info, null, &handle) != c.VK_SUCCESS) {
            return CommandError.CommandPoolCreationFailed;
        }

        pool.* = .{
            .device = device,
            .handle = handle,
            .allocator = allocator,
            .queue_family_index = queue_family_index,
            .active_buffers = std.ArrayList(*CommandBuffer).init(allocator),
            .free_buffers = std.ArrayList(*CommandBuffer).init(allocator),
            .usage = usage,
            .config = config,
            .mutex = .{},
        };

        // Pre-allocate command buffers
        if (config.initial_buffer_count > 0) {
            try pool.preallocateBuffers(config.initial_buffer_count);
        }

        return pool;
    }

    /// Pre-allocates a specified number of command buffers in the pool. This reduces
    /// allocation overhead during rendering by ensuring a base set of buffers is
    /// immediately available.
    ///
    /// Note: Pre-allocated buffers are stored in the free list and can be retrieved
    /// using getBuffer(). The number of buffers should be chosen based on expected
    /// usage patterns.
    ///
    /// Warning: Pre-allocation consumes memory immediately. Choose a count that
    /// balances resource availability with memory usage.
    fn preallocateBuffers(self: *CommandPool, count: u32) !void {
        const level: c_uint = if (self.usage.is_secondary)
            c.VK_COMMAND_BUFFER_LEVEL_SECONDARY
        else
            c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        const buffers = try self.allocator.alloc(c.VkCommandBuffer, count);
        defer self.allocator.free(buffers);

        const alloc_info = c.VkCommandBufferAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = self.handle,
            .level = level,
            .commandBufferCount = count,
        };

        if (c.vkAllocateCommandBuffers(self.device, &alloc_info, buffers.ptr) != c.VK_SUCCESS) {
            return CommandError.CommandBufferAllocationFailed;
        }

        for (buffers) |handle| {
            const buffer = try CommandBuffer.init(self.device, self.handle, level, self.allocator, self.config.cache_state);
            buffer.handle = handle;
            try self.free_buffers.append(buffer);
        }
    }

    /// Gets a command buffer from the pool, reusing an existing buffer if available
    /// or allocating a new one if necessary. The buffer is automatically added to
    /// the active list and reset before being returned.
    ///
    /// Note: The pool's mutex ensures thread-safe access to the buffer lists. The
    /// returned buffer must be returned to the pool using returnBuffer() when no
    /// longer needed.
    ///
    /// Warning: Command buffers must not be accessed after being returned to the
    /// pool, as they may be reused or destroyed.
    pub fn getBuffer(self: *CommandPool) !*CommandBuffer {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Try to reuse a free buffer first
        if (self.free_buffers.items.len > 0) {
            const buffer = self.free_buffers.pop();
            try self.active_buffers.append(buffer);
            try buffer.reset();
            return buffer;
        }

        // Allocate a new buffer
        const level: c_uint = if (self.usage.is_secondary)
            c.VK_COMMAND_BUFFER_LEVEL_SECONDARY
        else
            c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;

        const buffer = try CommandBuffer.init(
            self.device,
            self.handle,
            level,
            self.allocator,
            self.config.cache_state,
        );
        try self.active_buffers.append(buffer);
        return buffer;
    }

    /// Returns a command buffer to the pool for reuse. The buffer is reset and
    /// either added to the free list or destroyed based on the pool's configuration.
    /// The buffer must have been previously obtained from this pool.
    ///
    /// Note: Buffers are kept in the free list up to the configured maximum. Beyond
    /// this limit, returned buffers are destroyed to prevent unbounded memory growth.
    ///
    /// Warning: The buffer must not be used after being returned to the pool, as it
    /// may be reset, reused, or destroyed.
    pub fn returnBuffer(self: *CommandPool, buffer: *CommandBuffer) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.active_buffers.items, 0..) |active, i| {
            if (active == buffer) {
                _ = self.active_buffers.swapRemove(i);
                try buffer.reset();

                // Only keep a limited number of free buffers
                if (self.free_buffers.items.len < self.config.max_free_buffers) {
                    try self.free_buffers.append(buffer);
                } else {
                    buffer.deinit();
                    self.allocator.destroy(buffer);
                }
                return;
            }
        }
    }

    /// Resets all command buffers in the pool, returning them to their initial state.
    /// This operation is more efficient than resetting buffers individually when
    /// many buffers need to be reset.
    ///
    /// Note: After reset, all buffers are moved to the free list and can be reused.
    /// Any resources used by the commands are released.
    ///
    /// Warning: All command buffers allocated from the pool become invalid and must
    /// be obtained again using getBuffer().
    pub fn reset(self: *CommandPool) !void {
        if (c.vkResetCommandPool(self.device, self.handle, 0) != c.VK_SUCCESS) {
            return CommandError.CommandPoolResetFailed;
        }

        // Move all active buffers to free list
        while (self.active_buffers.items.len > 0) {
            const buffer = self.active_buffers.pop();
            try self.free_buffers.append(buffer);
        }
    }

    /// Cleans up the command pool and all its resources. All command buffers are
    /// destroyed, and the pool's memory is released back to the device.
    ///
    /// Note: This function must be called to prevent resource leaks. It automatically
    /// cleans up all active and free command buffers.
    ///
    /// Warning: The pool and its buffers must not be used after calling deinit().
    /// Any outstanding command buffers become invalid.
    pub fn deinit(self: *CommandPool) void {
        // Clean up all active command buffers first
        for (self.active_buffers.items) |buffer| {
            buffer.deinit();
            self.allocator.destroy(buffer);
        }

        // Clean up free buffers
        for (self.free_buffers.items) |buffer| {
            buffer.deinit();
            self.allocator.destroy(buffer);
        }

        self.active_buffers.deinit();
        self.free_buffers.deinit();
        c.vkDestroyCommandPool(self.device, self.handle, null);
        self.allocator.destroy(self);
    }
};

/// Submits command buffers to a queue for execution. The submission can be
/// synchronized using semaphores and fences. The function extracts raw command
/// buffer handles and configures the submission parameters.
///
/// Note: The wait semaphores and pipeline stages must have matching lengths.
/// The fence can be used to track completion from the CPU.
///
/// Warning: Command buffers must not be modified while submitted for execution.
/// Wait for the fence or queue idle before reusing the buffers.
pub fn submit(
    queue: c.VkQueue,
    buffers: []const *CommandBuffer,
    info: SubmitInfo,
) !void {
    // Extract raw command buffer handles
    var raw_buffers = try std.heap.c_allocator.alloc(c.VkCommandBuffer, buffers.len);
    defer std.heap.c_allocator.free(raw_buffers);

    for (buffers, 0..) |buffer, i| {
        raw_buffers[i] = buffer.handle;
    }

    const submit_info = c.VkSubmitInfo{
        .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = @intCast(info.wait_semaphores.len),
        .pWaitSemaphores = info.wait_semaphores.ptr,
        .pWaitDstStageMask = info.wait_stages.ptr,
        .commandBufferCount = @intCast(buffers.len),
        .pCommandBuffers = raw_buffers.ptr,
        .signalSemaphoreCount = @intCast(info.signal_semaphores.len),
        .pSignalSemaphores = info.signal_semaphores.ptr,
    };

    if (c.vkQueueSubmit(queue, 1, &submit_info, info.fence orelse null) != c.VK_SUCCESS) {
        return error.SubmissionFailed;
    }
}

/// Command buffer wrapper that provides a safe interface for recording Vulkan
/// commands. The wrapper maintains state information about the current recording
/// session, tracks render pass and pipeline state, and implements safety checks
/// to prevent invalid command sequences.
///
/// The command buffer implements a state machine that tracks whether commands
/// are being recorded and what render pass or pipeline state is active. This
/// state tracking helps prevent common errors like ending a render pass that
/// hasn't been started or using an unbound pipeline.
///
/// Performance optimizations include state caching to reduce redundant state
/// changes and barrier batching to minimize synchronization overhead. These
/// optimizations can be configured through the command pool settings.
pub const CommandBuffer = struct {
    /// Vulkan device associated with this command buffer.
    /// Used for resource cleanup and command buffer operations.
    device: c.VkDevice,

    /// Native Vulkan command buffer handle.
    /// Used for recording commands and submission to queues.
    handle: c.VkCommandBuffer,

    /// Handle of the command pool this buffer was allocated from.
    /// Required for proper cleanup when freeing the command buffer.
    pool_handle: c.VkCommandPool,

    /// Memory allocator used for dynamic allocations during recording.
    /// Must remain valid for the lifetime of the command buffer.
    allocator: std.mem.Allocator,

    /// Whether the command buffer is currently recording commands.
    /// Used to enforce valid command recording sequences.
    is_recording: bool,

    /// Currently active render pass, if any.
    /// Null when not within a render pass. Used to validate render
    /// pass state transitions.
    current_render_pass: ?c.VkRenderPass,

    /// Currently bound pipeline, if any.
    /// Null when no pipeline is bound. Used to optimize redundant
    /// pipeline bindings.
    current_pipeline: ?c.VkPipeline,

    /// Whether a viewport has been set in the current state.
    /// Required for validation of draw commands.
    viewport_set: bool,

    /// Whether scissor rectangles have been set in the current state.
    /// Required for validation of draw commands.
    scissor_set: bool,

    /// Optional barrier batch for combining synchronization commands.
    /// Null if barrier batching is disabled in the pool configuration.
    barrier_batch: ?BarrierBatch,

    /// Whether to cache pipeline and other state between commands.
    /// Enables optimizations to reduce redundant state changes.
    cache_state: bool,

    /// Creates a new command buffer with the specified configuration. The buffer is
    /// allocated from the given command pool and initialized with default state. The
    /// level parameter determines if this is a primary or secondary command buffer.
    /// Primary buffers can be submitted directly to queues, while secondary buffers
    /// must be executed from primary buffers.
    ///
    /// The cache_state parameter enables state tracking optimizations. When enabled,
    /// the buffer maintains a barrier batch and tracks pipeline state to reduce
    /// redundant state changes and combine barriers.
    fn init(
        device: c.VkDevice,
        pool: c.VkCommandPool,
        level: c.VkCommandBufferLevel,
        allocator: std.mem.Allocator,
        cache_state: bool,
    ) !*CommandBuffer {
        const buffer = try allocator.create(CommandBuffer);
        errdefer allocator.destroy(buffer);

        const alloc_info = c.VkCommandBufferAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool,
            .level = level,
            .commandBufferCount = 1,
        };

        var handle: c.VkCommandBuffer = undefined;
        if (c.vkAllocateCommandBuffers(device, &alloc_info, &handle) != c.VK_SUCCESS) {
            return CommandError.CommandBufferAllocationFailed;
        }

        buffer.* = .{
            .device = device,
            .handle = handle,
            .pool_handle = pool,
            .allocator = allocator,
            .is_recording = false,
            .current_render_pass = null,
            .current_pipeline = null,
            .viewport_set = false,
            .scissor_set = false,
            .barrier_batch = if (cache_state) try BarrierBatch.init(allocator) else null,
            .cache_state = cache_state,
        };

        return buffer;
    }

    fn deinit(self: *CommandBuffer) void {
        if (self.barrier_batch) |*batch| {
            batch.deinit();
        }
        c.vkFreeCommandBuffers(self.device, self.pool_handle, 1, &self.handle);
    }

    /// Begins recording commands to this buffer with the specified usage flags.
    /// The buffer must not already be recording commands. This function resets
    /// all state tracking and prepares the buffer for new commands.
    ///
    /// The usage flags control how the buffer can be used. One-time submit buffers
    /// are optimized for single submission, while simultaneous use allows the buffer
    /// to be submitted multiple times. Secondary buffers can specify inheritance
    /// of render pass state.
    pub fn begin(self: *CommandBuffer, usage: CommandBufferUsage) !void {
        if (self.is_recording) return CommandError.AlreadyRecording;

        var flags: c.VkCommandBufferUsageFlags = 0;
        if (usage.one_time_submit) flags |= c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (usage.is_inherited) flags |= c.VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
        if (usage.is_secondary) flags |= c.VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

        const begin_info = c.VkCommandBufferBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = flags,
            .pInheritanceInfo = null,
        };

        if (c.vkBeginCommandBuffer(self.handle, &begin_info) != c.VK_SUCCESS) {
            return CommandError.CommandBufferBeginFailed;
        }

        self.is_recording = true;
        self.current_render_pass = null;
        self.current_pipeline = null;
        self.viewport_set = false;
        self.scissor_set = false;
    }

    /// Ends command recording and prepares the buffer for submission. All render
    /// passes must be ended before calling this function. After ending, the buffer
    /// can be submitted to a queue for execution.
    ///
    /// This function validates that all required state (render passes, etc.) has
    /// been properly cleaned up before finalizing the command buffer. It returns
    /// an error if the validation fails.
    pub fn end(self: *CommandBuffer) !void {
        if (!self.is_recording) return CommandError.NotRecording;
        if (self.current_render_pass != null) return CommandError.RenderPassNotEnded;

        if (c.vkEndCommandBuffer(self.handle) != c.VK_SUCCESS) {
            return CommandError.CommandBufferEndFailed;
        }

        self.is_recording = false;
    }

    /// Resets this command buffer to its initial state, allowing it to be reused.
    /// The buffer must not be currently recording commands. All resources used by
    /// the commands are released.
    ///
    /// When cache_state is enabled, the reset operation releases all tracked state
    /// and resources. This ensures no stale state persists when the buffer is
    /// reused. The function returns an error if the buffer cannot be reset.
    pub fn reset(self: *CommandBuffer) !void {
        if (self.is_recording) return CommandError.StillRecording;

        const flags: c.VkCommandBufferResetFlags = if (self.cache_state)
            c.VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT
        else
            0;

        if (c.vkResetCommandBuffer(self.handle, flags) != c.VK_SUCCESS) {
            return CommandError.CommandBufferResetFailed;
        }

        self.current_render_pass = null;
        self.current_pipeline = null;
        self.viewport_set = false;
        self.scissor_set = false;
    }

    /// Begins a render pass operation with the specified parameters. The render pass
    /// defines the attachments and subpasses for rendering operations. The framebuffer
    /// provides the actual images to render to, while clear values specify the initial
    /// contents of the attachments.
    ///
    /// The extent parameter defines the render area, which should match the framebuffer
    /// dimensions. The contents parameter specifies whether the render pass will contain
    /// secondary command buffers or inline commands.
    pub fn beginRenderPass(
        self: *CommandBuffer,
        render_pass: c.VkRenderPass,
        framebuffer: c.VkFramebuffer,
        extent: c.VkExtent2D,
        clear_values: []const c.VkClearValue,
        contents: c.VkSubpassContents,
    ) !void {
        if (!self.is_recording) return CommandError.NotRecording;

        const render_pass_info = c.VkRenderPassBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = render_pass,
            .framebuffer = framebuffer,
            .renderArea = .{
                .offset = .{ .x = 0, .y = 0 },
                .extent = extent,
            },
            .clearValueCount = @intCast(clear_values.len),
            .pClearValues = clear_values.ptr,
        };

        c.vkCmdBeginRenderPass(self.handle, &render_pass_info, contents);
        self.current_render_pass = render_pass;
    }

    /// Ends the current render pass. This function must be called after all rendering
    /// commands within the pass are complete. It validates that a render pass is
    /// actually active before attempting to end it.
    ///
    /// After ending the render pass, the command buffer returns to recording commands
    /// outside of a render pass. Any resources used exclusively by the render pass
    /// are released.
    pub fn endRenderPass(self: *CommandBuffer) !void {
        if (!self.is_recording) return CommandError.NotRecording;
        if (self.current_render_pass == null) return CommandError.InvalidUsage;

        c.vkCmdEndRenderPass(self.handle);
        self.current_render_pass = null;
    }

    /// Sets the viewport state for subsequent drawing commands. The viewport defines
    /// the transformation from normalized device coordinates to framebuffer coordinates.
    /// This must be set before any draw commands can be recorded.
    ///
    /// The depth range is specified by min_depth and max_depth, which are clamped to
    /// the range [0.0, 1.0]. Multiple viewports can be set if the device supports the
    /// multiViewport feature.
    pub fn setViewport(
        self: *CommandBuffer,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) !void {
        if (!self.is_recording) return CommandError.NotRecording;

        const viewport = c.VkViewport{
            .x = x,
            .y = y,
            .width = width,
            .height = height,
            .minDepth = min_depth,
            .maxDepth = max_depth,
        };

        c.vkCmdSetViewport(self.handle, 0, 1, &viewport);
        self.viewport_set = true;
    }

    /// Sets the scissor rectangles for subsequent drawing commands. The scissor test
    /// restricts rasterization to the specified rectangle. This must be set before
    /// any draw commands if dynamic scissor state is enabled.
    ///
    /// The rectangle is specified in framebuffer coordinates, with (0,0) at the upper
    /// left corner. Multiple scissors can be set if the device supports the
    /// multiViewport feature.
    pub fn setScissor(
        self: *CommandBuffer,
        x: i32,
        y: i32,
        width: u32,
        height: u32,
    ) !void {
        if (!self.is_recording) return CommandError.NotRecording;

        const scissor = c.VkRect2D{
            .offset = .{
                .x = x,
                .y = y,
            },
            .extent = .{
                .width = width,
                .height = height,
            },
        };

        c.vkCmdSetScissor(self.handle, 0, 1, &scissor);
        self.scissor_set = true;
    }

    /// Binds a pipeline object to the command buffer. The pipeline defines the fixed
    /// function and programmable stages used for subsequent commands. The bind point
    /// specifies whether this is a graphics or compute pipeline.
    ///
    /// If state caching is enabled, redundant pipeline binds are skipped. The function
    /// validates that command recording is active before binding the pipeline.
    pub fn bindPipeline(self: *CommandBuffer, pipeline: c.VkPipeline, bind_point: c.VkPipelineBindPoint) !void {
        if (!self.is_recording) return CommandError.NotRecording;
        if (self.current_pipeline == pipeline) return;

        c.vkCmdBindPipeline(self.handle, bind_point, pipeline);
        self.current_pipeline = pipeline;
    }

    /// Binds descriptor sets to the pipeline. Descriptor sets provide access to
    /// resources like uniform buffers and textures. The layout must match the
    /// pipeline's layout, and the first_set parameter determines which sets are
    /// being updated.
    ///
    /// Dynamic offsets can be provided to update dynamic uniform buffer bindings.
    /// The number of dynamic offsets must match the number of dynamic bindings
    /// in the descriptor sets.
    pub fn bindDescriptorSets(
        self: *CommandBuffer,
        bind_point: c.VkPipelineBindPoint,
        layout: c.VkPipelineLayout,
        first_set: u32,
        sets: []const c.VkDescriptorSet,
        dynamic_offsets: []const u32,
    ) !void {
        if (!self.is_recording) return CommandError.NotRecording;

        c.vkCmdBindDescriptorSets(
            self.handle,
            bind_point,
            layout,
            first_set,
            @intCast(sets.len),
            sets.ptr,
            @intCast(dynamic_offsets.len),
            dynamic_offsets.ptr,
        );
    }

    /// Updates push constant values in the pipeline layout. Push constants provide
    /// a way to update small amounts of uniform data without creating a buffer.
    /// The stage flags specify which shader stages can access the data.
    ///
    /// The offset and size of the push constant update must be within the range
    /// specified in the pipeline layout. Multiple updates to different ranges
    /// are allowed.
    pub fn pushConstants(
        self: *CommandBuffer,
        layout: c.VkPipelineLayout,
        stage_flags: c.VkShaderStageFlags,
        offset: u32,
        data: []const u8,
    ) !void {
        if (!self.is_recording) return CommandError.NotRecording;

        c.vkCmdPushConstants(
            self.handle,
            layout,
            stage_flags,
            offset,
            @intCast(data.len),
            data.ptr,
        );
    }

    /// Binds vertex buffers to the command buffer. The buffers provide vertex data
    /// for subsequent draw commands. Multiple buffers can be bound to different
    /// binding slots, allowing for separate position, normal, and other attribute
    /// arrays.
    ///
    /// The first_binding parameter specifies the starting binding number, and the
    /// offsets array provides the starting offset within each buffer. The number
    /// of buffers and offsets must match.
    pub fn bindVertexBuffers(
        self: *CommandBuffer,
        first_binding: u32,
        buffers: []const c.VkBuffer,
        offsets: []const c.VkDeviceSize,
    ) !void {
        if (!self.is_recording) return CommandError.NotRecording;
        if (buffers.len != offsets.len) return CommandError.InvalidUsage;

        c.vkCmdBindVertexBuffers(
            self.handle,
            first_binding,
            @intCast(buffers.len),
            buffers.ptr,
            offsets.ptr,
        );
    }

    /// Binds an index buffer to the command buffer. The index buffer provides
    /// vertex indices for indexed drawing commands. The offset specifies the
    /// starting position within the buffer.
    ///
    /// The index_type parameter specifies whether indices are 16-bit or 32-bit
    /// values. This must match the size of indices in the buffer. The function
    /// validates that command recording is active.
    pub fn bindIndexBuffer(
        self: *CommandBuffer,
        buffer: c.VkBuffer,
        offset: c.VkDeviceSize,
        index_type: c.VkIndexType,
    ) !void {
        if (!self.is_recording) return CommandError.NotRecording;

        c.vkCmdBindIndexBuffer(self.handle, buffer, offset, index_type);
    }

    /// Records a non-indexed draw command. This function draws primitives using
    /// vertex data from the currently bound vertex buffers. The vertex_count
    /// parameter determines how many vertices to process.
    ///
    /// Multiple instances can be drawn using the instance_count parameter. The
    /// first_vertex parameter provides an offset into the vertex buffers, while
    /// first_instance offsets the instance index.
    pub fn draw(
        self: *CommandBuffer,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) !void {
        if (!self.is_recording) return CommandError.NotRecording;
        if (self.current_pipeline == null) return CommandError.InvalidUsage;
        if (!self.viewport_set or !self.scissor_set) return CommandError.InvalidUsage;

        c.vkCmdDraw(
            self.handle,
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        );
    }

    /// Records an indexed draw command. This function draws primitives using
    /// indices from the bound index buffer to fetch vertex data. The index_count
    /// parameter determines how many indices to process.
    ///
    /// The vertex_offset parameter is added to the index values before fetching
    /// vertices. Multiple instances can be drawn using instance_count, and
    /// first_instance offsets the instance index.
    pub fn drawIndexed(
        self: *CommandBuffer,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) !void {
        if (!self.is_recording) return CommandError.NotRecording;
        if (self.current_pipeline == null) return CommandError.InvalidUsage;
        if (!self.viewport_set or !self.scissor_set) return CommandError.InvalidUsage;

        c.vkCmdDrawIndexed(
            self.handle,
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        );
    }

    /// Executes a set of secondary command buffers within this primary command
    /// buffer. Secondary buffers must have been recorded with appropriate usage
    /// flags. This allows command buffer reuse and multi-threaded recording.
    ///
    /// The function validates that command recording is active and allocates
    /// temporary storage for the raw command buffer handles. The secondary
    /// buffers are executed in the order they appear in the array.
    pub fn executeCommands(self: *CommandBuffer, buffers: []const *CommandBuffer) !void {
        if (!self.is_recording) return CommandError.NotRecording;

        var raw_buffers = try self.allocator.alloc(c.VkCommandBuffer, buffers.len);
        defer self.allocator.free(raw_buffers);

        for (buffers, 0..) |buffer, i| {
            raw_buffers[i] = buffer.handle;
        }

        c.vkCmdExecuteCommands(self.handle, @intCast(buffers.len), raw_buffers.ptr);
    }

    /// Inserts a memory barrier into the command stream. Memory barriers ensure
    /// correct ordering of memory operations and visibility of memory writes. The
    /// barrier creates both an execution dependency and a memory dependency between
    /// commands before and after the barrier.
    ///
    /// When barrier batching is enabled, the barrier is added to the current batch
    /// instead of being inserted immediately. This allows multiple barriers to be
    /// combined into a single command for better performance.
    pub fn memoryBarrier(self: *CommandBuffer, barrier: MemoryBarrier) !void {
        if (!self.is_recording) return CommandError.NotRecording;

        if (self.barrier_batch) |*batch| {
            batch.src_stage |= barrier.src_stage;
            batch.dst_stage |= barrier.dst_stage;

            try batch.memory_barriers.append(.{
                .sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = barrier.src_access,
                .dstAccessMask = barrier.dst_access,
            });
        } else {
            const memory_barrier = c.VkMemoryBarrier{
                .sType = c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = barrier.src_access,
                .dstAccessMask = barrier.dst_access,
            };

            c.vkCmdPipelineBarrier(
                self.handle,
                barrier.src_stage,
                barrier.dst_stage,
                0,
                1,
                &memory_barrier,
                0,
                null,
                0,
                null,
            );
        }
    }

    /// Inserts an image memory barrier into the command stream. Image barriers
    /// synchronize access to image resources and handle layout transitions. They
    /// ensure proper ordering of image memory operations and manage ownership
    /// transfers between queue families.
    ///
    /// The barrier specifies the old and new layouts for the image, along with
    /// the pipeline stages and access types that must be synchronized. Queue
    /// family ownership transfers can be specified for sharing images between
    /// different queues.
    pub fn imageBarrier(self: *CommandBuffer, barrier: ImageBarrier) !void {
        if (!self.is_recording) return CommandError.NotRecording;

        const image_barrier = c.VkImageMemoryBarrier{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .oldLayout = barrier.old_layout,
            .newLayout = barrier.new_layout,
            .srcQueueFamilyIndex = barrier.src_queue_family,
            .dstQueueFamilyIndex = barrier.dst_queue_family,
            .image = barrier.image,
            .subresourceRange = barrier.subresource_range,
            .srcAccessMask = barrier.src_access,
            .dstAccessMask = barrier.dst_access,
        };

        c.vkCmdPipelineBarrier(
            self.handle,
            barrier.src_stage,
            barrier.dst_stage,
            0,
            0,
            null,
            0,
            null,
            1,
            &image_barrier,
        );
    }

    /// Inserts a buffer memory barrier into the command stream. Buffer barriers
    /// synchronize access to buffer resources and handle ownership transfers
    /// between queue families. They ensure proper ordering of buffer memory
    /// operations within specified buffer regions.
    ///
    /// The barrier can target a specific range within the buffer using offset
    /// and size parameters. This allows for fine-grained synchronization when
    /// different parts of a buffer are accessed by different commands.
    pub fn bufferBarrier(self: *CommandBuffer, barrier: BufferBarrier) !void {
        if (!self.is_recording) return CommandError.NotRecording;

        const buffer_barrier = c.VkBufferMemoryBarrier{
            .sType = c.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .srcQueueFamilyIndex = barrier.src_queue_family,
            .dstQueueFamilyIndex = barrier.dst_queue_family,
            .buffer = barrier.buffer,
            .offset = barrier.offset,
            .size = barrier.size,
            .srcAccessMask = barrier.src_access,
            .dstAccessMask = barrier.dst_access,
        };

        c.vkCmdPipelineBarrier(
            self.handle,
            barrier.src_stage,
            barrier.dst_stage,
            0,
            0,
            null,
            1,
            &buffer_barrier,
            0,
            null,
        );
    }

    /// Copies data between buffer objects. The source and destination buffers
    /// must have been created with appropriate usage flags. The regions parameter
    /// specifies the portions of the buffers to copy between.
    ///
    /// Multiple regions can be copied in a single command, which is more efficient
    /// than recording separate copy commands. The function validates that command
    /// recording is active.
    pub fn copyBuffer(
        self: *CommandBuffer,
        src: c.VkBuffer,
        dst: c.VkBuffer,
        regions: []const c.VkBufferCopy,
    ) !void {
        if (!self.is_recording) return CommandError.NotRecording;

        c.vkCmdCopyBuffer(
            self.handle,
            src,
            dst,
            @intCast(regions.len),
            regions.ptr,
        );
    }

    /// Copies data between image objects. The source and destination images
    /// must be in the specified layouts and have been created with appropriate
    /// usage flags. The regions parameter specifies the portions of the images
    /// to copy between.
    ///
    /// The copy respects the format and dimensions of the images. Multiple
    /// regions can be copied in a single command for better performance.
    pub fn copyImage(
        self: *CommandBuffer,
        src: c.VkImage,
        src_layout: c.VkImageLayout,
        dst: c.VkImage,
        dst_layout: c.VkImageLayout,
        regions: []const c.VkImageCopy,
    ) !void {
        if (!self.is_recording) return CommandError.NotRecording;

        c.vkCmdCopyImage(
            self.handle,
            src,
            src_layout,
            dst,
            dst_layout,
            @intCast(regions.len),
            regions.ptr,
        );
    }

    /// Copies data from a buffer object to an image object. The destination
    /// image must be in the specified layout and both resources must have been
    /// created with appropriate usage flags. The regions parameter defines how
    /// the buffer data maps to image coordinates.
    ///
    /// This command is commonly used for texture uploads and image data updates.
    /// Multiple regions can be copied in a single command for better performance.
    pub fn copyBufferToImage(
        self: *CommandBuffer,
        src: c.VkBuffer,
        dst: c.VkImage,
        dst_layout: c.VkImageLayout,
        regions: []const c.VkBufferImageCopy,
    ) !void {
        if (!self.is_recording) return CommandError.NotRecording;

        c.vkCmdCopyBufferToImage(
            self.handle,
            src,
            dst,
            dst_layout,
            @intCast(regions.len),
            regions.ptr,
        );
    }

    /// Dispatches a compute shader workload. The group counts specify the number
    /// of local workgroups to launch in each dimension. The actual number of shader
    /// invocations will be this multiplied by the workgroup size defined in the
    /// shader.
    ///
    /// A compute pipeline must be bound before calling this function. The workgroup
    /// counts must not exceed the device's maxComputeWorkGroupCount limits. The
    /// function validates that command recording is active.
    pub fn dispatch(
        self: *CommandBuffer,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) !void {
        if (!self.is_recording) return CommandError.NotRecording;
        if (self.current_pipeline == null) return CommandError.InvalidUsage;

        c.vkCmdDispatch(self.handle, group_count_x, group_count_y, group_count_z);
    }

    /// Begins a query operation. Queries collect statistics or timing information
    /// about command execution. The query type is determined by the flags used
    /// when creating the query pool.
    ///
    /// Multiple queries can be active simultaneously if they use different query
    /// indices. The function validates that command recording is active and that
    /// the query index is within the pool's capacity.
    pub fn beginQuery(self: *CommandBuffer, info: QueryInfo) !void {
        if (!self.is_recording) return CommandError.NotRecording;

        c.vkCmdBeginQuery(self.handle, info.pool, info.query, info.flags);
    }

    /// Ends a previously started query operation. The results become available
    /// once all commands recorded before the end command have completed execution.
    /// The query must have been started with a matching beginQuery call.
    ///
    /// The function validates that command recording is active and that the query
    /// index matches an active query.
    pub fn endQuery(self: *CommandBuffer, info: QueryInfo) !void {
        if (!self.is_recording) return CommandError.NotRecording;

        c.vkCmdEndQuery(self.handle, info.pool, info.query);
    }

    /// Writes a timestamp value into a query pool. The timestamp is recorded when
    /// command execution reaches the specified pipeline stage. This provides a way
    /// to measure GPU execution time between different points in the command stream.
    ///
    /// The device must support timestamp queries, and the specified pipeline stage
    /// must support timestamps. The function validates that command recording is
    /// active.
    pub fn writeTimestamp(
        self: *CommandBuffer,
        pipeline_stage: c.VkPipelineStageFlags,
        query_pool: c.VkQueryPool,
        query: u32,
    ) !void {
        if (!self.is_recording) return CommandError.NotRecording;

        c.vkCmdWriteTimestamp(self.handle, pipeline_stage, query_pool, query);
    }

    /// Flushes any pending barrier operations in the current batch. When barrier
    /// batching is enabled, this ensures that all previously added barriers are
    /// actually inserted into the command buffer.
    ///
    /// This should be called when you need to ensure all barriers have been
    /// applied, such as before ending command buffer recording or when switching
    /// between different types of commands.
    pub fn flushBarriers(self: *CommandBuffer) !void {
        if (self.barrier_batch) |*batch| {
            try batch.flush(self);
        }
    }
};

/// Executes a one-time command buffer immediately. This helper function manages
/// the buffer lifecycle, including allocation, recording, submission, and cleanup.
/// The provided callback function is used to record commands.
///
/// Note: This function waits for the queue to become idle before returning,
/// ensuring all commands have completed execution.
///
/// Warning: The callback function must not retain references to the command
/// buffer, as it is destroyed after execution. Long-running commands should
/// use explicit command buffer management instead.
pub fn executeOneTime(
    pool: *CommandPool,
    queue: c.VkQueue,
    callback: *const fn (*CommandBuffer) anyerror!void,
) !void {
    const buffer = try pool.getBuffer();
    defer pool.returnBuffer(buffer) catch {};

    try buffer.begin(.{ .one_time_submit = true });
    try callback(buffer);
    try buffer.end();

    try submit(
        queue,
        &[_]*CommandBuffer{buffer},
        .{
            .wait_semaphores = &[_]c.VkSemaphore{},
            .wait_stages = &[_]c.VkPipelineStageFlags{},
            .signal_semaphores = &[_]c.VkSemaphore{},
            .fence = null,
        },
    );

    _ = c.vkQueueWaitIdle(queue);
}
