const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const glfw = @import("mach-glfw");

const instance = @import("instance.zig");
const device = @import("device.zig");
const swapchain = @import("swapchain.zig");
const sync = @import("sync.zig");
const commands = @import("commands.zig");
const pipeline = @import("pipeline.zig");

fn makeVersion(major: u32, minor: u32, patch: u32) u32 {
    return c.VK_MAKE_VERSION(major, minor, patch);
}

pub const Context = struct {
    instance: instance.Instance,
    surface: c.VkSurfaceKHR,
    device: device.Device,
    swapchain: swapchain.Swapchain,
    sync_objects: sync.SyncObjects,
    command_pool: commands.CommandPool,
    pipeline: *pipeline.Pipeline,
    window: glfw.Window,
    allocator: std.mem.Allocator,
    current_frame: u32,

    pub const Error = error{
        InstanceCreationFailed,
        SurfaceCreationFailed,
        DeviceCreationFailed,
        SwapchainCreationFailed,
        SyncObjectCreationFailed,
        CommandPoolCreationFailed,
        PipelineCreationFailed,
    };

    pub const Config = struct {
        instance_config: instance.InstanceConfig = .{
            .application_name = "Vulkan Application",
            .engine_name = "No Engine",
            .application_version = makeVersion(0, 1, 0),
            .engine_version = makeVersion(0, 1, 0),
            .api_version = c.VK_API_VERSION_1_3,
            .enable_validation = true,
            .enable_debug_utils = true,
            .enable_surface_extensions = true,
            .enable_portability = true,
        },
        vsync: bool = true,
        max_frames_in_flight: u32 = 2,
    };
};

pub const RenderPassCreateInfo = struct {
    color_format: c.VkFormat,
    depth_format: c.VkFormat,
    sample_count: c.VkSampleCountFlagBits = c.VK_SAMPLE_COUNT_1_BIT,
    load_op: c.VkAttachmentLoadOp = c.VK_ATTACHMENT_LOAD_OP_CLEAR,
    store_op: c.VkAttachmentStoreOp = c.VK_ATTACHMENT_STORE_OP_STORE,
    initial_layout: c.VkImageLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
    final_layout: c.VkImageLayout = c.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
};

pub const FramebufferCreateInfo = struct {
    render_pass: c.VkRenderPass,
    attachments: []const c.VkImageView,
    width: u32,
    height: u32,
    layers: u32 = 1,
};

pub const ShaderStageCreateInfo = struct {
    stage: c.VkShaderStageFlagBits,
    code: []const u8,
    entry_point: [*:0]const u8 = "main",
};

pub const VertexInputDescription = struct {
    binding_descriptions: []const c.VkVertexInputBindingDescription,
    attribute_descriptions: []const c.VkVertexInputAttributeDescription,
};

pub const PipelineLayoutCreateInfo = struct {
    push_constant_ranges: []const c.VkPushConstantRange = &[_]c.VkPushConstantRange{},
    descriptor_set_layouts: []const c.VkDescriptorSetLayout = &[_]c.VkDescriptorSetLayout{},
};

pub const GraphicsPipelineCreateInfo = struct {
    vertex_shader: ShaderStageCreateInfo,
    fragment_shader: ShaderStageCreateInfo,
    vertex_input: VertexInputDescription,
    topology: c.VkPrimitiveTopology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    polygon_mode: c.VkPolygonMode = c.VK_POLYGON_MODE_FILL,
    cull_mode: c.VkCullModeFlags = c.VK_CULL_MODE_BACK_BIT,
    front_face: c.VkFrontFace = c.VK_FRONT_FACE_CLOCKWISE,
    line_width: f32 = 1.0,
    blend_enable: bool = true,
    depth_test_enable: bool = false,
    depth_write_enable: bool = false,
    layout_info: PipelineLayoutCreateInfo = .{},
};

pub const BufferCreateInfo = struct {
    size: usize,
    usage: c.VkBufferUsageFlags,
    memory_properties: c.VkMemoryPropertyFlags = c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    sharing_mode: c.VkSharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
};

pub const ImageCreateInfo = struct {
    width: u32,
    height: u32,
    format: c.VkFormat,
    usage: c.VkImageUsageFlags,
    memory_properties: c.VkMemoryPropertyFlags = c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    tiling: c.VkImageTiling = c.VK_IMAGE_TILING_OPTIMAL,
    sample_count: c.VkSampleCountFlagBits = c.VK_SAMPLE_COUNT_1_BIT,
    mip_levels: u32 = 1,
    array_layers: u32 = 1,
};
