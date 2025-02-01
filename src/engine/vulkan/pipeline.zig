const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});
const logger = @import("../core/logger.zig");
const sprite = @import("sprite.zig");

const PipelineVariant = struct {
    pipeline: c.VkPipeline,
    config: PipelineConfigKey,
};

const MAX_PIPELINE_VARIANTS = 16;

const PipelineConfigKey = struct {
    blend_enable: bool = true,
    cull_mode: c.VkCullModeFlags = c.VK_CULL_MODE_NONE,
    depth_test: bool = false,
    topology: c.VkPrimitiveTopology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    sample_count: c.VkSampleCountFlagBits = c.VK_SAMPLE_COUNT_1_BIT,
    color_write_mask: c.VkColorComponentFlags = c.VK_COLOR_COMPONENT_R_BIT | c.VK_COLOR_COMPONENT_G_BIT | c.VK_COLOR_COMPONENT_B_BIT | c.VK_COLOR_COMPONENT_A_BIT,

    fn hash(self: PipelineConfigKey) u64 {
        var hasher = std.hash.Wyhash.init(0);
        std.hash.autoHash(&hasher, self);
        return hasher.final();
    }
};

pub const Pipeline = struct {
    pipeline: c.VkPipeline,
    pipeline_layout: c.VkPipelineLayout,
    render_pass: c.VkRenderPass,
    pipeline_cache: c.VkPipelineCache,
    device: c.VkDevice,
    allocator: std.mem.Allocator,

    vert_module: c.VkShaderModule,
    frag_module: c.VkShaderModule,

    shader_stages: [2]c.VkPipelineShaderStageCreateInfo,
    vertex_input_info: c.VkPipelineVertexInputStateCreateInfo,
    viewport_state: c.VkPipelineViewportStateCreateInfo,
    dynamic_state: c.VkPipelineDynamicStateCreateInfo,
    specialization_info: c.VkSpecializationInfo,
    spec_entries: [4]c.VkSpecializationMapEntry,
    spec_data: SpecData,

    blend_attachment: c.VkPipelineColorBlendAttachmentState,
    blend_state: c.VkPipelineColorBlendStateCreateInfo,

    variants: std.AutoHashMap(u64, PipelineVariant),
    variant_count: usize = 0,

    cache_data: ?[]const u8,

    const SpecData = struct {
        scale_x: f32,
        scale_y: f32,
        fast_path: u32,
        texture_array_layers: u32,
    };

    fn createColorBlendState(config: PipelineConfigKey) struct {
        attachment: c.VkPipelineColorBlendAttachmentState,
        state: c.VkPipelineColorBlendStateCreateInfo,
    } {
        const attachment = c.VkPipelineColorBlendAttachmentState{
            .colorWriteMask = config.color_write_mask,
            .blendEnable = if (config.blend_enable) c.VK_TRUE else c.VK_FALSE,
            .srcColorBlendFactor = c.VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = c.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp = c.VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = c.VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = c.VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = c.VK_BLEND_OP_ADD,
        };

        return .{
            .attachment = attachment,
            .state = .{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                .logicOpEnable = c.VK_FALSE,
                .logicOp = c.VK_LOGIC_OP_COPY,
                .attachmentCount = 1,
                .pAttachments = &attachment,
                .blendConstants = .{ 0.0, 0.0, 0.0, 0.0 },
                .flags = 0,
                .pNext = null,
            },
        };
    }

    pub fn init(
        device: c.VkDevice,
        swapchain_format: c.VkFormat,
        extent: c.VkExtent2D,
        alloc: std.mem.Allocator,
    ) !*Pipeline {
        const self = try alloc.create(Pipeline);
        errdefer alloc.destroy(self);

        self.variants = std.AutoHashMap(u64, PipelineVariant).init(alloc);

        const cache_data = loadPipelineCache(alloc) catch |err| switch (err) {
            error.NoCacheFile => null,
            else => return err,
        };
        errdefer if (cache_data) |data| alloc.free(data);

        const vert_code = try std.fs.cwd().readFileAlloc(alloc, "shaders/sprite.vert.spv", std.math.maxInt(usize));
        defer alloc.free(vert_code);

        const frag_code = try std.fs.cwd().readFileAlloc(alloc, "shaders/sprite.frag.spv", std.math.maxInt(usize));
        defer alloc.free(frag_code);

        const vert_module = try createShaderModule(device, vert_code);
        errdefer c.vkDestroyShaderModule(device, vert_module, null);

        const frag_module = try createShaderModule(device, frag_code);
        errdefer c.vkDestroyShaderModule(device, frag_module, null);

        const cache_info = c.VkPipelineCacheCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .initialDataSize = if (cache_data) |data| data.len else 0,
            .pInitialData = if (cache_data) |data| data.ptr else null,
        };

        var pipeline_cache: c.VkPipelineCache = undefined;
        if (c.vkCreatePipelineCache(device, &cache_info, null, &pipeline_cache) != c.VK_SUCCESS) {
            return error.PipelineCacheCreationFailed;
        }
        errdefer c.vkDestroyPipelineCache(device, pipeline_cache, null);

        const spec_entries = [_]c.VkSpecializationMapEntry{
            .{
                .constantID = 0,
                .offset = 0,
                .size = @sizeOf(f32),
            },
            .{
                .constantID = 1,
                .offset = @sizeOf(f32),
                .size = @sizeOf(f32),
            },
            .{
                .constantID = 2,
                .offset = @sizeOf(f32) * 2,
                .size = @sizeOf(u32),
            },
            .{
                .constantID = 3,
                .offset = @sizeOf(f32) * 2 + @sizeOf(u32),
                .size = @sizeOf(u32),
            },
        };

        const spec_data = SpecData{
            .scale_x = 1.0,
            .scale_y = 1.0,
            .fast_path = 1,
            .texture_array_layers = 1,
        };

        var specialization_info = c.VkSpecializationInfo{
            .mapEntryCount = spec_entries.len,
            .pMapEntries = &spec_entries,
            .dataSize = @sizeOf(SpecData),
            .pData = &spec_data,
        };

        const shader_stages = [_]c.VkPipelineShaderStageCreateInfo{
            .{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = c.VK_SHADER_STAGE_VERTEX_BIT,
                .module = vert_module,
                .pName = "main",
                .flags = 0,
                .pSpecializationInfo = &specialization_info,
                .pNext = null,
            },
            .{
                .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = c.VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = frag_module,
                .pName = "main",
                .flags = 0,
                .pSpecializationInfo = null,
                .pNext = null,
            },
        };

        const color_attachment = c.VkAttachmentDescription{
            .format = swapchain_format,
            .samples = c.VK_SAMPLE_COUNT_1_BIT,
            .loadOp = c.VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = c.VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = c.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = c.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            .flags = 0,
        };

        const color_attachment_ref = c.VkAttachmentReference{
            .attachment = 0,
            .layout = c.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        const subpass = c.VkSubpassDescription{
            .pipelineBindPoint = c.VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_attachment_ref,
            .pInputAttachments = null,
            .pResolveAttachments = null,
            .pDepthStencilAttachment = null,
            .pPreserveAttachments = null,
            .flags = 0,
        };

        const dependencies = [_]c.VkSubpassDependency{
            .{
                .srcSubpass = c.VK_SUBPASS_EXTERNAL,
                .dstSubpass = 0,
                .srcStageMask = c.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                .dstStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .srcAccessMask = c.VK_ACCESS_MEMORY_READ_BIT,
                .dstAccessMask = c.VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | c.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .dependencyFlags = c.VK_DEPENDENCY_BY_REGION_BIT,
            },
            .{
                .srcSubpass = 0,
                .dstSubpass = c.VK_SUBPASS_EXTERNAL,
                .srcStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                .dstStageMask = c.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                .srcAccessMask = c.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                .dstAccessMask = c.VK_ACCESS_MEMORY_READ_BIT,
                .dependencyFlags = c.VK_DEPENDENCY_BY_REGION_BIT,
            },
        };

        const render_pass_info = c.VkRenderPassCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &color_attachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = dependencies.len,
            .pDependencies = &dependencies,
            .flags = 0,
            .pNext = null,
        };

        var render_pass: c.VkRenderPass = undefined;
        if (c.vkCreateRenderPass(device, &render_pass_info, null, &render_pass) != c.VK_SUCCESS) {
            return error.RenderPassCreationFailed;
        }
        errdefer c.vkDestroyRenderPass(device, render_pass, null);

        const push_constant_range = c.VkPushConstantRange{
            .stageFlags = c.VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = @sizeOf([4][4]f32),
        };

        const pipeline_layout_info = c.VkPipelineLayoutCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant_range,
            .setLayoutCount = 0,
            .pSetLayouts = null,
            .flags = 0,
            .pNext = null,
        };

        var pipeline_layout: c.VkPipelineLayout = undefined;
        if (c.vkCreatePipelineLayout(device, &pipeline_layout_info, null, &pipeline_layout) != c.VK_SUCCESS) {
            return error.PipelineLayoutCreationFailed;
        }
        errdefer c.vkDestroyPipelineLayout(device, pipeline_layout, null);

        const dynamic_states = [_]c.VkDynamicState{
            c.VK_DYNAMIC_STATE_VIEWPORT,
            c.VK_DYNAMIC_STATE_SCISSOR,
            c.VK_DYNAMIC_STATE_LINE_WIDTH,
            c.VK_DYNAMIC_STATE_BLEND_CONSTANTS,
            c.VK_DYNAMIC_STATE_DEPTH_BIAS,
            c.VK_DYNAMIC_STATE_STENCIL_REFERENCE,
        };

        const dynamic_state = c.VkPipelineDynamicStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = dynamic_states.len,
            .pDynamicStates = &dynamic_states,
            .flags = 0,
            .pNext = null,
        };

        const binding_description = sprite.Vertex.getBindingDescription();
        const attribute_descriptions = sprite.Vertex.getAttributeDescriptions();

        const vertex_input_info = c.VkPipelineVertexInputStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &binding_description,
            .vertexAttributeDescriptionCount = attribute_descriptions.len,
            .pVertexAttributeDescriptions = &attribute_descriptions,
            .flags = 0,
            .pNext = null,
        };

        const input_assembly = c.VkPipelineInputAssemblyStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = c.VK_FALSE,
            .flags = 0,
            .pNext = null,
        };

        const viewport = c.VkViewport{
            .x = 0.0,
            .y = 0.0,
            .width = @floatFromInt(extent.width),
            .height = @floatFromInt(extent.height),
            .minDepth = 0.0,
            .maxDepth = 1.0,
        };

        const scissor = c.VkRect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = extent,
        };

        const viewport_state = c.VkPipelineViewportStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
            .flags = 0,
            .pNext = null,
        };

        const rasterizer = c.VkPipelineRasterizationStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = c.VK_FALSE,
            .rasterizerDiscardEnable = c.VK_FALSE,
            .polygonMode = c.VK_POLYGON_MODE_FILL,
            .lineWidth = 1.0,
            .cullMode = c.VK_CULL_MODE_NONE,
            .frontFace = c.VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = c.VK_FALSE,
            .depthBiasConstantFactor = 0.0,
            .depthBiasClamp = 0.0,
            .depthBiasSlopeFactor = 0.0,
            .flags = 0,
            .pNext = null,
        };

        const multisampling = c.VkPipelineMultisampleStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .sampleShadingEnable = c.VK_FALSE,
            .rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT,
            .minSampleShading = 1.0,
            .pSampleMask = null,
            .alphaToCoverageEnable = c.VK_FALSE,
            .alphaToOneEnable = c.VK_FALSE,
            .flags = 0,
            .pNext = null,
        };

        const base_blend = createColorBlendState(.{
            .blend_enable = true,
            .cull_mode = c.VK_CULL_MODE_NONE,
            .depth_test = false,
            .color_write_mask = c.VK_COLOR_COMPONENT_R_BIT | c.VK_COLOR_COMPONENT_G_BIT | c.VK_COLOR_COMPONENT_B_BIT | c.VK_COLOR_COMPONENT_A_BIT,
        });

        self.blend_attachment = base_blend.attachment;
        self.blend_state = base_blend.state;
        self.blend_state.pAttachments = &self.blend_attachment;

        const pipeline_info = c.VkGraphicsPipelineCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .flags = c.VK_PIPELINE_CREATE_ALLOW_DERIVATIVES_BIT,
            .stageCount = shader_stages.len,
            .pStages = &shader_stages,
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_state,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = null,
            .pColorBlendState = &self.blend_state,
            .pDynamicState = &dynamic_state,
            .layout = pipeline_layout,
            .renderPass = render_pass,
            .subpass = 0,
            .basePipelineHandle = null,
            .basePipelineIndex = -1,
            .pNext = null,
        };

        var pipeline: c.VkPipeline = undefined;
        if (c.vkCreateGraphicsPipelines(device, pipeline_cache, 1, &pipeline_info, null, &pipeline) != c.VK_SUCCESS) {
            return error.PipelineCreationFailed;
        }

        self.* = .{
            .pipeline = pipeline,
            .pipeline_layout = pipeline_layout,
            .render_pass = render_pass,
            .pipeline_cache = pipeline_cache,
            .device = device,
            .allocator = alloc,
            .cache_data = cache_data,
            .variants = std.AutoHashMap(u64, PipelineVariant).init(alloc),
            .variant_count = 0,
            .vert_module = vert_module,
            .frag_module = frag_module,
            .shader_stages = shader_stages,
            .vertex_input_info = vertex_input_info,
            .viewport_state = viewport_state,
            .dynamic_state = dynamic_state,
            .specialization_info = specialization_info,
            .spec_entries = spec_entries,
            .spec_data = spec_data,
            .blend_attachment = self.blend_attachment,
            .blend_state = self.blend_state,
        };

        try self.createCommonVariants();

        return self;
    }

    fn createCommonVariants(self: *Pipeline) !void {
        const common_configs = [_]PipelineConfigKey{
            .{
                .blend_enable = true,
                .cull_mode = c.VK_CULL_MODE_NONE,
                .depth_test = false,
            },
            .{
                .blend_enable = false,
                .cull_mode = c.VK_CULL_MODE_BACK_BIT,
                .depth_test = true,
            },
            .{
                .blend_enable = true,
                .cull_mode = c.VK_CULL_MODE_NONE,
                .depth_test = false,
                .topology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
            },
        };

        for (common_configs) |config| {
            if (self.variant_count >= MAX_PIPELINE_VARIANTS) break;
            const hash = config.hash();
            const pipeline = try self.createPipelineVariant(config);
            try self.variants.put(hash, .{
                .pipeline = pipeline,
                .config = config,
            });
            self.variant_count += 1;
        }
    }

    pub fn getPipelineForConfig(self: *Pipeline, config: PipelineConfigKey) !c.VkPipeline {
        const hash = config.hash();
        if (self.variants.get(hash)) |variant| {
            return variant.pipeline;
        }

        if (self.variant_count >= MAX_PIPELINE_VARIANTS) {
            return error.TooManyPipelineVariants;
        }

        const pipeline = try self.createPipelineVariant(config);
        try self.variants.put(hash, .{
            .pipeline = pipeline,
            .config = config,
        });
        self.variant_count += 1;
        return pipeline;
    }

    pub fn deinit(self: *Pipeline) void {
        var variant_iter = self.variants.valueIterator();
        while (variant_iter.next()) |variant| {
            c.vkDestroyPipeline(self.device, variant.pipeline, null);
        }
        self.variants.deinit();

        if (self.savePipelineCache()) |_| {
            logger.debug("vulkan: pp cache saved successfully", .{});
        } else |err| {
            logger.err("vulkan: failed to save pp cache: {}", .{err});
        }

        if (self.cache_data) |data| {
            self.allocator.free(data);
        }

        c.vkDestroyShaderModule(self.device, self.vert_module, null);
        c.vkDestroyShaderModule(self.device, self.frag_module, null);

        c.vkDestroyPipeline(self.device, self.pipeline, null);
        c.vkDestroyPipelineLayout(self.device, self.pipeline_layout, null);
        c.vkDestroyRenderPass(self.device, self.render_pass, null);
        c.vkDestroyPipelineCache(self.device, self.pipeline_cache, null);
        self.allocator.destroy(self);
    }

    fn loadPipelineCache(allocator: std.mem.Allocator) ![]const u8 {
        const cache_file = std.fs.cwd().openFile("pipeline_cache.bin", .{}) catch |err| switch (err) {
            error.FileNotFound => return error.NoCacheFile,
            else => return err,
        };
        defer cache_file.close();

        const file_size = try cache_file.getEndPos();
        const data = try allocator.alloc(u8, file_size);
        errdefer allocator.free(data);

        const bytes_read = try cache_file.readAll(data);
        if (bytes_read != file_size) {
            return error.InvalidCacheFile;
        }

        return data;
    }

    fn savePipelineCache(self: *Pipeline) !void {
        var size: usize = 0;
        _ = c.vkGetPipelineCacheData(self.device, self.pipeline_cache, &size, null);

        const data = try self.allocator.alloc(u8, size);
        defer self.allocator.free(data);

        if (c.vkGetPipelineCacheData(self.device, self.pipeline_cache, &size, data.ptr) != c.VK_SUCCESS) {
            return error.FailedToGetPipelineCacheData;
        }

        const cache_file = try std.fs.cwd().createFile("pipeline_cache.bin", .{});
        defer cache_file.close();

        try cache_file.writeAll(data);
    }

    fn createPipelineVariant(self: *Pipeline, config: PipelineConfigKey) !c.VkPipeline {
        self.blend_attachment = c.VkPipelineColorBlendAttachmentState{
            .colorWriteMask = config.color_write_mask,
            .blendEnable = if (config.blend_enable) c.VK_TRUE else c.VK_FALSE,
            .srcColorBlendFactor = c.VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = c.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp = c.VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = c.VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = c.VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = c.VK_BLEND_OP_ADD,
        };

        self.blend_state.pAttachments = &self.blend_attachment;

        const rasterizer = c.VkPipelineRasterizationStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = c.VK_FALSE,
            .rasterizerDiscardEnable = c.VK_FALSE,
            .polygonMode = c.VK_POLYGON_MODE_FILL,
            .lineWidth = 1.0,
            .cullMode = config.cull_mode,
            .frontFace = c.VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = c.VK_FALSE,
            .depthBiasConstantFactor = 0.0,
            .depthBiasClamp = 0.0,
            .depthBiasSlopeFactor = 0.0,
            .flags = 0,
            .pNext = null,
        };

        const depth_stencil = c.VkPipelineDepthStencilStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = if (config.depth_test) c.VK_TRUE else c.VK_FALSE,
            .depthWriteEnable = if (config.depth_test) c.VK_TRUE else c.VK_FALSE,
            .depthCompareOp = c.VK_COMPARE_OP_LESS,
            .depthBoundsTestEnable = c.VK_FALSE,
            .minDepthBounds = 0.0,
            .maxDepthBounds = 1.0,
            .stencilTestEnable = c.VK_FALSE,
            .front = .{
                .failOp = c.VK_STENCIL_OP_KEEP,
                .passOp = c.VK_STENCIL_OP_KEEP,
                .depthFailOp = c.VK_STENCIL_OP_KEEP,
                .compareOp = c.VK_COMPARE_OP_ALWAYS,
                .compareMask = 0,
                .writeMask = 0,
                .reference = 0,
            },
            .back = .{
                .failOp = c.VK_STENCIL_OP_KEEP,
                .passOp = c.VK_STENCIL_OP_KEEP,
                .depthFailOp = c.VK_STENCIL_OP_KEEP,
                .compareOp = c.VK_COMPARE_OP_ALWAYS,
                .compareMask = 0,
                .writeMask = 0,
                .reference = 0,
            },
            .flags = 0,
            .pNext = null,
        };

        const input_assembly = c.VkPipelineInputAssemblyStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = config.topology,
            .primitiveRestartEnable = c.VK_FALSE,
            .flags = 0,
            .pNext = null,
        };

        const multisampling = c.VkPipelineMultisampleStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .sampleShadingEnable = c.VK_FALSE,
            .rasterizationSamples = config.sample_count,
            .minSampleShading = 1.0,
            .pSampleMask = null,
            .alphaToCoverageEnable = c.VK_FALSE,
            .alphaToOneEnable = c.VK_FALSE,
            .flags = 0,
            .pNext = null,
        };

        const pipeline_info = c.VkGraphicsPipelineCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .flags = c.VK_PIPELINE_CREATE_DERIVATIVE_BIT,
            .stageCount = self.shader_stages.len,
            .pStages = &self.shader_stages,
            .pVertexInputState = &self.vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &self.viewport_state,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depth_stencil,
            .pColorBlendState = &self.blend_state,
            .pDynamicState = &self.dynamic_state,
            .layout = self.pipeline_layout,
            .renderPass = self.render_pass,
            .subpass = 0,
            .basePipelineHandle = self.pipeline,
            .basePipelineIndex = -1,
            .pNext = null,
        };

        var pipeline: c.VkPipeline = undefined;
        if (c.vkCreateGraphicsPipelines(self.device, self.pipeline_cache, 1, &pipeline_info, null, &pipeline) != c.VK_SUCCESS) {
            return error.PipelineVariantCreationFailed;
        }

        return pipeline;
    }
};

fn createShaderModule(device: c.VkDevice, code: []const u8) !c.VkShaderModule {
    const create_info = c.VkShaderModuleCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.len,
        .pCode = @ptrCast(@alignCast(code.ptr)),
        .flags = 0,
        .pNext = null,
    };

    var shader_module: c.VkShaderModule = undefined;
    if (c.vkCreateShaderModule(device, &create_info, null, &shader_module) != c.VK_SUCCESS) {
        return error.ShaderModuleCreationFailed;
    }

    return shader_module;
}
