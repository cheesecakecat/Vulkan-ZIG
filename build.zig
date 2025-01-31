const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const vulkan_sdk = b.option(
        []const u8,
        "vulkan-sdk",
        "Path to Vulkan SDK (defaults to VULKAN_SDK env var)",
    ) orelse blk: {
        if (std.process.getEnvVarOwned(b.allocator, "VULKAN_SDK")) |sdk_path| {
            break :blk sdk_path;
        } else |_| {
            std.debug.print(
                \\Error: VULKAN_SDK environment variable not found!
                \\Please set it to your Vulkan SDK installation path:
                \\
                \\Windows (PowerShell):
                \\    $env:VULKAN_SDK = "C:/VulkanSDK/x.x.x"
                \\
                \\Linux/MacOS:
                \\    export VULKAN_SDK=/path/to/vulkansdk
                \\
                \\Alternatively, you can pass it directly:
                \\    zig build -Dvulkan-sdk="/path/to/vulkansdk"
                \\
            , .{});
            std.process.exit(1);
        }
    };

    const glfw_dep = b.dependency("mach-glfw", .{
        .target = target,
        .optimize = optimize,
    });

    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe_mod.addImport("mach-glfw", glfw_dep.module("mach-glfw"));
    exe_mod.addImport("VK-ZIG_lib", lib_mod);

    const lib = b.addStaticLibrary(.{
        .name = "VK-ZIG",
        .root_module = lib_mod,
    });

    const include_path = switch (target.result.os.tag) {
        .windows => b.pathJoin(&.{ vulkan_sdk, "Include" }),
        .linux, .macos => b.pathJoin(&.{ vulkan_sdk, "include" }),
        else => {
            std.debug.print("Unsupported operating system\n", .{});
            std.process.exit(1);
        },
    };

    const vma_include_path = "deps";

    const lib_path = switch (target.result.os.tag) {
        .windows => b.pathJoin(&.{ vulkan_sdk, "Lib" }),
        .linux => b.pathJoin(&.{ vulkan_sdk, "lib" }),
        .macos => b.pathJoin(&.{ vulkan_sdk, "lib" }),
        else => {
            std.debug.print("Unsupported operating system\n", .{});
            std.process.exit(1);
        },
    };

    lib.addIncludePath(.{ .cwd_relative = include_path });
    lib.addIncludePath(.{ .cwd_relative = vma_include_path });
    lib.addLibraryPath(.{ .cwd_relative = lib_path });
    if (target.result.os.tag == .windows) {
        lib.linkSystemLibrary("vulkan-1");
    } else {
        lib.linkSystemLibrary("vulkan");
    }
    lib.linkLibC();

    b.installArtifact(lib);

    const exe = b.addExecutable(.{
        .name = "VK-ZIG",
        .root_module = exe_mod,
    });

    exe.addIncludePath(.{ .cwd_relative = include_path });
    exe.addIncludePath(.{ .cwd_relative = vma_include_path });
    exe.addLibraryPath(.{ .cwd_relative = lib_path });
    if (target.result.os.tag == .windows) {
        exe.linkSystemLibrary("vulkan-1");
    } else {
        exe.linkSystemLibrary("vulkan");
    }
    exe.linkLibC();

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Compile shaders
    const shader_step = b.step("shaders", "Compile shaders to SPIR-V");
    const shader_dir = "shaders/";
    const shaders = [_][]const u8{ "sprite.vert", "sprite.frag" };

    // Create shader output directory
    const shader_out_dir = "zig-out/shaders";
    std.fs.cwd().makePath(shader_out_dir) catch |err| {
        std.debug.print("Failed to create shader directory: {}\n", .{err});
        return;
    };

    inline for (shaders) |shader| {
        const shader_path = shader_dir ++ shader;
        const spv_path = shader_out_dir ++ "/" ++ shader ++ ".spv";
        const shader_compile = b.addSystemCommand(&.{
            "glslc",
            shader_path,
            "-o",
            spv_path,
        });
        shader_step.dependOn(&shader_compile.step);
        exe.step.dependOn(&shader_compile.step);
    }

    const lib_unit_tests = b.addTest(.{
        .root_module = lib_mod,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const exe_unit_tests = b.addTest(.{
        .root_module = exe_mod,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
}
