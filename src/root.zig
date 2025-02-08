const mod_logger = @import("engine/core/debug/log/logger.zig");

// LOGGER EXPORTS
pub const logger = struct {
    pub const Logger = mod_logger.Logger;
};
