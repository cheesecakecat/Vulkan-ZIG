// NOTE: I was using pico ECS before. I don't want to move that file out right from deps directory right now.
// TODO: Move it out from here.
// NOTE: It just a basic implementation. I want something working.
// TODO: Optimize.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Id = u32;
pub const null_id: Id = std.math.maxInt(Id);
pub const DeltaTime = f32;

pub const SystemFn = *const fn (*Context, DeltaTime) anyerror!void;

pub const System = struct {
    name: []const u8,
    func: SystemFn,
    enabled: bool = true,
    required_components: []const Id,
};

pub const Query = struct {
    required_components: []const Id,
    excluded_components: []const Id = &[_]Id{},
};

pub const ComponentStorage = struct {
    type_id: Id,
    allocator: Allocator,
    data: std.AutoHashMap(Id, []u8),
    size: usize,

    pub fn init(allocator: Allocator, type_id: Id, size: usize) ComponentStorage {
        return .{
            .type_id = type_id,
            .allocator = allocator,
            .data = std.AutoHashMap(Id, []u8).init(allocator),
            .size = size,
        };
    }

    pub fn deinit(self: *ComponentStorage) void {
        var iter = self.data.valueIterator();
        while (iter.next()) |value| {
            self.allocator.free(value.*);
        }
        self.data.deinit();
    }

    pub fn put(self: *ComponentStorage, entity: Id, data: []const u8) !void {
        const copy = try self.allocator.alloc(u8, self.size);
        @memcpy(copy, data[0..self.size]);
        try self.data.put(entity, copy);
    }

    pub fn get(self: *ComponentStorage, entity: Id) ?[]const u8 {
        return self.data.get(entity);
    }

    pub fn remove(self: *ComponentStorage, entity: Id) void {
        if (self.data.get(entity)) |data| {
            self.allocator.free(data);
            _ = self.data.remove(entity);
        }
    }
};

pub const Context = struct {
    allocator: Allocator,
    next_entity: Id,
    components: std.AutoHashMap(Id, ComponentStorage),
    systems: std.ArrayList(System),
    queries: std.AutoHashMap(Id, Query),
    next_query_id: Id,

    const Self = @This();

    pub fn init(allocator: Allocator) !*Self {
        const self = try allocator.create(Self);
        self.* = .{
            .allocator = allocator,
            .next_entity = 0,
            .components = std.AutoHashMap(Id, ComponentStorage).init(allocator),
            .systems = std.ArrayList(System).init(allocator),
            .queries = std.AutoHashMap(Id, Query).init(allocator),
            .next_query_id = 0,
        };
        return self;
    }

    pub fn deinit(self: *Self) void {
        var component_iter = self.components.valueIterator();
        while (component_iter.next()) |storage| {
            storage.deinit();
        }
        self.components.deinit();

        for (self.systems.items) |system| {
            self.allocator.free(system.name);
            self.allocator.free(system.required_components);
        }
        self.systems.deinit();

        var query_iter = self.queries.valueIterator();
        while (query_iter.next()) |query| {
            self.allocator.free(query.required_components);
            self.allocator.free(query.excluded_components);
        }
        self.queries.deinit();

        self.allocator.destroy(self);
    }

    pub fn create(self: *Self) Id {
        const id = self.next_entity;
        self.next_entity += 1;
        return id;
    }

    pub fn destroy(self: *Self, entity: Id) void {
        var component_iter = self.components.valueIterator();
        while (component_iter.next()) |storage| {
            storage.remove(entity);
        }
    }

    pub fn registerComponent(self: *Self, comptime T: type) !Id {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(@typeName(T));
        const type_id = @as(Id, @truncate(hasher.final()));

        try self.components.put(type_id, ComponentStorage.init(self.allocator, type_id, @sizeOf(T)));
        return type_id;
    }

    pub fn registerSystem(self: *Self, name: []const u8, func: SystemFn, required_components: []const Id) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        const components_copy = try self.allocator.dupe(Id, required_components);
        try self.systems.append(.{
            .name = name_copy,
            .func = func,
            .required_components = components_copy,
        });
    }

    pub fn createQuery(self: *Self, required: []const Id, excluded: []const Id) !Id {
        const query_id = self.next_query_id;
        self.next_query_id += 1;

        const required_copy = try self.allocator.dupe(Id, required);
        const excluded_copy = try self.allocator.dupe(Id, excluded);

        try self.queries.put(query_id, .{
            .required_components = required_copy,
            .excluded_components = excluded_copy,
        });

        return query_id;
    }

    pub fn queryEntities(self: *Self, query_id: Id, result: *std.ArrayList(Id)) !void {
        const query = self.queries.get(query_id) orelse return error.QueryNotFound;
        result.clearRetainingCapacity();

        var entity: Id = 0;
        while (self.isReady(entity)) : (entity += 1) {
            var matches = true;

            for (query.required_components) |component_id| {
                if (!self.has(entity, component_id)) {
                    matches = false;
                    break;
                }
            }

            for (query.excluded_components) |component_id| {
                if (self.has(entity, component_id)) {
                    matches = false;
                    break;
                }
            }

            if (matches) {
                try result.append(entity);
            }
        }
    }

    pub fn add(self: *Self, entity: Id, component_id: Id, data: anytype) !void {
        const storage = self.components.getPtr(component_id) orelse return error.ComponentNotRegistered;
        const bytes = std.mem.asBytes(&data);
        try storage.put(entity, bytes);
    }

    pub fn get(self: *Self, entity: Id, component_id: Id) ?*anyopaque {
        const storage = self.components.getPtr(component_id) orelse return null;
        const bytes = storage.get(entity) orelse return null;
        return @ptrCast(@constCast(bytes.ptr));
    }

    pub fn has(self: *Self, entity: Id, component_id: Id) bool {
        const storage = self.components.get(component_id) orelse return false;
        return storage.data.contains(entity);
    }

    pub fn remove(self: *Self, entity: Id, component_id: Id) void {
        if (self.components.getPtr(component_id)) |storage| {
            storage.remove(entity);
        }
    }

    pub fn isReady(self: *Self, entity: Id) bool {
        return entity < self.next_entity;
    }

    pub fn updateSystems(self: *Self, dt: DeltaTime) !void {
        for (self.systems.items) |system| {
            if (!system.enabled) continue;
            try system.func(self, dt);
        }
    }

    pub fn enableSystem(self: *Self, name: []const u8) void {
        for (self.systems.items) |*system| {
            if (std.mem.eql(u8, system.name, name)) {
                system.enabled = true;
                return;
            }
        }
    }

    pub fn disableSystem(self: *Self, name: []const u8) void {
        for (self.systems.items) |*system| {
            if (std.mem.eql(u8, system.name, name)) {
                system.enabled = false;
                return;
            }
        }
    }
};

pub fn getAs(comptime T: type, ctx: *Context, entity: Id, component_id: Id) ?*T {
    const ptr = ctx.get(entity, component_id) orelse return null;
    return @ptrCast(@alignCast(@constCast(ptr)));
}
