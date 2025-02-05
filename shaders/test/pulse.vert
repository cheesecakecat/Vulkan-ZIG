#version 450

layout(location = 0) in vec4 inTransform0;
layout(location = 1) in vec4 inTransform1;
layout(location = 2) in vec4 inTransform2;
layout(location = 3) in vec4 inTransform3;
layout(location = 4) in vec4 inColor;
layout(location = 5) in vec4 inTexRect;
layout(location = 6) in uint inLayer;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out vec4 fragColor;
layout(location = 2) out float fragTime;

layout(push_constant) uniform PushConstants {
    mat4 projection;
    float time;
    float pulse_speed;
    float pulse_min;
    float pulse_max;
} push;

void main() {
    vec2 positions[4] = vec2[](
        vec2(-0.5, -0.5),
        vec2(0.5, -0.5),
        vec2(-0.5, 0.5),
        vec2(0.5, 0.5)
    );

    vec2 uvs[4] = vec2[](
        vec2(0.0, 0.0),
        vec2(1.0, 0.0),
        vec2(0.0, 1.0),
        vec2(1.0, 1.0)
    );

    vec2 pos = positions[gl_VertexIndex];
    vec2 uv = uvs[gl_VertexIndex];

    mat4 transform = mat4(
        inTransform0,
        inTransform1,
        inTransform2,
        inTransform3
    );

    gl_Position = push.projection * transform * vec4(pos, 0.0, 1.0);
    fragTexCoord = inTexRect.xy + uv * inTexRect.zw;
    fragColor = inColor;
    fragTime = push.time;
} 