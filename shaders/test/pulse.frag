#version 450

layout(location = 0) in vec2 fragTexCoord;
layout(location = 1) in vec4 fragColor;
layout(location = 2) in float fragTime;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstants {
    mat4 projection;
    float time;
    float pulse_speed;
    float pulse_min;
    float pulse_max;
} push;

void main() {
    
    float pulse = mix(push.pulse_min, push.pulse_max, 
        (sin(fragTime * push.pulse_speed) + 1.0) * 0.5);
    
    
    vec4 color = fragColor;
    color.a *= pulse;
    
    outColor = color;
} 