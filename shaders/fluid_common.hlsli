// Shared resources, bindings, and helpers for all compute passes. Bindings match ComputePipeline's shared descriptor set layout.

struct PushConstants {
    float dt;
    uint  gridResolution;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWTexture3D<float4> velocityRead;
[[vk::binding(1, 0)]] RWTexture3D<float4> velocityWrite;
[[vk::binding(2, 0)]] RWTexture3D<float>  densityRead;
[[vk::binding(3, 0)]] RWTexture3D<float>  densityWrite;
[[vk::binding(4, 0)]] RWTexture3D<float>  pressureRead;
[[vk::binding(5, 0)]] RWTexture3D<float>  pressureWrite;
[[vk::binding(6, 0)]] RWTexture3D<float>  temperatureRead;
[[vk::binding(7, 0)]] RWTexture3D<float>  temperatureWrite;
[[vk::binding(8, 0)]] RWTexture3D<float>  divergence;

uint3 clamp_to_grid(int3 p) {
    int m = int(pc.gridResolution) - 1;
    return uint3(clamp(p, int3(0, 0, 0), int3(m, m, m)));
}

// Manual trilinear sample, RWTexture3D (storage image) has no hardware filtering so semi-Lagrangian advection interpolates across 8 neighbors by hand.
float sample_scalar_trilinear(RWTexture3D<float> field, float3 pos) {
    float3 f = floor(pos);
    float3 t = pos - f;
    int3   i0 = int3(f);

    float c000 = field[clamp_to_grid(i0 + int3(0,0,0))];
    float c100 = field[clamp_to_grid(i0 + int3(1,0,0))];
    float c010 = field[clamp_to_grid(i0 + int3(0,1,0))];
    float c110 = field[clamp_to_grid(i0 + int3(1,1,0))];
    float c001 = field[clamp_to_grid(i0 + int3(0,0,1))];
    float c101 = field[clamp_to_grid(i0 + int3(1,0,1))];
    float c011 = field[clamp_to_grid(i0 + int3(0,1,1))];
    float c111 = field[clamp_to_grid(i0 + int3(1,1,1))];

    float c00 = lerp(c000, c100, t.x), c10 = lerp(c010, c110, t.x);
    float c01 = lerp(c001, c101, t.x), c11 = lerp(c011, c111, t.x);
    return lerp(lerp(c00, c10, t.y), lerp(c01, c11, t.y), t.z);
}

float3 sample_velocity_trilinear(RWTexture3D<float4> field, float3 pos) {
    float3 f = floor(pos);
    float3 t = pos - f;
    int3   i0 = int3(f);

    float3 c000 = field[clamp_to_grid(i0 + int3(0,0,0))].xyz;
    float3 c100 = field[clamp_to_grid(i0 + int3(1,0,0))].xyz;
    float3 c010 = field[clamp_to_grid(i0 + int3(0,1,0))].xyz;
    float3 c110 = field[clamp_to_grid(i0 + int3(1,1,0))].xyz;
    float3 c001 = field[clamp_to_grid(i0 + int3(0,0,1))].xyz;
    float3 c101 = field[clamp_to_grid(i0 + int3(1,0,1))].xyz;
    float3 c011 = field[clamp_to_grid(i0 + int3(0,1,1))].xyz;
    float3 c111 = field[clamp_to_grid(i0 + int3(1,1,1))].xyz;

    float3 c00 = lerp(c000, c100, t.x), c10 = lerp(c010, c110, t.x);
    float3 c01 = lerp(c001, c101, t.x), c11 = lerp(c011, c111, t.x);
    return lerp(lerp(c00, c10, t.y), lerp(c01, c11, t.y), t.z);
}