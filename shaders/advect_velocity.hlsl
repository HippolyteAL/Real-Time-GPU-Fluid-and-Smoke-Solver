// advect_velocity.hlsl
#include "fluid_common.hlsli"

[numthreads(8, 8, 8)]
void main(uint3 id : SV_DispatchThreadID) {
    if (any(id >= pc.gridResolution)) return;
    float3 velocity  = velocityRead[id].xyz;
    float3 backtrace = float3(id) - pc.dt * velocity;
    velocityWrite[id] = float4(sample_velocity_trilinear(velocityRead, backtrace), 0.0);
}