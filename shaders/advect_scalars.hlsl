// advect_scalars.hlsl, density & temperature, using the final divergence-free velocity
#include "fluid_common.hlsli"

[numthreads(8, 8, 8)]
void main(uint3 id : SV_DispatchThreadID) {
    if (any(id >= pc.gridResolution)) return;
    float3 velocity  = velocityWrite[id].xyz;
    float3 backtrace = float3(id) - pc.dt * velocity;
    densityWrite[id]     = sample_scalar_trilinear(densityRead, backtrace);
    temperatureWrite[id] = sample_scalar_trilinear(temperatureRead, backtrace);
}