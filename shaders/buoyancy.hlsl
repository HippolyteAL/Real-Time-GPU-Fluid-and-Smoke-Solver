// buoyancy.hlsl, pointwise, in place on velocityRead
#include "fluid_common.hlsli"
static const float AMBIENT_TEMPERATURE = 0.0;
static const float BUOYANCY_ALPHA = 1.0;  // density weight (downward)
static const float BUOYANCY_BETA  = 1.0;  // temperature weight (upward)

[numthreads(8, 8, 8)]
void main(uint3 id : SV_DispatchThreadID) {
    if (any(id >= pc.gridResolution)) return;
    float density     = densityRead[id];
    float temperature = temperatureRead[id];
    float4 v = velocityRead[id];
    v.y += pc.dt * (BUOYANCY_BETA * (temperature - AMBIENT_TEMPERATURE) - BUOYANCY_ALPHA * density);
    velocityRead[id] = v;
}