#include "fluid_common.hlsli"

/* Fixed continuous emitter near the bottom-center of the grid. It is the simplest possible source term to validate
buoyancy/advection/projection end to end. Input driven injection is the next step once this confirms the chain works.*/ 
static const float3 EMITTER_CENTER          = float3(0.5, 0.15, 0.5);   // strange placement, TODO: rework this
static const float  EMITTER_RADIUS          = 0.06;
static const float  EMIT_DENSITY_RATE       = 2.0;                      // units/sec
static const float  EMIT_TEMPERATURE_RATE   = 4.0;
static const float  DENSITY_MAX             = 1.0;
static const float  TEMPERATURE_MAX         = 4.0;

[numthreads(8, 8, 8)]
void main(uint3 id : SV_DispatchThreadID) {
    if (any(id >= pc.gridResolution)) return;

    float3 normPos  = (float3(id) + 0.5) / float(pc.gridResolution);
    float  dist     = length(normPos - EMITTER_CENTER);
    float  falloff  = saturate(1.0 - dist / EMITTER_RADIUS);
    if (falloff <= 0.0) return;

    float d = densityRead[id];
    float t = temperatureRead[id];
    d = min(d + falloff * EMIT_DENSITY_RATE * pc.dt, DENSITY_MAX);
    t = min(t + falloff * EMIT_TEMPERATURE_RATE * pc.dt, TEMPERATURE_MAX);
    densityRead[id]     = d;
    temperatureRead[id] = t;
}