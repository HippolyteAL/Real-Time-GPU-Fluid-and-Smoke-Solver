// projection.hlsl, pointwise, in place on velocityWrite. Assumes jacobiIterations is even (default 40), so the final solved pressure sits back in the "Read" slot post-loop.

#include "fluid_common.hlsli"

[numthreads(8, 8, 8)]
void main(uint3 id : SV_DispatchThreadID) {
    if (any(id >= pc.gridResolution)) return;
    float pL = pressureRead[clamp_to_grid(int3(id)+int3(-1,0,0))];
    float pR = pressureRead[clamp_to_grid(int3(id)+int3( 1,0,0))];
    float pD = pressureRead[clamp_to_grid(int3(id)+int3(0,-1,0))];
    float pU = pressureRead[clamp_to_grid(int3(id)+int3(0, 1,0))];
    float pB = pressureRead[clamp_to_grid(int3(id)+int3(0,0,-1))];
    float pF = pressureRead[clamp_to_grid(int3(id)+int3(0,0, 1))];
    float3 gradient = 0.5 * float3(pR - pL, pU - pD, pF - pB);
    float4 v = velocityWrite[id];
    v.xyz -= gradient;
    velocityWrite[id] = v;
}