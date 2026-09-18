// divergence.hlsl
#include "fluid_common.hlsli"

[numthreads(8, 8, 8)]
void main(uint3 id : SV_DispatchThreadID) {
    if (any(id >= pc.gridResolution)) return;
    float3 vL = velocityWrite[clamp_to_grid(int3(id)+int3(-1,0,0))].xyz;
    float3 vR = velocityWrite[clamp_to_grid(int3(id)+int3( 1,0,0))].xyz;
    float3 vD = velocityWrite[clamp_to_grid(int3(id)+int3(0,-1,0))].xyz;
    float3 vU = velocityWrite[clamp_to_grid(int3(id)+int3(0, 1,0))].xyz;
    float3 vB = velocityWrite[clamp_to_grid(int3(id)+int3(0,0,-1))].xyz;
    float3 vF = velocityWrite[clamp_to_grid(int3(id)+int3(0,0, 1))].xyz;
    divergence[id] = 0.5 * ((vR.x - vL.x) + (vU.y - vD.y) + (vF.z - vB.z));
}