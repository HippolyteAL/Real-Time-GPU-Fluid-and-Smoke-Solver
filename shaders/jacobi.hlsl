// jacobi.hlsl, one iteration. pressureRead/Write roles alternate via descriptor set parity
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
    pressureWrite[id] = (pL + pR + pD + pU + pB + pF - divergence[id]) / 6.0;
}