// boundary.hlsl, pointwise, in place on velocityWrite. called before AND after projection
#include "fluid_common.hlsli"

[numthreads(8, 8, 8)]
void main(uint3 id : SV_DispatchThreadID) {
    if (any(id >= pc.gridResolution)) return;
    float4 v = velocityWrite[id];
    uint last = pc.gridResolution - 1;
    if (id.x == 0 || id.x == last) v.x = 0.0;
    if (id.y == 0 || id.y == last) v.y = 0.0;
    if (id.z == 0 || id.z == last) v.z = 0.0;
    velocityWrite[id] = v;
}