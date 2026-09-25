// fullscreen.vert.hlsl
struct CameraUBO { float4x4 invViewProj; float3 rayOrigin; float _pad; };
[[vk::binding(0, 1)]] ConstantBuffer<CameraUBO> camera;

struct VSOutput { float4 position : SV_Position; float3 worldPos : TEXCOORD0; };

VSOutput main(uint vertexID : SV_VertexID) {
    VSOutput output;
    float2 uv = float2(
        (vertexID << 1) & 2, 
        vertexID & 2
    );
    float4 clipPos = float4(
        uv * 2.0 - 1.0, 
        1.0, 
        1.0
    );
    output.position = clipPos;
    float4 worldPos = mul(camera.invViewProj, clipPos);
    output.worldPos = worldPos.xyz / worldPos.w;
    return output;
}