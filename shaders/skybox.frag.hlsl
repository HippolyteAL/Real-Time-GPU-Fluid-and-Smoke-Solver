// skybox.frag.hlsl
[[vk::binding(0,0)]] [[vk::combinedImageSampler]] TextureCube  cubemapTexture;
[[vk::binding(0,0)]] [[vk::combinedImageSampler]] SamplerState cubemapSampler;

struct CameraUBO { float4x4 invViewProj; float3 rayOrigin; float _pad; };
[[vk::binding(0, 1)]] ConstantBuffer<CameraUBO> camera;

struct VSOutput { float4 position : SV_Position; float3 worldPos : TEXCOORD0; };

float4 main(VSOutput input) : SV_Target {
    float3 viewDir = normalize(input.worldPos - camera.rayOrigin);

    return float4(
        cubemapTexture.Sample(cubemapSampler, viewDir).rgb,
        1.0
    );
}