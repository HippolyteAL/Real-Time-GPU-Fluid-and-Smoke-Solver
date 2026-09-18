// skybox.frag.hlsl
[[vk::binding(0,0)]] [[vk::combinedImageSampler]] TextureCube  cubemapTexture;
[[vk::binding(0,0)]] [[vk::combinedImageSampler]] SamplerState cubemapSampler;

struct VSOutput { float4 position : SV_Position; float3 viewDir : TEXCOORD0; };

float4 main(VSOutput input) : SV_Target {
    return float4(cubemapTexture.Sample(cubemapSampler, input.viewDir).rgb, 1.0);
}