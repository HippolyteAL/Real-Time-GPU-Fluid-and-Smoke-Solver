// raymarch.frag.hlsl, placeholder unit-cube domain bounds. TODO: parameterize appropriately
[[vk::binding(1,0)]] [[vk::combinedImageSampler]] Texture3D    densityTexture;
[[vk::binding(1,0)]] [[vk::combinedImageSampler]] SamplerState densitySampler;
[[vk::binding(2,0)]] [[vk::combinedImageSampler]] Texture3D    temperatureTexture;
[[vk::binding(2,0)]] [[vk::combinedImageSampler]] SamplerState temperatureSampler;

struct CameraPushConstants { float4x4 invViewProj; float3 rayOrigin; float _pad; };
[[vk::push_constant]] CameraPushConstants camera;

struct VSOutput { float4 position : SV_Position; float3 viewDir : TEXCOORD0; };

static const int    STEP_COUNT = 64;
static const float  ABSORPTION = 4.0;
static const float3 BOX_MIN = float3(-1,-1,-1);
static const float3 BOX_MAX = float3( 1, 1, 1);

bool intersect_box(float3 origin, float3 dir, out float tNear, out float tFar) {
    float3 invDir = 1.0 / dir;
    float3 t0 = (BOX_MIN - origin) * invDir, t1 = (BOX_MAX - origin) * invDir;
    float3 lo = min(t0, t1), hi = max(t0, t1);
    tNear = max(max(lo.x, lo.y), lo.z);
    tFar  = min(min(hi.x, hi.y), hi.z);
    return tNear < tFar && tFar > 0.0;
}

float4 main(VSOutput input) : SV_Target {
    float3 origin = camera.rayOrigin;
    float3 dir    = normalize(input.viewDir);
    float tNear, tFar;
    if (!intersect_box(origin, dir, tNear, tFar)) discard;
    tNear = max(tNear, 0.0);

    float stepSize = (tFar - tNear) / STEP_COUNT;
    float3 accumColor = float3(0,0,0);
    float  accumAlpha = 0.0;

    for (int i = 0; i < STEP_COUNT; ++i) {
        float3 worldPos = origin + dir * (tNear + stepSize * (i + 0.5));
        float3 uvw = (worldPos - BOX_MIN) / (BOX_MAX - BOX_MIN);

        float density     = densityTexture.SampleLevel(densitySampler, uvw, 0).r;
        float temperature = temperatureTexture.SampleLevel(temperatureSampler, uvw, 0).r;
        float sampleAlpha = saturate(density * ABSORPTION * stepSize);
        float3 blackbody  = saturate(float3(temperature, temperature * 0.5, temperature * 0.1)); // crude placeholder ramp

        accumColor += (1.0 - accumAlpha) * sampleAlpha * blackbody;
        accumAlpha += (1.0 - accumAlpha) * sampleAlpha;
        if (accumAlpha > 0.995) break;
    }
    return float4(accumColor, accumAlpha);
}