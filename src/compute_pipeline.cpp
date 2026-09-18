#include "compute_pipeline.h"

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>

/* Implementation detail helpers. None of these need anything beyond what's passed in, 
so there's no reason to give them access to ComputePipeline's private state. */
namespace {

struct ComputePushConstants {
    float    dt;
    uint32_t gridResolution;
};

std::string get_executable_dir() {
    #ifdef _WIN32
        char path[MAX_PATH];
        GetModuleFileNameA(nullptr, path, MAX_PATH);
        std::string full(path);
        size_t pos = full.find_last_of("\\/");
        return (pos == std::string::npos) ? "" : full.substr(0, pos + 1);
    #else
        #error "get_executable_dir: only Win32 implemented"
    #endif
}

std::vector<char> read_file(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("failed to open shader file: " + path);
    size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), size);
    return buffer;
}

VkShaderModule create_shader_module(VkDevice device, const std::string& relativePath) {
    std::vector<char> code = read_file(get_executable_dir() + relativePath);
    VkShaderModuleCreateInfo createInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    createInfo.codeSize = code.size();
    createInfo.pCode    = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule module;
    if (vkCreateShaderModule(device, &createInfo, nullptr, &module) != VK_SUCCESS)
        throw std::runtime_error("failed to create shader module: " + relativePath);
    return module;
}

VkPipeline create_compute_pipeline(VkDevice device, const std::string& shaderPath, VkPipelineLayout layout) {
    VkShaderModule module = create_shader_module(device, shaderPath);

    VkPipelineShaderStageCreateInfo stageInfo{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = module;
    stageInfo.pName  = "main";

    VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pipelineInfo.stage  = stageInfo;
    pipelineInfo.layout = layout;

    VkPipeline pipeline;
    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    vkDestroyShaderModule(device, module, nullptr);   // safe once the pipeline itself is built

    if (result != VK_SUCCESS) throw std::runtime_error("failed to create compute pipeline: " + shaderPath);
    return pipeline;
}

uint32_t workgroup_count(uint32_t gridResolution, uint32_t localSize = 8) {
    return (gridResolution + localSize - 1) / localSize;
}

void dispatch_pass(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout, VkDescriptorSet set, uint32_t gridResolution, float dt)
{
    ComputePushConstants pc{ dt, gridResolution };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    uint32_t groups = workgroup_count(gridResolution);
    vkCmdDispatch(cmd, groups, groups, groups);
}

} // namespace

void ComputePipeline::init(VkDevice device, uint32_t computeQueueFamily) {
    /* computeQueueFamily currently unused here since command pool ownership lives in FrameResources, not ComputePipeline. 
    Kept as a parameter in case a future validation step (e.g. confirming pipeline compatibility with the family) needs it.*/
    (void)computeQueueFamily;

    // Descriptor set layout 
    // 8 storage image bindings: {velocity, density, pressure, temperature} x {slot0, slot1} + divergence. Every pass shares this one layout; each shader only reads the bindings it needs.
    // TODO: revisit once real shaders exist (Placeholder binding scheme)
    std::array<VkDescriptorSetLayoutBinding, 9> bindings{};
    for (uint32_t i = 0; i < bindings.size(); ++i) {
        bindings[i].binding         = i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings    = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute descriptor set layout");

    // Pipeline layout, shared across all passes
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset     = 0;
    pushRange.size       = sizeof(ComputePushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipelineLayoutInfo.setLayoutCount         = 1;
    pipelineLayoutInfo.pSetLayouts            = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges    = &pushRange;
    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &layout) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute pipeline layout");

    // Descriptor pool & sets
    /* 4 sets = 2 (field parity) x 2 (pressure parity): index = fieldPingPong*2 + pressurePingPong. Field parity (velocity/density/temperature) flips once per record(); 
    pressure parity flips once per Jacobi iteration and resets each frame. Four instances in one shared layout. */
    constexpr uint32_t kSetCount = 4;

    VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 9 * kSetCount };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = &poolSize;
    poolInfo.maxSets       = kSetCount;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute descriptor pool");

    std::vector<VkDescriptorSetLayout> setLayouts(kSetCount, descriptorSetLayout);
    VkDescriptorSetAllocateInfo dsAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsAlloc.descriptorPool     = descriptorPool;
    dsAlloc.descriptorSetCount = kSetCount;
    dsAlloc.pSetLayouts        = setLayouts.data();
    descriptorSets.resize(kSetCount);
    if (vkAllocateDescriptorSets(device, &dsAlloc, descriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate compute descriptor sets");
    // TODO: vkUpdateDescriptorSets x4 — needs FluidGridResources' actual VkImageView handles, which don't exist until grid allocation is implemented.

    // Pipelines, one per pass, all sharing "layout"
    buoyancyPipeline        = create_compute_pipeline(device, "shaders/buoyancy.comp.spv", layout);
    advectVelocityPipeline  = create_compute_pipeline(device, "shaders/advect_velocity.comp.spv", layout);
    divergencePipeline      = create_compute_pipeline(device, "shaders/divergence.comp.spv", layout);
    jacobiPipeline          = create_compute_pipeline(device, "shaders/jacobi.comp.spv", layout);
    projectionPipeline      = create_compute_pipeline(device, "shaders/projection.comp.spv", layout);
    advectScalarsPipeline   = create_compute_pipeline(device, "shaders/advect_scalars.comp.spv", layout);
    boundaryPipeline        = create_compute_pipeline(device, "shaders/boundary.comp.spv", layout);
}

void ComputePipeline::cleanup(VkDevice device) {
    vkDestroyPipeline(device, buoyancyPipeline, nullptr);
    vkDestroyPipeline(device, advectVelocityPipeline, nullptr);
    vkDestroyPipeline(device, divergencePipeline, nullptr);
    vkDestroyPipeline(device, jacobiPipeline, nullptr);
    vkDestroyPipeline(device, projectionPipeline, nullptr);
    vkDestroyPipeline(device, advectScalarsPipeline, nullptr);
    vkDestroyPipeline(device, boundaryPipeline, nullptr);

    vkDestroyPipelineLayout(device, layout, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);            // also frees its descriptor sets
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

void ComputePipeline::record(VkCommandBuffer cmd, const FluidGridResources& grid, float dt) {
    record_buoyancy(cmd, grid, dt);
    barrier(cmd);

    record_advect_velocity(cmd, grid, dt);
    barrier(cmd);

    record_boundary(cmd, grid);             // enforce before divergence reads velocity
    barrier(cmd);

    record_divergence(cmd, grid);
    barrier(cmd);

    pressurePingPong = 0;
    for (uint32_t i = 0; i < jacobiIterations; ++i) {
        record_jacobi_iteration(cmd, grid);
        barrier(cmd);
        pressurePingPong ^= 1;
    }

    record_projection(cmd, grid);
    barrier(cmd);

    record_boundary(cmd, grid);             // re-enforce after projection modifies velocity
    barrier(cmd);

    record_advect_scalars(cmd, grid, dt);   // density/temperature, using the now-divergence-free velocity

    fieldPingPong ^= 1;                     // velocity/density/temperature swap once per full timestep (NOTE: ^= 1 should work, might not work)
}

void ComputePipeline::record_buoyancy(VkCommandBuffer cmd, const FluidGridResources& grid, float dt) {
    dispatch_pass(cmd, buoyancyPipeline, layout, descriptorSets[fieldPingPong * 2 + pressurePingPong], grid.gridResolution, dt);
}

void ComputePipeline::record_advect_velocity(VkCommandBuffer cmd, const FluidGridResources& grid, float dt) {
    dispatch_pass(cmd, advectVelocityPipeline, layout, descriptorSets[fieldPingPong * 2 + pressurePingPong], grid.gridResolution, dt);
}

void ComputePipeline::record_divergence(VkCommandBuffer cmd, const FluidGridResources& grid) {
    dispatch_pass(cmd, divergencePipeline, layout, descriptorSets[fieldPingPong * 2 + pressurePingPong], grid.gridResolution, 0.0f);
}

void ComputePipeline::record_jacobi_iteration(VkCommandBuffer cmd, const FluidGridResources& grid) {
    dispatch_pass(cmd, jacobiPipeline, layout, descriptorSets[fieldPingPong * 2 + pressurePingPong], grid.gridResolution, 0.0f);
}

void ComputePipeline::record_projection(VkCommandBuffer cmd, const FluidGridResources& grid) {
    dispatch_pass(cmd, projectionPipeline, layout, descriptorSets[fieldPingPong * 2 + pressurePingPong], grid.gridResolution, 0.0f);
}

void ComputePipeline::record_advect_scalars(VkCommandBuffer cmd, const FluidGridResources& grid, float dt) {
    dispatch_pass(cmd, advectScalarsPipeline, layout, descriptorSets[fieldPingPong * 2 + pressurePingPong], grid.gridResolution, dt);
}

void ComputePipeline::record_boundary(VkCommandBuffer cmd, const FluidGridResources& grid) {
    dispatch_pass(cmd, boundaryPipeline, layout, descriptorSets[fieldPingPong * 2 + pressurePingPong], grid.gridResolution, 0.0f);
}

void ComputePipeline::barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}