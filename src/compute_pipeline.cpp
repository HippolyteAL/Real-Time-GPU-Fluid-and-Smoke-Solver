#include "compute_pipeline.h"

#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

/* Implementation detail helpers. None of these need anything beyond what's passed in, 
so there's no reason to give them access to ComputePipeline's private state. */
namespace {

struct ComputePushConstants {
    float    dt;
    uint32_t gridResolution;
};

uint32_t find_memory_type(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    throw std::runtime_error("failed to find suitable memory type");
}

void create_grid_image(VkDevice device, VkPhysicalDevice physicalDevice, uint32_t resolution, VkFormat format, VkImage& image, VkDeviceMemory& memory, VkImageView& view)
{
    VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imageInfo.imageType     = VK_IMAGE_TYPE_3D;
    imageInfo.format        = format;
    imageInfo.extent        = { resolution, resolution, resolution };
    imageInfo.mipLevels     = 1;
    imageInfo.arrayLayers   = 1;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    // STORAGE for compute imageLoad/imageStore, SAMPLED for the raymarch pass reading density/temperature
    imageInfo.usage         = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
        throw std::runtime_error("failed to create grid image");

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, image, &memReq);
    VkMemoryAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.allocationSize  = memReq.size;
    allocInfo.memoryTypeIndex = find_memory_type(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate grid image memory");
    vkBindImageMemory(device, image, memory, 0);

    VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    viewInfo.image    = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;
    viewInfo.format   = format;
    viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    if (vkCreateImageView(device, &viewInfo, nullptr, &view) != VK_SUCCESS)
        throw std::runtime_error("failed to create grid image view");
}

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

void ComputePipeline::init(VkDevice device, uint32_t computeQueueFamily, uint32_t framesInFlight, float timestampPeriod) {
    /* computeQueueFamily currently unused here since command pool ownership lives in FrameResources, not ComputePipeline. 
    Kept as a parameter in case a future validation step (e.g. confirming pipeline compatibility with the family) needs it.*/
    (void)computeQueueFamily;
    
    // Profiling start
    timestampPeriodNs = timestampPeriod;

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
    constexpr uint32_t SET_COUNT = 4;

    VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 9 * SET_COUNT };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = &poolSize;
    poolInfo.maxSets       = SET_COUNT;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute descriptor pool");

    std::vector<VkDescriptorSetLayout> setLayouts(SET_COUNT, descriptorSetLayout);
    VkDescriptorSetAllocateInfo dsAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsAlloc.descriptorPool     = descriptorPool;
    dsAlloc.descriptorSetCount = SET_COUNT;
    dsAlloc.pSetLayouts        = setLayouts.data();
    descriptorSets.resize(SET_COUNT);
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

    // Timestamp query pools, one per frame-in-flight
    timestampPools.resize(framesInFlight);
    VkQueryPoolCreateInfo queryPoolInfo{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    queryPoolInfo.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolInfo.queryCount = kTimingRegionCount * 2;   // start & end per region

    for (uint32_t i = 0; i < framesInFlight; ++i) {
        if (vkCreateQueryPool(device, &queryPoolInfo, nullptr, &timestampPools[i]) != VK_SUCCESS)
            throw std::runtime_error("failed to create timestamp query pool");
    }

    std::cout   << "[init] compute timing: "                    << kTimingRegionCount 
                << " regions x "                                << framesInFlight 
                << " frame(s) in flight, timestamp period = "   << timestampPeriodNs
                << " ns/tick\n";
}

void ComputePipeline::cleanup(VkDevice device) {
    for (VkQueryPool pool : timestampPools) {
        vkDestroyQueryPool(device, pool, nullptr);
    }
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

void ComputePipeline::record(VkCommandBuffer cmd, const FluidGridResources& grid, float dt, uint32_t frameIndex) {
    VkQueryPool pool = timestampPools[frameIndex];
    vkCmdResetQueryPool(cmd, pool, 0, kTimingRegionCount * 2);

    auto begin_region = [&](TimingRegion region) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, region * 2);
    };
    auto end_region = [&](TimingRegion region) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, region * 2 + 1);
    };

    begin_region(kTimingBuoyancy);
    record_buoyancy(cmd, grid, dt);
    end_region(kTimingBuoyancy);
    barrier(cmd);

    begin_region(kTimingAdvectVelocity);
    record_advect_velocity(cmd, grid, dt);
    end_region(kTimingAdvectVelocity);
    barrier(cmd);

    begin_region(kTimingBoundaryPre);
    record_boundary(cmd, grid);             // enforce before divergence reads velocity
    end_region(kTimingBoundaryPre);
    barrier(cmd);

    begin_region(kTimingDivergence);
    record_divergence(cmd, grid);
    end_region(kTimingDivergence);
    barrier(cmd);

    pressurePingPong = 0;
    begin_region(kTimingJacobi);
    for (uint32_t i = 0; i < jacobiIterations; ++i) {
        record_jacobi_iteration(cmd, grid);
        barrier(cmd);
        pressurePingPong ^= 1;
    }
    end_region(kTimingJacobi);

    begin_region(kTimingProjection);
    record_projection(cmd, grid);
    end_region(kTimingProjection);
    barrier(cmd);

    begin_region(kTimingBoundaryPost);
    record_boundary(cmd, grid);             // re-enforce after projection modifies velocity
    end_region(kTimingBoundaryPost);
    barrier(cmd);

    begin_region(kTimingAdvectScalars);
    record_advect_scalars(cmd, grid, dt);   // density/temperature, using the now-divergence-free velocity
    end_region(kTimingAdvectScalars);

    fieldPingPong ^= 1;                     // velocity/density/temperature swap once per full timestep (NOTE: ^= 1 should work, might not work)
}

void ComputePipeline::read_timestamp_results(VkDevice device, uint32_t frameIndex) {
    std::array<uint64_t, kTimingRegionCount * 2> raw{};
    VkResult result = vkGetQueryPoolResults(
        device, timestampPools[frameIndex], 0, kTimingRegionCount * 2,
        sizeof(raw), raw.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);

    if (result != VK_SUCCESS) {
        return;   // Pool hasn't been written to yet
    }

    for (uint32_t i = 0; i < kTimingRegionCount; ++i) {
        uint64_t start = raw[i * 2];
        uint64_t end   = raw[i * 2 + 1];
        lastTimings[i] = static_cast<float>(end - start) * timestampPeriodNs / 1e6f;   // ns -> ms
    }
}

void ComputePipeline::log_timings() const {
    static const char* kRegionNames[kTimingRegionCount] = {
        "buoyancy", "advect_velocity", "boundary(pre)", "divergence",
        "jacobi", "projection", "boundary(post)", "advect_scalars"
    };
    float total = 0.0f;
    std::cout << "[timing] compute breakdown (ms):\n";
    for (uint32_t i = 0; i < kTimingRegionCount; ++i) {
        std::cout << "  " << kRegionNames[i] << ": " << lastTimings[i] << "\n";
        total += lastTimings[i];
    }
    std::cout << "  total: " << total << "\n";
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

void ComputePipeline::allocate_grid(VkDevice device, VkPhysicalDevice physicalDevice, VkQueue queue, uint32_t queueFamily, FluidGridResources& grid, uint32_t resolution)
{
    grid.gridResolution = resolution;

    // float4 for velocity (xyz used, w unused), scalar float for everything else, has to match fluid_common.hlsli's RWTexture3D<float4> / RWTexture3D<float> declarations exactly.
    for (int i = 0; i < 2; ++i) {
        create_grid_image(device, physicalDevice, resolution, VK_FORMAT_R32G32B32A32_SFLOAT, grid.velocity[i], grid.velocityMemory[i], grid.velocityView[i]);
        create_grid_image(device, physicalDevice, resolution, VK_FORMAT_R32_SFLOAT, grid.density[i], grid.densityMemory[i], grid.densityView[i]);
        create_grid_image(device, physicalDevice, resolution, VK_FORMAT_R32_SFLOAT, grid.pressure[i], grid.pressureMemory[i], grid.pressureView[i]);
        create_grid_image(device, physicalDevice, resolution, VK_FORMAT_R32_SFLOAT, grid.temperature[i], grid.temperatureMemory[i], grid.temperatureView[i]);
    }
    create_grid_image(device, physicalDevice, resolution, VK_FORMAT_R32_SFLOAT, grid.divergence, grid.divergenceMemory, grid.divergenceView);

    // One-time transition, UNDEFINED -> GENERAL, for all 9 images. GENERAL works for both storage image and sampled image access, which both compute and the raymarch pass need.
    // NOTE: reviewing / fine-tuning might be an optimisation opotion later
    VkCommandPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolInfo.flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = queueFamily;
    VkCommandPool transientPool;
    vkCreateCommandPool(device, &poolInfo, nullptr, &transientPool);

    VkCommandBufferAllocateInfo cbAlloc{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbAlloc.commandPool        = transientPool;
    cbAlloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAlloc.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &cbAlloc, &cmd);

    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    std::array<VkImage, 9> allImages = {
        grid.velocity[0], grid.velocity[1], grid.density[0], grid.density[1],
        grid.pressure[0], grid.pressure[1], grid.temperature[0], grid.temperature[1],
        grid.divergence
    };
    std::vector<VkImageMemoryBarrier> barriers;
    for (VkImage img : allImages) {
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = img;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.srcAccessMask       = 0;
        b.dstAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barriers.push_back(b);
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data());
    VkClearColorValue zero{};
    VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    for (VkImage img : allImages) {
        vkCmdClearColorImage(cmd, img, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    }
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers    = &cmd;
    vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);                                 // one time stall at init

    vkDestroyCommandPool(device, transientPool, nullptr);   // also frees cmd

    std::cout << "[init] fluid grid allocated: " << resolution << "^3, 9 images transitioned to GENERAL\n";
}

void ComputePipeline::free_grid(VkDevice device, FluidGridResources& grid) {
    for (int i = 0; i < 2; ++i) {
        vkDestroyImageView(device, grid.velocityView[i], nullptr);
        vkDestroyImage(device, grid.velocity[i], nullptr);
        vkFreeMemory(device, grid.velocityMemory[i], nullptr);
        vkDestroyImageView(device, grid.densityView[i], nullptr);
        vkDestroyImage(device, grid.density[i], nullptr);
        vkFreeMemory(device, grid.densityMemory[i], nullptr);
        vkDestroyImageView(device, grid.pressureView[i], nullptr);
        vkDestroyImage(device, grid.pressure[i], nullptr);
        vkFreeMemory(device, grid.pressureMemory[i], nullptr);
        vkDestroyImageView(device, grid.temperatureView[i], nullptr);
        vkDestroyImage(device, grid.temperature[i], nullptr);
        vkFreeMemory(device, grid.temperatureMemory[i], nullptr);
    }
    vkDestroyImageView(device, grid.divergenceView, nullptr);
    vkDestroyImage(device, grid.divergence, nullptr);
    vkFreeMemory(device, grid.divergenceMemory, nullptr);
}

void ComputePipeline::update_descriptor_sets(VkDevice device, const FluidGridResources& grid) {
    // Views are permanent once allocated, only the read and write slots change.
    for (uint32_t fieldPP = 0; fieldPP < 2; ++fieldPP) {
        for (uint32_t pressPP = 0; pressPP < 2; ++pressPP) {
            VkDescriptorSet set = descriptorSets[fieldPP * 2 + pressPP];

            std::array<VkDescriptorImageInfo, 9> imageInfos{};
            imageInfos[0] = { VK_NULL_HANDLE, grid.velocityView[fieldPP],        VK_IMAGE_LAYOUT_GENERAL };
            imageInfos[1] = { VK_NULL_HANDLE, grid.velocityView[1 - fieldPP],    VK_IMAGE_LAYOUT_GENERAL };
            imageInfos[2] = { VK_NULL_HANDLE, grid.densityView[fieldPP],         VK_IMAGE_LAYOUT_GENERAL };
            imageInfos[3] = { VK_NULL_HANDLE, grid.densityView[1 - fieldPP],     VK_IMAGE_LAYOUT_GENERAL };
            imageInfos[4] = { VK_NULL_HANDLE, grid.pressureView[pressPP],        VK_IMAGE_LAYOUT_GENERAL };
            imageInfos[5] = { VK_NULL_HANDLE, grid.pressureView[1 - pressPP],    VK_IMAGE_LAYOUT_GENERAL };
            imageInfos[6] = { VK_NULL_HANDLE, grid.temperatureView[fieldPP],     VK_IMAGE_LAYOUT_GENERAL };
            imageInfos[7] = { VK_NULL_HANDLE, grid.temperatureView[1 - fieldPP], VK_IMAGE_LAYOUT_GENERAL };
            imageInfos[8] = { VK_NULL_HANDLE, grid.divergenceView,               VK_IMAGE_LAYOUT_GENERAL };

            std::array<VkWriteDescriptorSet, 9> writes{};
            for (uint32_t b = 0; b < 9; ++b) {
                writes[b].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet          = set;
                writes[b].dstBinding      = b;
                writes[b].descriptorCount = 1;
                writes[b].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                writes[b].pImageInfo      = &imageInfos[b];
            }
            vkUpdateDescriptorSets(device, 9, writes.data(), 0, nullptr);
        }
    }
}