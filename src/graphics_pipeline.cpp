#include "graphics_pipeline.h"
#include "compute_pipeline.h"   // full definition of FluidGridResources needed here

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct CubeFacePlacement { int col; int row; };
// Vulkan/OpenGL cube array layer order: +X, -X, +Y, -Y, +Z, -Z
constexpr CubeFacePlacement kFaceLayout[6] = {
    { 2, 1 },  // +X
    { 0, 1 },  // -X
    { 1, 0 },  // +Y
    { 1, 2 },  // -Y
    { 1, 1 },  // +Z
    { 3, 1 },  // -Z
};

void extract_face(const uint8_t* src, int srcWidth, int faceSize, int col, int row, uint8_t* dst) {
    constexpr int channels = 4;
    for (int y = 0; y < faceSize; ++y) {
        const uint8_t* srcRow = src + ((row * faceSize + y) * srcWidth + col * faceSize) * channels;
        uint8_t* dstRow = dst + y * faceSize * channels;
        std::memcpy(dstRow, srcRow, faceSize * channels);
    }
}

uint32_t find_memory_type(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }
    throw std::runtime_error("failed to find suitable memory type");
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

} // namespace

void GraphicsPipeline::init(VkDevice device, VkFormat swapChainImageFormat, VkExtent2D extent) {
    // Render pass: single color attachment, no depth (draw order handles skybox, then volume) 
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format         = swapChainImageFormat;
    colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &colorRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass    = 0;
    dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments    = &colorAttachment;
    renderPassInfo.subpassCount    = 1;
    renderPassInfo.pSubpasses      = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies   = &dependency;
    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
        throw std::runtime_error("failed to create render pass");

    // Shared descriptor set layout
    // binding 0: cubemap (skybox), binding 1: density volume, binding 2: temperature volume
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    bindings[0] = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
    bindings[1] = { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
    bindings[2] = { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };

    VkDescriptorSetLayoutCreateInfo dslInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    dslInfo.pBindings    = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics descriptor set layout");

    // Trilinear sampler for the volume; clamps to border so rays exiting the box read as empty
    VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    samplerInfo.magFilter    = VK_FILTER_LINEAR;
    samplerInfo.minFilter    = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    samplerInfo.borderColor  = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    samplerInfo.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &volumeSampler) != VK_SUCCESS)
        throw std::runtime_error("failed to create volume sampler");

    // Descriptor pool & sets: 2 sets, one per ComputePipeline field parity 
    VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 * 2 };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = &poolSize;
    poolInfo.maxSets       = 2;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics descriptor pool");

    std::vector<VkDescriptorSetLayout> setLayouts(2, descriptorSetLayout);
    VkDescriptorSetAllocateInfo dsAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsAlloc.descriptorPool     = descriptorPool;
    dsAlloc.descriptorSetCount = 2;
    dsAlloc.pSetLayouts        = setLayouts.data();
    descriptorSets.resize(2);
    if (vkAllocateDescriptorSets(device, &dsAlloc, descriptorSets.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate graphics descriptor sets");
    // TODO: vkUpdateDescriptorSets x2 (one per field parity) — needs grid.densityView[i]/ temperatureView[i] and the cubemap's view/sampler (need grid allocation and cubemap)

    // Pipeline layouts 
    VkPushConstantRange pushRange{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(CameraPushConstants) };

    VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plInfo.setLayoutCount         = 1;
    plInfo.pSetLayouts            = &descriptorSetLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges    = &pushRange;
    if (vkCreatePipelineLayout(device, &plInfo, nullptr, &skyboxLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create skybox pipeline layout");
    if (vkCreatePipelineLayout(device, &plInfo, nullptr, &volumeLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create volume pipeline layout");

    //  Shared fixed-function state (both draws are a fullscreen triangle, no vertex input)
    VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkViewport viewport{ 0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height), 0.0f, 1.0f };
    VkRect2D scissor{ {0, 0}, extent };
    VkPipelineViewportStateCreateInfo viewportState{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewportState.viewportCount = 1;
    viewportState.pViewports    = &viewport;
    viewportState.scissorCount  = 1;
    viewportState.pScissors     = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode    = VK_CULL_MODE_NONE;
    rasterizer.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    depthStencil.depthTestEnable  = VK_FALSE;
    depthStencil.depthWriteEnable = VK_FALSE;

    // Shared vertex shader module since both pipelines use the identical fullscreen triangle + view-ray reconstruction trick, so this is loaded once and reused for both.
    VkShaderModule fullscreenVert = create_shader_module(device, "shaders/fullscreen.vert.spv");

    // Skybox pipeline: opaque background (NOTE: temporary?)
    VkPipelineColorBlendAttachmentState skyboxBlend{};
    skyboxBlend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    skyboxBlend.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo skyboxBlendState{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    skyboxBlendState.attachmentCount = 1;
    skyboxBlendState.pAttachments    = &skyboxBlend;

    VkShaderModule skyboxFrag = create_shader_module(device, "shaders/skybox.frag.spv");
    VkPipelineShaderStageCreateInfo skyboxStages[] = {
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,   fullscreenVert, "main", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, skyboxFrag,     "main", nullptr },
    };

    VkGraphicsPipelineCreateInfo skyboxInfo{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    skyboxInfo.stageCount          = 2;
    skyboxInfo.pStages             = skyboxStages;
    skyboxInfo.pVertexInputState   = &vertexInput;
    skyboxInfo.pInputAssemblyState = &inputAssembly;
    skyboxInfo.pViewportState      = &viewportState;
    skyboxInfo.pRasterizationState = &rasterizer;
    skyboxInfo.pMultisampleState   = &multisampling;
    skyboxInfo.pDepthStencilState  = &depthStencil;
    skyboxInfo.pColorBlendState    = &skyboxBlendState;
    skyboxInfo.layout              = skyboxLayout;
    skyboxInfo.renderPass          = renderPass;
    skyboxInfo.subpass             = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &skyboxInfo, nullptr, &skyboxPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create skybox pipeline");
    vkDestroyShaderModule(device, skyboxFrag, nullptr);     // fullscreenVert is NOT destroyed here since the volume pipeline below still needs it.

    // Volume pipeline: alpha blended over the skybox 
    VkPipelineColorBlendAttachmentState volumeBlend{};
    volumeBlend.colorWriteMask      = skyboxBlend.colorWriteMask;
    volumeBlend.blendEnable         = VK_TRUE;
    volumeBlend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    volumeBlend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    volumeBlend.colorBlendOp        = VK_BLEND_OP_ADD;
    volumeBlend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    volumeBlend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    volumeBlend.alphaBlendOp        = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo volumeBlendState{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    volumeBlendState.attachmentCount = 1;
    volumeBlendState.pAttachments    = &volumeBlend;

    VkShaderModule raymarchFrag = create_shader_module(device, "shaders/raymarch.frag.spv");
    VkPipelineShaderStageCreateInfo volumeStages[] = {
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT,   fullscreenVert, "main", nullptr },
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, raymarchFrag,   "main", nullptr },
    };

    VkGraphicsPipelineCreateInfo volumeInfo = skyboxInfo;   // reuse the fixed-function state, swap stages/layout/blend
    volumeInfo.stageCount       = 2;
    volumeInfo.pStages          = volumeStages;
    volumeInfo.pColorBlendState = &volumeBlendState;
    volumeInfo.layout           = volumeLayout;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &volumeInfo, nullptr, &volumePipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create volume pipeline");

    vkDestroyShaderModule(device, raymarchFrag, nullptr);
    vkDestroyShaderModule(device, fullscreenVert, nullptr);
}

void GraphicsPipeline::cleanup(VkDevice device) {
    vkDestroyPipeline(device, skyboxPipeline, nullptr);
    vkDestroyPipeline(device, volumePipeline, nullptr);
    vkDestroyPipelineLayout(device, skyboxLayout, nullptr);
    vkDestroyPipelineLayout(device, volumeLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);
    vkDestroySampler(device, volumeSampler, nullptr);
    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    if (cubemap.cubemapImageView != VK_NULL_HANDLE) vkDestroyImageView(device, cubemap.cubemapImageView, nullptr);
    if (cubemap.cubemapSampler   != VK_NULL_HANDLE) vkDestroySampler(device, cubemap.cubemapSampler, nullptr);
    if (cubemap.cubemapImage     != VK_NULL_HANDLE) vkDestroyImage(device, cubemap.cubemapImage, nullptr);
    if (cubemap.cubemapMemory    != VK_NULL_HANDLE) vkFreeMemory(device, cubemap.cubemapMemory, nullptr);
}

void GraphicsPipeline::record_skybox(VkCommandBuffer cmd, VkFramebuffer framebuffer, VkExtent2D extent) {
    VkClearValue clearColor{ {{0.0f, 0.0f, 0.0f, 1.0f}} };

    VkRenderPassBeginInfo rpInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    rpInfo.renderPass        = renderPass;
    rpInfo.framebuffer       = framebuffer;
    rpInfo.renderArea.extent = extent;
    rpInfo.clearValueCount   = 1;
    rpInfo.pClearValues      = &clearColor;

    vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

    if (cubemap.skyboxEnabled) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipeline);
        // Descriptor binding 0 (cubemap sampler) becomes meaningful once init()'s vkUpdateDescriptorSets TODO is filled in.
        vkCmdDraw(cmd, 3, 1, 0, 0);   // fullscreen triangle; view direction reconstructed in the vertex shader
    }

    /* Render pass deliberately left open: record_volume must be called immediately after this, in the same command buffer, to close it. 
    The two are one draw call pair, not independently callable. */
}

void GraphicsPipeline::record_volume(VkCommandBuffer cmd, const FluidGridResources& grid, const CameraPushConstants& camera, uint32_t currentFieldIndex)
{
    (void)grid;   // binding is resolved through the pre-built descriptor set, not read directly here
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, volumePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, volumeLayout, 0, 1,  &descriptorSets[currentFieldIndex], 0, nullptr);
    vkCmdPushConstants(cmd, volumeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(camera), &camera);
    vkCmdDraw(cmd, 3, 1, 0, 0);   // fullscreen triangle; ray/box intersection done per-pixel in the fragment shader

    vkCmdEndRenderPass(cmd);   // closes the pass record_skybox opened
}

void GraphicsPipeline::init_cubemap(VkDevice device, VkPhysicalDevice physicalDevice, VkQueue queue, uint32_t queueFamily, const std::string& crossImagePath)
{
    std::string fullPath = get_executable_dir() + crossImagePath;

    int srcWidth, srcHeight, srcChannels;
    uint8_t* pixels = stbi_load(fullPath.c_str(), &srcWidth, &srcHeight, &srcChannels, STBI_rgb_alpha);
    if (!pixels) {
       throw std::runtime_error("failed to load cubemap cross image: " + fullPath + " (stb reason: " + stbi_failure_reason() + ")");
    }

    if (srcWidth % 4 != 0 || srcHeight % 3 != 0 || (srcWidth / 4) != (srcHeight / 3)) {
        stbi_image_free(pixels);
        throw std::runtime_error("cubemap image is not a 4x3 horizontal cross with square faces: " + fullPath);
    }
    const int faceSize = srcWidth / 4;

    // sRGB: matches the swapchain's own sRGB surface format from chooseSwapSurfaceFormat.
    constexpr VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;

    VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imageInfo.flags         = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    imageInfo.imageType     = VK_IMAGE_TYPE_2D;
    imageInfo.format        = format;
    imageInfo.extent        = { static_cast<uint32_t>(faceSize), static_cast<uint32_t>(faceSize), 1 };
    imageInfo.mipLevels     = 1;
    imageInfo.arrayLayers   = 6;
    imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage         = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(device, &imageInfo, nullptr, &cubemap.cubemapImage) != VK_SUCCESS) {
        stbi_image_free(pixels);
        throw std::runtime_error("failed to create cubemap image");
    }

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device, cubemap.cubemapImage, &memReq);
    VkMemoryAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.allocationSize  = memReq.size;
    allocInfo.memoryTypeIndex = find_memory_type(physicalDevice, memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &cubemap.cubemapMemory) != VK_SUCCESS) {
        stbi_image_free(pixels);
        throw std::runtime_error("failed to allocate cubemap memory");
    }
    vkBindImageMemory(device, cubemap.cubemapImage, cubemap.cubemapMemory, 0);

    VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    viewInfo.image      = cubemap.cubemapImage;
    viewInfo.viewType   = VK_IMAGE_VIEW_TYPE_CUBE;
    viewInfo.format     = format;
    viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6 };
    if (vkCreateImageView(device, &viewInfo, nullptr, &cubemap.cubemapImageView) != VK_SUCCESS) {
        stbi_image_free(pixels);
        throw std::runtime_error("failed to create cubemap image view");
    }

    VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    samplerInfo.magFilter = samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    if (vkCreateSampler(device, &samplerInfo, nullptr, &cubemap.cubemapSampler) != VK_SUCCESS) {
        stbi_image_free(pixels);
        throw std::runtime_error("failed to create cubemap sampler");
    }

    // Staging buffer: extract the 6 cross arm faces into layer ordered data
    VkDeviceSize faceBytes  = static_cast<VkDeviceSize>(faceSize) * faceSize * 4;
    VkDeviceSize bufferSize = faceBytes * 6;

    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size         = bufferSize;
    bufferInfo.usage        = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode  = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer stagingBuffer;
    vkCreateBuffer(device, &bufferInfo, nullptr, &stagingBuffer);

    VkMemoryRequirements bufMemReq;
    vkGetBufferMemoryRequirements(device, stagingBuffer, &bufMemReq);
    VkMemoryAllocateInfo bufAllocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    bufAllocInfo.allocationSize  = bufMemReq.size;
    bufAllocInfo.memoryTypeIndex = find_memory_type(physicalDevice, bufMemReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory stagingMemory;
    vkAllocateMemory(device, &bufAllocInfo, nullptr, &stagingMemory);
    vkBindBufferMemory(device, stagingBuffer, stagingMemory, 0);

    void* mapped;
    vkMapMemory(device, stagingMemory, 0, bufferSize, 0, &mapped);
    for (int layer = 0; layer < 6; ++layer) {
        extract_face(pixels, srcWidth, faceSize, kFaceLayout[layer].col, kFaceLayout[layer].row, static_cast<uint8_t*>(mapped) + layer * faceBytes);
    }
    vkUnmapMemory(device, stagingMemory);
    stbi_image_free(pixels);

    // Upload: one copy region covering all 6 layers since staging data is packed in layer order
    VkCommandPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolInfo.flags              = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex   = queueFamily;
    VkCommandPool transientPool;
    vkCreateCommandPool(device, &poolInfo, nullptr, &transientPool);

    VkCommandBufferAllocateInfo cbAlloc{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbAlloc.commandPool         = transientPool;
    cbAlloc.level               = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAlloc.commandBufferCount  = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &cbAlloc, &cmd);

    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    VkImageMemoryBarrier toDst{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    toDst.oldLayout             = VK_IMAGE_LAYOUT_UNDEFINED;
    toDst.newLayout             = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toDst.srcQueueFamilyIndex   = VK_QUEUE_FAMILY_IGNORED;
    toDst.dstQueueFamilyIndex   = VK_QUEUE_FAMILY_IGNORED;
    toDst.image                 = cubemap.cubemapImage;
    toDst.subresourceRange      = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6 };
    toDst.dstAccessMask         = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toDst);

    VkBufferImageCopy copyRegion{};
    copyRegion.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 6 };
    copyRegion.imageExtent      = { static_cast<uint32_t>(faceSize), static_cast<uint32_t>(faceSize), 1 };
    vkCmdCopyBufferToImage(cmd, stagingBuffer, cubemap.cubemapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

    VkImageMemoryBarrier toRead = toDst;
    toRead.oldLayout        = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toRead.newLayout        = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toRead.srcAccessMask    = VK_ACCESS_TRANSFER_WRITE_BIT;
    toRead.dstAccessMask    = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &toRead);

    vkEndCommandBuffer(cmd);
    VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount   = 1;
    submit.pCommandBuffers      = &cmd;
    vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkDestroyCommandPool(device, transientPool, nullptr);
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);

    cubemap.skyboxEnabled = 1u;
    std::cout << "[init] cubemap loaded from " << fullPath << " (" << faceSize << "x" << faceSize << " per face)\n";
}

void GraphicsPipeline::update_descriptor_sets(VkDevice device, const FluidGridResources& grid) {
    for (uint32_t i = 0; i < 2; ++i) {
        VkDescriptorImageInfo cubemapInfo{ cubemap.cubemapSampler, cubemap.cubemapImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo densityInfo{ volumeSampler, grid.densityView[i], VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo temperatureInfo{ volumeSampler, grid.temperatureView[i], VK_IMAGE_LAYOUT_GENERAL };

        std::array<VkWriteDescriptorSet, 3> writes{};
        for (auto& w : writes) {
            w.sType             = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet            = descriptorSets[i];
            w.descriptorCount   = 1;
            w.descriptorType    = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        }
        writes[0].dstBinding    = 0; writes[0].pImageInfo = &cubemapInfo;
        writes[1].dstBinding    = 1; writes[1].pImageInfo = &densityInfo;
        writes[2].dstBinding    = 2; writes[2].pImageInfo = &temperatureInfo;

        vkUpdateDescriptorSets(device, 3, writes.data(), 0, nullptr);
    }
}