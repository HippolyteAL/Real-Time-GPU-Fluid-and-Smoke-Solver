#include "graphics_pipeline.h"
#include "compute_pipeline.h"   // full definition of FluidGridResources needed here

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

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

void GraphicsPipeline::init_cubemap(VkDevice device) {
    // Stub for now
    // TODO later: pick an image-loading library (stb_image probably), load each of the 6 faces, create a VK_IMAGE_VIEW_TYPE_CUBE
    // image + view + sampler, populate cubemap's fields, set skyboxEnabled = 1
    (void)device;
    // cubemap.skyboxEnabled = 1u;
}