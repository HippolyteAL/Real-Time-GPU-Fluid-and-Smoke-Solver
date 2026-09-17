#ifndef GRAPHICS_PIPELINE_H
#define GRAPHICS_PIPELINE_H

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

struct FluidGridResources;

struct CameraPushConstants {
    glm::mat4   invViewProj;
    glm::vec3   rayOrigin;
    float       _pad;   // alignment
};

// Everything is prettier with a skybox
struct Cubemap {
    VkImage         cubemapImage;
    VkDeviceMemory  cubemapMemory;
    VkImageView     cubemapImageView;
    VkSampler       cubemapSampler;
    uint32_t        skyboxEnabled = 0u;
};

class GraphicsPipeline {
public:
    void init(VkDevice device, VkFormat swapChainImageFormat, VkExtent2D extent);
    void cleanup(VkDevice device);
    void record_skybox(VkCommandBuffer cmd, const Cubemap& cubemap, VkFramebuffer framebuffer, VkExtent2D extent);
    void record_volume(VkCommandBuffer cmd, const FluidGridResources& grid, const CameraPushConstants& camera);

private:
    VkRenderPass                    renderPass;
    VkPipeline                      skyboxPipeline;
    VkPipelineLayout                skyboxLayout;
    VkPipeline                      volumePipeline;
    VkPipelineLayout                volumeLayout;
    VkDescriptorSetLayout           descriptorSetLayout;
    VkDescriptorPool                descriptorPool;
    std::vector<VkDescriptorSet>    descriptorSets;
    VkSampler                       volumeSampler;

    Cubemap cubemap;
    // Cubemap: right, left, top, bottom, front, back
    void init_cubemap(std::array<std::string, 6> faces);
};

#endif // GRAPHICS_PIPELINE_H