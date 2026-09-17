#include "graphics_pipeline.h"
#include "compute_pipeline.h"   // full definition of FluidGridResources needed here

void GraphicsPipeline::init(VkDevice device, VkFormat swapChainImageFormat, VkExtent2D extent) {
    // TODO
}

void GraphicsPipeline::cleanup(VkDevice device) {
    // TODO
}

void GraphicsPipeline::record_skybox(VkCommandBuffer cmd, const Cubemap& cubemap, VkFramebuffer framebuffer, VkExtent2D extent) {
    // TODO
}

void GraphicsPipeline::record_volume(VkCommandBuffer cmd, const FluidGridResources& grid, const CameraPushConstants& camera) {
    // TODO: vkCmdPushConstants(, &camera, ), bind grid's sampled images, draw fullscreen triangle
}

void GraphicsPipeline::init_cubemap(VkDevice device, Cubemap& cubemap) {
    // Called, but stub for now
    // TODO later: pick an image-loading library (stb_image probably), load each of the 6 faces, create a VK_IMAGE_VIEW_TYPE_CUBE
    // image + view + sampler, populate cubemap's fields, set skyboxEnabled = 1
}