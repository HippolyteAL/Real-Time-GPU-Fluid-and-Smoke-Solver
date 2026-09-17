#include "frame_resources.h"

void FrameResources::init(VkDevice device, uint32_t graphicsFamily, uint32_t computeFamily) {
    // TODO
}

void FrameResources::cleanup(VkDevice device) {
    // TODO
}

void FrameResources::begin_frame(VkDevice device, VkSwapchainKHR swapChain) {
    // TODO: wait fence, vkAcquireNextImageKHR
}

VkCommandBuffer FrameResources::acquire_compute_cmd_buffer() {
    // TODO
    return VK_NULL_HANDLE;
}

VkCommandBuffer FrameResources::acquire_graphics_cmd_buffer() {
    // TODO
    return VK_NULL_HANDLE;
}

void FrameResources::submit_compute(VkQueue computeQueue, VkCommandBuffer cmd) {
    // TODO
}

void FrameResources::submit_graphics(VkQueue graphicsQueue, VkCommandBuffer cmd) {
    // TODO
}

void FrameResources::present(VkQueue presentQueue, VkSwapchainKHR swapChain, uint32_t imageIndex) {
    // TODO: vkQueuePresentKHR, advance currentFrame
}