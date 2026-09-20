#ifndef FRAME_RESOURCES_H
#define FRAME_RESOURCES_H

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

// Command and Sync, shared between compute dispatch and graphics draw
class FrameResources {
public:
    void init(VkDevice device, uint32_t graphicsFamily, uint32_t computeFamily);
    void cleanup(VkDevice device);
    uint32_t begin_frame(VkDevice device, VkSwapchainKHR swapChain);
    VkCommandBuffer acquire_compute_cmd_buffer();
    VkCommandBuffer acquire_graphics_cmd_buffer();
    void submit_compute(VkQueue computeQueue, VkCommandBuffer cmd);
    void submit_graphics(VkQueue graphicsQueue, VkCommandBuffer cmd);
    void present(VkQueue presentQueue, VkSwapchainKHR swapChain, uint32_t imageIndex);
    uint32_t current_frame_index() const { return currentFrame; }
private:
    VkCommandPool                 computePool, graphicsPool;
    std::vector<VkCommandBuffer>  computeCmdBuffers, graphicsCmdBuffers;
    std::vector<VkSemaphore>      imageAvailableSemaphores;
    std::vector<VkSemaphore>      computeFinishedSemaphores, renderFinishedSemaphores;
    std::vector<VkFence>          inFlightFences;
    uint32_t                      currentFrame = 0;
};

#endif // FRAME_RESOURCES_H