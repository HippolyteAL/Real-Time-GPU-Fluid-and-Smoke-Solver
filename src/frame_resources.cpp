#include "frame_resources.h"
#include "vulkan_context.h"     // Constants::MAX_FRAMES_IN_FLIGHT

#include <stdexcept>

void FrameResources::init(VkDevice device, uint32_t graphicsFamily, uint32_t computeFamily) {
    VkCommandPoolCreateInfo graphicsPoolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    graphicsPoolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    graphicsPoolInfo.queueFamilyIndex = graphicsFamily;
    if (vkCreateCommandPool(device, &graphicsPoolInfo, nullptr, &graphicsPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create graphics command pool");

    VkCommandPoolCreateInfo computePoolInfo = graphicsPoolInfo;
    computePoolInfo.queueFamilyIndex = computeFamily;
    if (vkCreateCommandPool(device, &computePoolInfo, nullptr, &computePool) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute command pool");

    graphicsCmdBuffers.resize(Constants::MAX_FRAMES_IN_FLIGHT);
    computeCmdBuffers.resize(Constants::MAX_FRAMES_IN_FLIGHT);

    VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = Constants::MAX_FRAMES_IN_FLIGHT;

    allocInfo.commandPool = graphicsPool;
    if (vkAllocateCommandBuffers(device, &allocInfo, graphicsCmdBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate graphics command buffers");

    allocInfo.commandPool = computePool;
    if (vkAllocateCommandBuffers(device, &allocInfo, computeCmdBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate compute command buffers");

    imageAvailableSemaphores.resize(Constants::MAX_FRAMES_IN_FLIGHT);
    computeFinishedSemaphores.resize(Constants::MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(Constants::MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(Constants::MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkFenceCreateInfo fenceInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;   // start signaled so the first begin_frame() doesn't block forever

    for (uint32_t i = 0; i < Constants::MAX_FRAMES_IN_FLIGHT; ++i) {
        if (vkCreateSemaphore(device, &semInfo, nullptr, &imageAvailableSemaphores[i])  != VK_SUCCESS ||
            vkCreateSemaphore(device, &semInfo, nullptr, &computeFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device, &semInfo, nullptr, &renderFinishedSemaphores[i])  != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i])              != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create sync objects for a frame");
        }
    }
}

void FrameResources::cleanup(VkDevice device) {
    for (size_t i = 0; i < inFlightFences.size(); ++i) {
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroySemaphore(device, computeFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }
    if (graphicsPool != VK_NULL_HANDLE) vkDestroyCommandPool(device, graphicsPool, nullptr);
    if (computePool  != VK_NULL_HANDLE) vkDestroyCommandPool(device, computePool, nullptr);
}

uint32_t FrameResources::begin_frame(VkDevice device, VkSwapchainKHR swapChain) {
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex = 0;
    VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX,
        imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        // TODO: trigger swapchain recreation — surface resized/invalidated
        return imageIndex;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swapchain image");
    }

    vkResetFences(device, 1, &inFlightFences[currentFrame]);
    vkResetCommandBuffer(computeCmdBuffers[currentFrame], 0);
    vkResetCommandBuffer(graphicsCmdBuffers[currentFrame], 0);

    return imageIndex;
}

VkCommandBuffer FrameResources::acquire_compute_cmd_buffer() {
    return computeCmdBuffers[currentFrame];
}

VkCommandBuffer FrameResources::acquire_graphics_cmd_buffer() {
    return graphicsCmdBuffers[currentFrame];
}

void FrameResources::submit_compute(VkQueue computeQueue, VkCommandBuffer cmd) {
    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount   = 1;
    submitInfo.pCommandBuffers      = &cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores    = &computeFinishedSemaphores[currentFrame];

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
        throw std::runtime_error("failed to submit compute command buffer");
}

void FrameResources::submit_graphics(VkQueue graphicsQueue, VkCommandBuffer cmd) {
    /* Waits on both the swapchain image being ready and compute's write to the grid having finished. 
    Assumes cmd already contains the compute->graphics  memory barrier; that's recorded at the render_loop level, not here. */
    VkSemaphore          waitSemaphores[] = { imageAvailableSemaphores[currentFrame], computeFinishedSemaphores[currentFrame] };
    VkPipelineStageFlags waitStages[]     = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT };

    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.waitSemaphoreCount   = 2;
    submitInfo.pWaitSemaphores      = waitSemaphores;
    submitInfo.pWaitDstStageMask    = waitStages;
    submitInfo.commandBufferCount   = 1;
    submitInfo.pCommandBuffers      = &cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores    = &renderFinishedSemaphores[currentFrame];

    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
        throw std::runtime_error("failed to submit graphics command buffer");
}

void FrameResources::present(VkQueue presentQueue, VkSwapchainKHR swapChain, uint32_t imageIndex) {
    VkPresentInfoKHR presentInfo{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = &renderFinishedSemaphores[currentFrame];
    presentInfo.swapchainCount     = 1;
    presentInfo.pSwapchains        = &swapChain;
    presentInfo.pImageIndices      = &imageIndex;

    VkResult result = vkQueuePresentKHR(presentQueue, &presentInfo);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
        // TODO: trigger swapchain recreation
    } else if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swapchain image");
    }

    currentFrame = (currentFrame + 1) % Constants::MAX_FRAMES_IN_FLIGHT;
}