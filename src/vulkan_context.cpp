#include "vulkan_context.h"

#include "compute_pipeline.h"       // full def of ComputePipeline & FluidGridResources
#include "graphics_pipeline.h"      // full def of GraphicsPipeline & Cubemap
#include "frame_resources.h"        // full def of FrameResources
#include "debug_utils_messenger.h"  // debug messenger

#include <GLFW/glfw3native.h>       // Needs system to be defined in CMakeList.txt

#include <glm/glm.hpp>                  
#include <glm/gtc/constants.hpp>    // Camera

#include <array>        // DEVICE_EXTENSIONS, init_cubemap's faces param
#include <cmath>        // std::acos in Camera
#include <cstdint>      // uint32_t
#include <memory>       // std::unique_ptr
#include <optional>     // QueueFamilyIndices
#include <string> 
#include <vector>       // SwapChain / SwapChainSupportDetails

VulkanContext::VulkanContext() {
    init_window(); 
    init_vulkan();
}

VulkanContext::~VulkanContext() {
    cleanup();
    // TODO?: check how std::unique_ptr should be handled
}

void VulkanContext::render_loop() {
    // TODO: poll input, dispatch compute, record + submit graphics, present
}

void VulkanContext::process_input() {
    // TODO: glfwPollEvents, camera orbit/zoom key handling
}

void VulkanContext::is_extension_available(const std::string& extensionName) {
    // TODO: query vkEnumerateInstanceExtensionProperties, search for extensionName, log/show result
}

void VulkanContext::init_window(uint32_t width, uint32_t height) {
    // TODO: glfwInit, glfwCreateWindow, store into renderWindow.window
}

void VulkanContext::init_vulkan() {
    /* TODO: create instance, setup_debug_messenger(...), create surface, pick physical device, 
    create logical device + queues, build swapChain, then:

    fluidGrid        = std::make_unique<FluidGridResources>();
    computePipeline  = std::make_unique<ComputePipeline>();
    cubemap          = std::make_unique<Cubemap>(); 
    graphicsPipeline = std::make_unique<GraphicsPipeline>();
    frameResources   = std::make_unique<FrameResources>();
    */
}

void VulkanContext::cleanup() {
    // TODO: destroy in reverse dependency order
}