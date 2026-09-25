#ifndef VULKAN_CONTEXT_H
#define VULKAN_CONTEXT_H

#include <GLFW/glfw3.h>

#include <glm/glm.hpp>                  
#include <glm/gtc/constants.hpp>   // Camera

#include <array>        // DEVICE_EXTENSIONS, init_cubemap's faces param
#include <cmath>        // std::acos in Camera
#include <cstdint>      // uint32_t
#include <memory>       // std::unique_ptr
#include <optional>     // QueueFamilyIndices
#include <string> 
#include <vector>       // SwapChain / SwapChainSupportDetails

// Constants relevant to the vulkan pipeline
namespace Constants {
    inline constexpr uint32_t   WIDTH                   = 1280;
    inline constexpr uint32_t   HEIGHT                  = 720;
    inline constexpr int        MAX_FRAMES_IN_FLIGHT    = 2;
    inline constexpr std::array<const char*, 1> DEVICE_EXTENSIONS = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };
    inline constexpr uint32_t GRID_RESOLUTION = 64;     // TODO: 64^2 for now, ultimately aiming for 128^3 
}

// These structs hold temporary values needed for initialization, easier to handle when grouped
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily; // Can also be used for compute, and is used for compute
    std::optional<uint32_t> presentFamily;

    bool is_complete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR        capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;
};

// Render window
struct RenderWindow {
    GLFWwindow*                 window = nullptr;
    VkInstance                  instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT    debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR                surface = VK_NULL_HANDLE;
    VkPhysicalDevice            physicalDevice = VK_NULL_HANDLE;
    VkDevice                    device = VK_NULL_HANDLE;
    VkQueue                     graphicsQueue = VK_NULL_HANDLE;
    VkQueue                     presentQueue = VK_NULL_HANDLE;
};

// Swapchain
struct SwapChain {
    VkSwapchainKHR              swapChain = VK_NULL_HANDLE;
    std::vector<VkImage>        images;
    VkFormat                    imageFormat{};
    VkExtent2D                  extent{};
    std::vector<VkImageView>    imageViews;
    std::vector<VkFramebuffer>  framebuffers;
};

struct Camera {
    float camAzimuth        = glm::radians(0.0f);   // horizontal angle from the meridian 0
    float camPolar          = glm::radians(90.0f);  // vertical angle from pole
    float camDistance       = 3.0f;
    float CAM_ORBIT_SPEED   = 0.002f;
    float CAM_ZOOM_SPEED    = 0.5f;
    float CAM_DIST_MIN      = 0.01f;
    float CAM_POLAR_MIN     = std::acos(0.999f);
    float CAM_POLAR_MAX     = glm::pi<float>() - std::acos(0.999f);
};

struct  FluidGridResources; // compute_pipeline.h
class   ComputePipeline;    // compute_pipeline.h
struct  Cubemap;            // graphics_pipeline.h
class   GraphicsPipeline;   // graphics_pipeline.h
class   FrameResources;     // frame_resources.h

class VulkanContext {
public:
    // Call before implementing/testing anything that requires an extension
    void is_extension_available(const std::string& extensionName);

    // Core
    VulkanContext();
    ~VulkanContext();
    void render_loop();

private:
    // Parameters
    RenderWindow    renderWindow;
    SwapChain       swapChain;
    Camera          camera;

    std::unique_ptr<FluidGridResources> fluidGrid;
    std::unique_ptr<ComputePipeline>    computePipeline;
    std::unique_ptr<GraphicsPipeline>   graphicsPipeline;
    std::unique_ptr<FrameResources>     frameResources;

    // Core initialization
    void init_window(uint32_t width = Constants::WIDTH, uint32_t height = Constants::HEIGHT);
    void init_vulkan();
    void cleanup();

    // Controls, call in render_loop()
    void process_input();
};

#endif  // VULKAN_CONTEXT_H