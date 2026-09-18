#include "vulkan_context.h"

#include "compute_pipeline.h"       // full def of ComputePipeline & FluidGridResources
#include "graphics_pipeline.h"      // full def of GraphicsPipeline & Cubemap
#include "frame_resources.h"        // full def of FrameResources
#include "debug_utils_messenger.h"  // debug messenger

#include <GLFW/glfw3native.h>       // Needs system to be defined in CMakeList.txt

#include <glm/glm.hpp>                  
#include <glm/gtc/constants.hpp>    // Camera

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>

/* Implementation detail helpers. None of these need anything beyond what's passed in, 
so there's no reason to give them access to VulkanContext's private state. */
namespace {

bool check_validation_layer_support() {
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> available(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, available.data());

    for (const char* required : Constants::validationLayers) {
        bool found = false;
        for (const auto& props : available) {
            if (std::strcmp(required, props.layerName) == 0) { found = true; break; }
        }
        if (!found) return false;
    }
    return true;
}

std::vector<const char*> get_required_instance_extensions() {
    uint32_t glfwCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwCount);

    if (Constants::enableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    return extensions;
}

void log_instance_extension_support(const std::vector<const char*>& required) {
    uint32_t availableCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &availableCount, nullptr);
    std::vector<VkExtensionProperties> available(availableCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &availableCount, available.data());

    std::cout << "[init] " << availableCount << " instance extensions available, "
              << required.size() << " required:\n";

    for (const char* req : required) {
        bool found = false;
        for (const auto& ext : available) {
            if (std::strcmp(req, ext.extensionName) == 0) { found = true; break; }
        }
        std::cout << "  [" << (found ? "OK" : "MISSING") << "] " << req << "\n";
        if (!found) {
            throw std::runtime_error(std::string("required instance extension not found: ") + req);
        }
    }
}

QueueFamilyIndices find_queue_families(VkPhysicalDevice device, VkSurfaceKHR surface) {
    QueueFamilyIndices indices;

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &familyCount, families.data());

    for (uint32_t i = 0; i < familyCount; ++i) {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphicsFamily = i;
        }
        VkBool32 presentSupport = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
        if (presentSupport) {
            indices.presentFamily = i;
        }
        if (indices.is_complete()) break;
    }
    return indices;
}

bool check_device_extension_support(VkPhysicalDevice device) {
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> available(extCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extCount, available.data());

    for (const char* required : Constants::DEVICE_EXTENSIONS) {
        bool found = false;
        for (const auto& ext : available) {
            if (std::strcmp(required, ext.extensionName) == 0) { found = true; break; }
        }
        if (!found) return false;
    }
    return true;
}

SwapChainSupportDetails query_swap_chain_support(VkPhysicalDevice device, VkSurfaceKHR surface) {
    SwapChainSupportDetails details;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (formatCount) {
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
    }

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    if (presentModeCount) {
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
    }
    return details;
}

bool is_device_suitable(VkPhysicalDevice device, VkSurfaceKHR surface) {
    QueueFamilyIndices indices = find_queue_families(device, surface);
    bool extensionsSupported = check_device_extension_support(device);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
        SwapChainSupportDetails support = query_swap_chain_support(device, surface);
        swapChainAdequate = !support.formats.empty() && !support.presentModes.empty();
    }
    return indices.is_complete() && extensionsSupported && swapChainAdequate;
}

VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }
    return formats[0];
}

VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR>& modes) {
    for (const auto& m : modes) {
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities, GLFWwindow* window) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    VkExtent2D extent{ static_cast<uint32_t>(width), static_cast<uint32_t>(height) };
    extent.width  = std::clamp(extent.width,  capabilities.minImageExtent.width,  capabilities.maxImageExtent.width);
    extent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    return extent;
}

void create_surface(RenderWindow& window) {
#ifdef _WIN32
    VkWin32SurfaceCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
    createInfo.hwnd      = glfwGetWin32Window(window.window);
    createInfo.hinstance = GetModuleHandle(nullptr);

    if (vkCreateWin32SurfaceKHR(window.instance, &createInfo, nullptr, &window.surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create Win32 surface");
    }
#else
    #error "create_surface: only Win32 surface creation is implemented"
#endif
}

void create_swap_chain(RenderWindow& window, SwapChain& sc) {
    SwapChainSupportDetails support = query_swap_chain_support(window.physicalDevice, window.surface);

    VkSurfaceFormatKHR surfaceFormat = choose_swap_surface_format(support.formats);
    VkPresentModeKHR   presentMode   = choose_swap_present_mode(support.presentModes);
    VkExtent2D         extent        = choose_swap_extent(support.capabilities, window.window);

    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount) {
        imageCount = support.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    createInfo.surface          = window.surface;
    createInfo.minImageCount    = imageCount;
    createInfo.imageFormat      = surfaceFormat.format;
    createInfo.imageColorSpace  = surfaceFormat.colorSpace;
    createInfo.imageExtent      = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = find_queue_families(window.physicalDevice, window.surface);
    uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices   = queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform   = support.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode    = presentMode;
    createInfo.clipped        = VK_TRUE;
    createInfo.oldSwapchain   = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(window.device, &createInfo, nullptr, &sc.swapChain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swapchain");
    }

    vkGetSwapchainImagesKHR(window.device, sc.swapChain, &imageCount, nullptr);
    sc.images.resize(imageCount);
    vkGetSwapchainImagesKHR(window.device, sc.swapChain, &imageCount, sc.images.data());

    sc.imageFormat = surfaceFormat.format;
    sc.extent      = extent;

    sc.imageViews.resize(sc.images.size());
    for (size_t i = 0; i < sc.images.size(); ++i) {
        VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        viewInfo.image    = sc.images[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format   = sc.imageFormat;
        viewInfo.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                 VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
        viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        if (vkCreateImageView(window.device, &viewInfo, nullptr, &sc.imageViews[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swapchain image view");
        }
    }
}

} //namespace

VulkanContext::VulkanContext() {
    std::cout << "==== VulkanContext startup ====\n";
    init_window();
    init_vulkan();
    std::cout << "==== initialization complete ====\n";
}

VulkanContext::~VulkanContext() {
    cleanup();
}

void VulkanContext::render_loop() {
    while (!glfwWindowShouldClose(renderWindow.window)) {
        process_input();
        
        // TODO: uint32_t imageIndex = frameResources->begin_frame(...);
        // TODO: computePipeline->record(...) on the compute cmd buffer, submit
        // TODO: graphicsPipeline->record_skybox(...) / record_volume(...), submit
        // TODO: frameResources->present(...)
    }
    
    vkDeviceWaitIdle(renderWindow.device);
}

void VulkanContext::process_input() {
    glfwPollEvents();
    // TODO: camera orbit/zoom key handling (nothing to see yet)
}
void VulkanContext::init_window(uint32_t width, uint32_t height) {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);   // no default OpenGL context

    renderWindow.window = glfwCreateWindow(
        static_cast<int>(width), static_cast<int>(height),
        "Vulkan Fluid Solver", nullptr, nullptr);

    if (!renderWindow.window) {
        throw std::runtime_error("failed to create GLFW window");
    }

    std::cout << "[init] window created (" << width << "x" << height << ")\n";
}

void VulkanContext::init_vulkan() {
    // Instance 
    if (Constants::enableValidationLayers && !check_validation_layer_support()) {
        throw std::runtime_error("validation layers requested but not supported");
    }

    VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    appInfo.pApplicationName   = "Vulkan Fluid Solver";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName        = "No Engine";
    appInfo.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion         = VK_API_VERSION_1_3;

    auto requiredExtensions = get_required_instance_extensions();
    log_instance_extension_support(requiredExtensions);                                                 // throws if anything is missing

    VkInstanceCreateInfo instanceInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    instanceInfo.pApplicationInfo        = &appInfo;
    instanceInfo.enabledExtensionCount   = static_cast<uint32_t>(requiredExtensions.size());
    instanceInfo.ppEnabledExtensionNames = requiredExtensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (Constants::enableValidationLayers) {
        instanceInfo.enabledLayerCount   = static_cast<uint32_t>(Constants::validationLayers.size());
        instanceInfo.ppEnabledLayerNames = Constants::validationLayers.data();

        populate_debug_messenger_create_info(debugCreateInfo);
        instanceInfo.pNext = &debugCreateInfo;
    } else {
        instanceInfo.enabledLayerCount = 0;
    }

    if (vkCreateInstance(&instanceInfo, nullptr, &renderWindow.instance) != VK_SUCCESS) {
        throw std::runtime_error("failed to create Vulkan instance");
    }
    std::cout << "[init] Vulkan instance created ("
              << (Constants::enableValidationLayers ? "validation ON" : "validation OFF") << ")\n";

    if (Constants::enableValidationLayers) {
        setup_debug_messenger(renderWindow.instance, renderWindow.debugMessenger);
        std::cout << "[init] debug messenger attached\n";
    }

    // Surface 
    create_surface(renderWindow);
    std::cout << "[init] window surface created\n";

    // Physical device 
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(renderWindow.instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        throw std::runtime_error("no GPUs with Vulkan support found");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(renderWindow.instance, &deviceCount, devices.data());

    std::cout << "[init] " << deviceCount << " physical device(s) found:\n";
    for (const auto& device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        bool suitable = is_device_suitable(device, renderWindow.surface);
        std::cout << "  [" << (suitable ? "suitable" : "skipped ") << "] " << props.deviceName << "\n";
        if (suitable && renderWindow.physicalDevice == VK_NULL_HANDLE) {
            renderWindow.physicalDevice = device;
        }
    }
    if (renderWindow.physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("no suitable GPU found");
    }

    // Logical device & queues 
    QueueFamilyIndices indices = find_queue_families(renderWindow.physicalDevice, renderWindow.surface);

    // No dedicated compute queue family is tracked (QueueFamilyIndices only has graphics/present) compute submissions reuse the graphics family. 
    uint32_t computeFamily = indices.graphicsFamily.value();
    std::cout << "[init] compute submissions reuse graphics queue family (family "
              << computeFamily << ")\n";

    std::set<uint32_t> uniqueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };
    float queuePriority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    for (uint32_t family : uniqueFamilies) {
        VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qci.queueFamilyIndex = family;
        qci.queueCount       = 1;
        qci.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(qci);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};   // TODO: none enabled yet, revisit once shaders need specific features

    VkDeviceCreateInfo deviceInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    deviceInfo.pQueueCreateInfos       = queueCreateInfos.data();
    deviceInfo.queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size());
    deviceInfo.pEnabledFeatures        = &deviceFeatures;
    deviceInfo.enabledExtensionCount   = static_cast<uint32_t>(Constants::DEVICE_EXTENSIONS.size());
    deviceInfo.ppEnabledExtensionNames = Constants::DEVICE_EXTENSIONS.data();
    if (Constants::enableValidationLayers) {
        deviceInfo.enabledLayerCount   = static_cast<uint32_t>(Constants::validationLayers.size());
        deviceInfo.ppEnabledLayerNames = Constants::validationLayers.data();
    } else {
        deviceInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(renderWindow.physicalDevice, &deviceInfo, nullptr, &renderWindow.device) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device");
    }

    vkGetDeviceQueue(renderWindow.device, indices.graphicsFamily.value(), 0, &renderWindow.graphicsQueue);
    vkGetDeviceQueue(renderWindow.device, indices.presentFamily.value(),  0, &renderWindow.presentQueue);

    std::cout << "[init] logical device created (graphics family " << indices.graphicsFamily.value()
              << ", present family " << indices.presentFamily.value() << ")\n";
    for (const char* ext : Constants::DEVICE_EXTENSIONS) {
        std::cout << "  [OK] device extension: " << ext << "\n";
    }

    // Swapchain
    create_swap_chain(renderWindow, swapChain);
    std::cout << "[init] swapchain created (" << swapChain.images.size() << " images, "
              << swapChain.extent.width << "x" << swapChain.extent.height << ")\n";

    VkPhysicalDeviceProperties selectedProps;
    vkGetPhysicalDeviceProperties(renderWindow.physicalDevice, &selectedProps);
    float timestampPeriodNs = selectedProps.limits.timestampPeriod;

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(renderWindow.physicalDevice, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(renderWindow.physicalDevice, &familyCount, families.data());

    bool timestampsSupported = families[computeFamily].timestampValidBits > 0;
    std::cout << "[init] timestamp queries " << (timestampsSupported ? "supported" : "NOT supported")
              << " on compute family " << computeFamily
              << " (period = " << timestampPeriodNs << " ns/tick)\n";
    if (!timestampsSupported) {
        throw std::runtime_error("compute queue family does not support timestamp queries");
    }

    // Subsystems
    computePipeline  = std::make_unique<ComputePipeline>();
    graphicsPipeline = std::make_unique<GraphicsPipeline>();
    frameResources   = std::make_unique<FrameResources>();
    fluidGrid        = std::make_unique<FluidGridResources>();

    computePipeline->init(renderWindow.device, computeFamily, Constants::MAX_FRAMES_IN_FLIGHT, timestampPeriodNs);
    computePipeline->allocate_grid(renderWindow.device, renderWindow.physicalDevice, renderWindow.graphicsQueue, computeFamily, *fluidGrid, Constants::GRID_RESOLUTION);
    graphicsPipeline->init(renderWindow.device, swapChain.imageFormat, swapChain.extent);
    graphicsPipeline->init_cubemap(renderWindow.device);
    frameResources->init(renderWindow.device, indices.graphicsFamily.value(), computeFamily);

    std::cout << "[init] compute pipeline, graphics pipeline, and frame resources constructed\n";
}

void VulkanContext::is_extension_available(const std::string& extensionName) {
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data());

    bool found = false;
    for (const auto& ext : extensions) {
        if (extensionName == ext.extensionName) { found = true; break; }
    }
    std::cout << "[check] " << extensionName << ": " << (found ? "available" : "NOT available") << "\n";
}

void VulkanContext::cleanup() {
    if (renderWindow.device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(renderWindow.device);
    }

    if (computePipeline && fluidGrid) computePipeline->free_grid(renderWindow.device, *fluidGrid);
    if (computePipeline)  computePipeline->cleanup(renderWindow.device);
    if (graphicsPipeline) graphicsPipeline->cleanup(renderWindow.device);
    if (frameResources)   frameResources->cleanup(renderWindow.device);

    // TODO: destroy fluidGrid's images/views/memory and cubemap's image/view/sampler
    //       once their allocation is implemented — nothing to free yet.

    for (VkImageView view : swapChain.imageViews) {
        vkDestroyImageView(renderWindow.device, view, nullptr);
    }
    if (swapChain.swapChain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(renderWindow.device, swapChain.swapChain, nullptr);
    }
    if (renderWindow.device != VK_NULL_HANDLE) {
        vkDestroyDevice(renderWindow.device, nullptr);
    }
    if (Constants::enableValidationLayers && renderWindow.debugMessenger != VK_NULL_HANDLE) {
        destroy_debug_utils_messenger_EXT(renderWindow.instance, renderWindow.debugMessenger, nullptr);
    }
    if (renderWindow.surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(renderWindow.instance, renderWindow.surface, nullptr);
    }
    if (renderWindow.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(renderWindow.instance, nullptr);
    }
    if (renderWindow.window) {
        glfwDestroyWindow(renderWindow.window);
    }
    glfwTerminate();

    std::cout << "[cleanup] all Vulkan/GLFW resources destroyed\n";
}