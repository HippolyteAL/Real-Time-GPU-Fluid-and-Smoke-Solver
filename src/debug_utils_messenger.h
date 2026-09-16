#ifndef DEBUG_UTILS_MESSENGER_H
#define DEBUG_UTILS_MESSENGER_H

#include <vulkan/vulkan.h>

#include <array>

namespace Constants {
    inline constexpr std::array<const char*, 1> validationLayers = { "VK_LAYER_KHRONOS_validation" };
    #ifdef NDEBUG
        inline constexpr bool enableValidationLayers = false;
    #else
        inline constexpr bool enableValidationLayers = true;
    #endif
}

VkResult create_debug_utils_messenger_EXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);
void destroy_debug_utils_messenger_EXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);
void setup_debug_messenger(VkInstance instance, VkDebugUtilsMessengerEXT& debugMessenger);
void populate_debug_messenger_create_info(VkDebugUtilsMessengerCreateInfoEXT& createInfo);

#endif // DEBUG_UTILS_MESSENGER_H