#include "debug_utils_messenger.h"

namespace {
    VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData)
    {
        // TODO: log pCallbackData->pMessage
        return VK_FALSE;
    }
}

VkResult create_debug_utils_messenger_EXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    // TODO: vkGetInstanceProcAddr lookup & call  or VK_ERROR_EXTENSION_NOT_PRESENT
    return VK_SUCCESS;
}

void destroy_debug_utils_messenger_EXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* pAllocator)
{
    // TODO: free ressources
}

void setup_debug_messenger(VkInstance instance, VkDebugUtilsMessengerEXT& debugMessenger) {
    // TODO: populate_debug_messenger_create_info(); create_debug_utils_messenger_EXT()
}

void populate_debug_messenger_create_info(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
    // TODO: fill createInfo, set createInfo.pfnUserCallback = debugCallback;
}