#include "vulkan_context.h"

#include <iostream>
#include <exception>

int main() {
    try {
        VulkanContext context;   // constructor runs init_window() & init_vulkan()
        context.render_loop();
    }
    catch (const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}