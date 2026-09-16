#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <iostream>

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    uint32_t extensionCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
    std::cout << extensionCount << " Vulkan extensions supported\n";

    glm::mat4 testMatrix(1.0f);
    (void)testMatrix;

    glfwTerminate();
    return 0;
}