cmake_minimum_required(VERSION 3.24)

project(VulkanFluidSolver
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm CONFIG REQUIRED)

# Application
add_executable(VulkanFluidSolver
    src/main.cpp
    src/vulkan_context.h
    src/vulkan_context.cpp
    src/debug_utils_messenger.h
    src/debug_utils_messenger.cpp
    src/frame_resources.h
    src/frame_resources.cpp
    src/compute_pipeline.h 
    src/compute_pipeline.cpp 
    src/graphics_pipeline.h 
    src/graphics_pipeline.cpp 
)

# Link dependencies
target_link_libraries(VulkanFluidSolver
    PRIVATE
        glfw
        Vulkan::Vulkan
        glm::glm
)

target_compile_definitions(VulkanFluidSolver
    PRIVATE
        GLFW_INCLUDE_VULKAN
)

# Windows specific definitions
if(WIN32)
    target_compile_definitions(VulkanFluidSolver
        PRIVATE
            NOMINMAX
            VK_USE_PLATFORM_WIN32_KHR
            GLFW_EXPOSE_NATIVE_WIN32
    )
endif()

add_subdirectory(shaders)

target_include_directories(VulkanFluidSolver PRIVATE includes)

if(EXISTS ${CMAKE_SOURCE_DIR}/assets)
    add_custom_command(TARGET VulkanFluidSolver POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:VulkanFluidSolver>/assets
        COMMENT "Copying assets/ to output directory"
    )
else()
    message(STATUS "assets/ not found (gitignored) — skipping asset copy")
endif()