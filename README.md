# Real-Time GPU Fluid & Smoke Solver in Vulkan

## Introduction

## Building and running

I'm keeping these in my README because I can easily acces it to copy paste, this can be built with other tools, but the filepath for the shaders expects shaders built in <.exe dir>/shaders/, which is what happens in single-config generators (like Ninja), but not in multi-configs ones (like the VS generator, for example).

cmake -B build -S . -G Ninja -DCMAKE_TOOLCHAIN_FILE=Drive:/path_to/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Debug
cmake --build build
.\build\VulkanFluidSolver.exe