# Real-Time GPU Fluid & Smoke Solver in Vulkan

## Introduction

## Building and running

windows (powershell, called in root)

cmake -B build -S . -G Ninja -DCMAKE_TOOLCHAIN_FILE=C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Debug
cmake --build build
.\build\VulkanFluidSolver.exe