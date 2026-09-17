#ifndef COMPUTE_PIPELINE_H
#define COMPUTE_PIPELINE_H

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

// Simulation data
struct FluidGridResources {
    VkImage         velocity[2];            // needs ping-pong for advection
    VkImageView     velocityView[2];
    VkDeviceMemory  velocityMemory[2];
    VkImage         density[2];
    VkImageView     densityView[2];
    VkDeviceMemory  densityMemory[2];
    VkImage         pressure[2];            // needs ping-pong for Jacobi
    VkImageView     pressureView[2];
    VkDeviceMemory  pressureMemory[2];
    VkImage         temperature[2];
    VkImageView     temperatureView[2];
    VkDeviceMemory  temperatureMemory[2];
    uint32_t        gridResolution;         // Aiming for comfortable 128^3
};

class ComputePipeline {
public:
    void init(VkDevice device, uint32_t computeQueueFamily);
    void cleanup(VkDevice device);
    // Orchestrates one full timestep in dependency order, inserting barriers between passes. This is what render_loop() should call normally.
    void record(VkCommandBuffer cmd, const FluidGridResources& grid, float dt);  
    // Individual passes for easier profiling
    void record_buoyancy(VkCommandBuffer cmd, const FluidGridResources& grid, float dt);
    void record_advect_velocity(VkCommandBuffer cmd, const FluidGridResources& grid, float dt);
    void record_divergence(VkCommandBuffer cmd, const FluidGridResources& grid);
    void record_jacobi_iteration(VkCommandBuffer cmd, const FluidGridResources& grid);
    void record_projection(VkCommandBuffer cmd, const FluidGridResources& grid);
    void record_advect_scalars(VkCommandBuffer cmd, const FluidGridResources& grid, float dt);  // density + temperature, post-projection
    void record_boundary(VkCommandBuffer cmd, const FluidGridResources& grid);
    uint32_t current_field_index() const { return fieldPingPong; }                              // lets GraphicsPipeline know which buffer to sample

    uint32_t jacobiIterations = 40;   // runtime tunable

private:
    VkPipeline buoyancyPipeline;
    VkPipeline advectVelocityPipeline;
    VkPipeline divergencePipeline;      // compute velocity divergence, feeds Jacobi
    VkPipeline jacobiPipeline;
    VkPipeline projectionPipeline;      // subtract pressure gradient
    VkPipeline advectScalarsPipeline;
    VkPipeline boundaryPipeline;
    uint32_t fieldPingPong = 0;         // velocity/density/temperature parity, flips once per record()
    uint32_t pressurePingPong = 0;      // pressure parity, flips once per Jacobi iteration, reset each frame

    VkPipelineLayout              layout;               // can be shared as long as push constant layout matches
    VkDescriptorSetLayout         descriptorSetLayout; 
    VkDescriptorPool              descriptorPool;
    std::vector<VkDescriptorSet>  descriptorSets;

    void barrier(VkCommandBuffer cmd);  // shared compute-to-compute memory barrier
};

#endif // COMPUTE_PIPELINE_H