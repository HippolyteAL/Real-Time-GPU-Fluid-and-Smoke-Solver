#ifndef COMPUTE_PIPELINE_H
#define COMPUTE_PIPELINE_H

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

// Simulation data
struct FluidGridResources {
    VkImage         velocity[2]             = { VK_NULL_HANDLE, VK_NULL_HANDLE };    // needs ping-pong for advection
    VkImageView     velocityView[2]         = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDeviceMemory  velocityMemory[2]       = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkImage         density[2]              = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkImageView     densityView[2]          = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDeviceMemory  densityMemory[2]        = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkImage         pressure[2]             = { VK_NULL_HANDLE, VK_NULL_HANDLE };    // needs ping-pong for Jacobi
    VkImageView     pressureView[2]         = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDeviceMemory  pressureMemory[2]       = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkImage         temperature[2]          = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkImageView     temperatureView[2]      = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDeviceMemory  temperatureMemory[2]    = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkImage         divergence              = VK_NULL_HANDLE;
    VkImageView     divergenceView          = VK_NULL_HANDLE;
    VkDeviceMemory  divergenceMemory        = VK_NULL_HANDLE;
    uint32_t        gridResolution          = 0;                                    // Aiming for comfortable 128^3
};

class ComputePipeline {
public:
    void init(VkDevice device, uint32_t computeQueueFamily, uint32_t framesInFlight, float timestampPeriod);
    void cleanup(VkDevice device);
    // Orchestrates one full timestep in dependency order, inserting barriers between passes. This is what render_loop() should call normally.
    void record(VkCommandBuffer cmd, const FluidGridResources& grid, float dt, uint32_t frameIndex);  
    // Profiling, called once per frame, right after FrameResources::begin_frame() has waited that frame's fence, to identify optimisation candidates.
    void read_timestamp_results(VkDevice device, uint32_t frameIndex);
    void log_timings() const;
    // Individual passes for easier profiling
    void record_buoyancy(VkCommandBuffer cmd, const FluidGridResources& grid, float dt);
    void record_advect_velocity(VkCommandBuffer cmd, const FluidGridResources& grid, float dt);
    void record_divergence(VkCommandBuffer cmd, const FluidGridResources& grid);
    void record_jacobi_iteration(VkCommandBuffer cmd, const FluidGridResources& grid);
    void record_projection(VkCommandBuffer cmd, const FluidGridResources& grid);
    void record_advect_scalars(VkCommandBuffer cmd, const FluidGridResources& grid, float dt);  // density + temperature, post-projection
    void record_boundary(VkCommandBuffer cmd, const FluidGridResources& grid);
    uint32_t current_field_index() const { return fieldPingPong; }                              // lets GraphicsPipeline know which buffer to sample
    void allocate_grid(VkDevice device, VkPhysicalDevice physicalDevice, VkQueue queue, uint32_t queueFamily, FluidGridResources& grid, uint32_t resolution);
    void free_grid(VkDevice device, FluidGridResources& grid);
    void update_descriptor_sets(VkDevice device, const FluidGridResources& grid);

    uint32_t jacobiIterations = 40;   // runtime tunable

private:
    // Performance logging enum
    enum TimingRegion : uint32_t {
        kTimingBuoyancy = 0,
        kTimingAdvectVelocity,
        kTimingBoundaryPre,     // boundary enforcement before the pressure solve
        kTimingDivergence,
        kTimingJacobi,          // whole 40 iteration jacobi loop timed as one region
        kTimingProjection,
        kTimingBoundaryPost,    // boundary enforcement after projection
        kTimingAdvectScalars,
        kTimingRegionCount
    };
    std::vector<VkQueryPool> timestampPools;                        // one per frame-in-flight
    float                    timestampPeriodNs = 1.0f;              // VkPhysicalDeviceLimits::timestampPeriod
    float                    lastTimings[kTimingRegionCount] = {};

    VkPipeline      buoyancyPipeline;
    VkPipeline      advectVelocityPipeline;
    VkPipeline      divergencePipeline;         // compute velocity divergence, feeds Jacobi
    VkPipeline      jacobiPipeline;
    uint32_t        fieldPingPong = 0;          // velocity/density/temperature parity, flips once per record()
    uint32_t        pressurePingPong = 0;       // pressure parity, flips once per Jacobi iteration, reset each frame
    VkPipeline      projectionPipeline;         // subtract pressure gradient
    VkPipeline      advectScalarsPipeline;
    VkPipeline      boundaryPipeline;

    VkPipelineLayout              layout;               // can be shared as long as push constant layout matches
    VkDescriptorSetLayout         descriptorSetLayout; 
    VkDescriptorPool              descriptorPool;
    std::vector<VkDescriptorSet>  descriptorSets;

    void barrier(VkCommandBuffer cmd);  // shared compute-to-compute memory barrier
};

#endif // COMPUTE_PIPELINE_H