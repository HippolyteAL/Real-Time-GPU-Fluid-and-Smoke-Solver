#include "compute_pipeline.h"

void ComputePipeline::init(VkDevice device, uint32_t computeQueueFamily) {
    // TODO
}

void ComputePipeline::cleanup(VkDevice device) {
    // TODO
}

void ComputePipeline::record(VkCommandBuffer cmd, const FluidGridResources& grid, float dt) {
    // TODO: orchestrate passes in order, with barrier() between each — see prior message
}

void ComputePipeline::record_buoyancy(VkCommandBuffer cmd, const FluidGridResources& grid, float dt) {
    // TODO
}

void ComputePipeline::record_advect_velocity(VkCommandBuffer cmd, const FluidGridResources& grid, float dt) {
    // TODO
}

void ComputePipeline::record_divergence(VkCommandBuffer cmd, const FluidGridResources& grid) {
    // TODO
}

void ComputePipeline::record_jacobi_iteration(VkCommandBuffer cmd, const FluidGridResources& grid) {
    // TODO: bind descriptor set based on pressurePingPong parity
}

void ComputePipeline::record_projection(VkCommandBuffer cmd, const FluidGridResources& grid) {
    // TODO
}

void ComputePipeline::record_advect_scalars(VkCommandBuffer cmd, const FluidGridResources& grid, float dt) {
    // TODO: density + temperature, post-projection velocity
}

void ComputePipeline::record_boundary(VkCommandBuffer cmd, const FluidGridResources& grid) {
    // TODO
}

void ComputePipeline::barrier(VkCommandBuffer cmd) {
    // TODO: vkCmdPipelineBarrier, compute-to-compute memory barrier
}