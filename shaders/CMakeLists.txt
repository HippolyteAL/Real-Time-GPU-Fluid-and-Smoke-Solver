find_program(DXC_EXECUTABLE dxc HINTS "$ENV{VULKAN_SDK}/Bin")
if(NOT DXC_EXECUTABLE)
    message(FATAL_ERROR "dxc not found — check Vulkan SDK install / VULKAN_SDK env var")
endif()

set(SHADER_COMMON ${CMAKE_CURRENT_SOURCE_DIR}/fluid_common.hlsli)
set(SHADER_OUTPUTS "")

function(add_hlsl_shader TARGET_NAME SOURCE PROFILE)
    set(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.spv)
    add_custom_command(
        OUTPUT ${OUTPUT}
        COMMAND ${DXC_EXECUTABLE} -spirv -T ${PROFILE} -E main
                ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE} -Fo ${OUTPUT}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE} ${SHADER_COMMON}
        COMMENT "Compiling ${SOURCE} -> ${TARGET_NAME}.spv"
        VERBATIM
    )
    set(SHADER_OUTPUTS ${SHADER_OUTPUTS} ${OUTPUT} PARENT_SCOPE)
endfunction()

add_hlsl_shader(buoyancy.comp           buoyancy.hlsl           cs_6_0)
add_hlsl_shader(advect_velocity.comp    advect_velocity.hlsl    cs_6_0)
add_hlsl_shader(divergence.comp         divergence.hlsl         cs_6_0)
add_hlsl_shader(jacobi.comp             jacobi.hlsl             cs_6_0)
add_hlsl_shader(projection.comp         projection.hlsl         cs_6_0)
add_hlsl_shader(advect_scalars.comp     advect_scalars.hlsl     cs_6_0)
add_hlsl_shader(boundary.comp           boundary.hlsl           cs_6_0)
add_hlsl_shader(fullscreen.vert         fullscreen.vert.hlsl    vs_6_0)
add_hlsl_shader(skybox.frag             skybox.frag.hlsl        ps_6_0)
add_hlsl_shader(raymarch.frag           raymarch.frag.hlsl      ps_6_0)
add_hlsl_shader(inject.comp             inject.hlsl             cs_6_0)

add_custom_target(shaders ALL DEPENDS ${SHADER_OUTPUTS})
add_dependencies(${PROJECT_NAME} shaders)