cmake_minimum_required(VERSION 3.14)
project(s2cpp LANGUAGES C CXX)

if (WIN32)
    add_definitions(-DWIN32_LEAN_AND_MEAN)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(S2_VULKAN  "Build with Vulkan backend"  OFF)
option(S2_CUDA    "Build with CUDA backend"    OFF)
option(S2_METAL   "Build with Metal backend"   OFF)
option(S2_BUILD_SHARED_LIBRARIES "Build S2 shared and static library targets" OFF)
option(S2_AUTO_APPLY_LOCAL_PATCHES "Automatically apply local patches from patches/*.patch during build" ON)

if(S2_METAL AND NOT APPLE)
    message(FATAL_ERROR "S2_METAL is only supported on Apple platforms.")
endif()

# MSVC provides math functions through the CRT and does not ship a separate
# libm. Some ggml CMake revisions still probe for "m" and can propagate
# MATH_LIBRARY-NOTFOUND into the generated Visual Studio linker inputs.
if(MSVC)
    set(MATH_LIBRARY "" CACHE FILEPATH "MSVC does not require a separate math library." FORCE)
endif()

set(GGML_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

if(S2_AUTO_APPLY_LOCAL_PATCHES)

	if(WIN32)
		find_program(S2_PATCH_EXECUTABLE git)
	else()
		find_program(S2_PATCH_EXECUTABLE patch)
	endif()
	
    if(NOT S2_PATCH_EXECUTABLE)
        message(FATAL_ERROR "S2_AUTO_APPLY_LOCAL_PATCHES=ON requires the 'patch' executable")
    endif()

    add_custom_target(s2_apply_local_patches
        COMMAND ${CMAKE_COMMAND}
            -DS2_PATCH_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
            -DS2_PATCH_EXECUTABLE=${S2_PATCH_EXECUTABLE}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/apply_local_patches.cmake
        COMMENT "Applying local source patches"
        VERBATIM
    )
endif()

if(S2_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
else()
    set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
endif()
if(S2_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
else()
    set(GGML_CUDA OFF CACHE BOOL "" FORCE)
endif()
if(S2_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
else()
    set(GGML_METAL OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(ggml)

if(TARGET s2_apply_local_patches)
    foreach(_target ggml ggml-base ggml-cpu ggml-cuda ggml-vulkan ggml-metal)
        if(TARGET ${_target})
            add_dependencies(${_target} s2_apply_local_patches)
        endif()
    endforeach()
endif()

set(S2_CORE_SOURCES
    src/s2_audio.cpp
    src/s2_tokenizer.cpp
    src/s2_log.cpp
    src/s2_sampler.cpp
    src/s2_model.cpp
    src/s2_codec.cpp
    src/s2_prompt.cpp
    src/s2_generate.cpp
    src/s2_pipeline.cpp
    src/s2_server.cpp
    src/s2_voice.cpp
)

function(s2_configure_target target_name)
target_include_directories(${target_name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src
)

target_link_libraries(${target_name} PRIVATE ggml)

if(S2_VULKAN)
    target_compile_definitions(${target_name} PRIVATE GGML_USE_VULKAN)
endif()
if(S2_CUDA)
    target_compile_definitions(${target_name} PRIVATE GGML_USE_CUDA)
endif()
if(S2_METAL)
    target_compile_definitions(${target_name} PRIVATE GGML_USE_METAL)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(${target_name} PRIVATE pthread m)
endif()

if(WIN32)
    target_link_libraries(${target_name} PRIVATE ws2_32 wsock32)
endif()
endfunction()

add_executable(s2 ${S2_CORE_SOURCES} src/main.cpp)
s2_configure_target(s2)
if(TARGET s2_apply_local_patches)
    add_dependencies(s2 s2_apply_local_patches)
endif()

if(S2_BUILD_SHARED_LIBRARIES)
    set(S2_LIBRARY_SOURCES
        ${S2_CORE_SOURCES}
        src/s2_export_api.cpp
    )

    add_library(s2_shared SHARED ${S2_LIBRARY_SOURCES})
    set_target_properties(s2_shared PROPERTIES OUTPUT_NAME s2)
    s2_configure_target(s2_shared)
    target_compile_definitions(s2_shared PRIVATE S2_LIBRARY)
    if(TARGET s2_apply_local_patches)
        add_dependencies(s2_shared s2_apply_local_patches)
    endif()

    add_library(s2_static STATIC ${S2_LIBRARY_SOURCES})
    set_target_properties(s2_static PROPERTIES OUTPUT_NAME s2_static)
    s2_configure_target(s2_static)
    target_compile_definitions(s2_static PRIVATE S2_LIBRARY S2_STATIC)
    if(TARGET s2_apply_local_patches)
        add_dependencies(s2_static s2_apply_local_patches)
    endif()

    install(TARGETS s2_shared
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
    install(TARGETS s2_static
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
endif()

install(TARGETS s2 RUNTIME DESTINATION bin)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h")
