cmake_minimum_required(VERSION 3.14)
project(s2cpp LANGUAGES C CXX)

if (WIN32)
    add_definitions(-DWIN32_LEAN_AND_MEAN)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(S2_VULKAN  "Build with Vulkan backend"  OFF)
option(S2_CUDA    "Build with CUDA backend"    OFF)
option(S2_METAL   "Build with Metal backend"   OFF)

# ---------------------------------------------------------------------------
# GGML (git submodule — not modified)
# ---------------------------------------------------------------------------
set(GGML_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

if(S2_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(S2_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()
if(S2_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(ggml)

# ---------------------------------------------------------------------------
# s2 executable
# ---------------------------------------------------------------------------
set(S2_SOURCES
    src/s2_audio.cpp
    src/s2_tokenizer.cpp
    src/s2_sampler.cpp
    src/s2_model.cpp
    src/s2_codec.cpp
    src/s2_prompt.cpp
    src/s2_generate.cpp
    src/s2_pipeline.cpp
    src/s2_server.cpp
    src/s2_voice.cpp
    src/main.cpp
)

add_executable(s2 ${S2_SOURCES})

target_include_directories(s2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ggml/src
)

target_link_libraries(s2 PRIVATE ggml)

if(S2_VULKAN)
    target_compile_definitions(s2 PRIVATE GGML_USE_VULKAN)
endif()
if(S2_CUDA)
    target_compile_definitions(s2 PRIVATE GGML_USE_CUDA)
endif()

# Platform-specific
if(UNIX AND NOT APPLE)
    target_link_libraries(s2 PRIVATE pthread m)
endif()

if(WIN32)
    target_link_libraries(s2 PRIVATE ws2_32 wsock32)
endif()

# Install
install(TARGETS s2 RUNTIME DESTINATION bin)
