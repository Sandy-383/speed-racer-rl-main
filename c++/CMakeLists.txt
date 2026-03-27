cmake_minimum_required(VERSION 3.12)
project(RacingDQN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyTorch (LibTorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find Raylib (installed via vcpkg)
find_package(raylib CONFIG REQUIRED)

# Create models directory at build time
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/models)

# Training executable (headless mode - no rendering during training)
add_executable(racing_trainer racing_trainer.cpp)
target_link_libraries(racing_trainer "${TORCH_LIBRARIES}" raylib)

# Replay executable (visual mode - watch trained agent)
add_executable(racing_replay racing_replay.cpp)
target_link_libraries(racing_replay "${TORCH_LIBRARIES}" raylib)

# Analysis tool (statistics viewer)
add_executable(analyze_training analyze_training.cpp)

# Copy assets to build directory at configure time
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets 
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Also copy assets to Release/Debug directories for MSVC (done at build time)
add_custom_command(TARGET racing_trainer POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_CURRENT_SOURCE_DIR}/assets
                   $<TARGET_FILE_DIR:racing_trainer>/assets)

add_custom_command(TARGET racing_replay POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                   ${CMAKE_CURRENT_SOURCE_DIR}/assets
                   $<TARGET_FILE_DIR:racing_replay>/assets)

# Windows-specific: Copy LibTorch DLLs to executable directories
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET racing_trainer
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:racing_trainer>)
    add_custom_command(TARGET racing_replay
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:racing_replay>)
endif()