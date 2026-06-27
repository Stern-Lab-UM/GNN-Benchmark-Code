# Purpose: configure the spring embedding C executable with CMake.
# Inputs: CMake generator variables and source files in external/spring_embed/src.
# Outputs: a build system for the spring_embed executable.

cmake_minimum_required(VERSION 3.12)
project(dcg_spring_embed LANGUAGES CXX)

add_executable(spring_embed src/main.c)
set_source_files_properties(src/main.c PROPERTIES LANGUAGE CXX)
target_compile_features(spring_embed PRIVATE cxx_std_11)

if (MSVC)
    target_compile_options(spring_embed PRIVATE /O2)
else()
    target_compile_options(spring_embed PRIVATE -O2 -Wall -Wextra)
    target_link_libraries(spring_embed PRIVATE m)
endif()
