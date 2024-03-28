cmake_minimum_required(VERSION 3.22)
project(SofaShaders)

file(GLOB_RECURSE SHADER_FILES *.vert *.frag *.geo)

add_library(${PROJECT_NAME} INTERFACE ${SHADER_FILES})