cmake_minimum_required(VERSION 3.22)
project(SofaScenes)

file(GLOB_RECURSE EXAMPLES_FILES ../*.scn ../*.pscn ../*.pyscn)

add_library(${PROJECT_NAME} INTERFACE ${EXAMPLES_FILES})