cmake_minimum_required(VERSION 3.22)

sofa_add_subdirectory(directory SofaGLFW SofaGLFW EXTERNAL GIT_REF master)
sofa_add_subdirectory(plugin Sofa.Qt Sofa.Qt EXTERNAL GIT_REF master OFF)

