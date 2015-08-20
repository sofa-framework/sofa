include(${SOFA_CMAKE_DIR}/preProject.cmake)

include_directories("${SOFA_EXTLIBS_DIR}/gtest/include")
link_directories("${SOFA_EXTLIBS_DIR}/gtest/lib")

if(WIN32)
    # MSVC2012 has some troubles with the way gtest use the STL, this preprocessor macro fix this issue
    AddCompilerDefinitions("_VARIADIC_MAX=10")
endif()

add_library(SofaGTestMain SofaGTestMain.cpp)
target_link_libraries(SofaGTestMain gtest SofaSimulationTree)
if(SOFA-LIB_SIMULATION_GRAPH_DAG)
    target_link_libraries(SofaGTestMain SofaSimulationGraph)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
