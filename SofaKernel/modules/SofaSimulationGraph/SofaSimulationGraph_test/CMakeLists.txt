cmake_minimum_required(VERSION 3.1)

project (SofaSimulationGraph_test)

set (SOURCE_FILES
  MutationListener_test.cpp
  DAGNode_test.cpp
  )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)
add_test(Name ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
