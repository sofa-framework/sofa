include(${SOFA_CMAKE_DIR}/preProject.cmake)


add_executable(${PROJECT_NAME} Main.cpp)

AddLinkerDependencies( SofaComponentMain )

include(${SOFA_CMAKE_DIR}/postProject.cmake)

