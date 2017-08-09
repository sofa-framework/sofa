Cmake_minimum_required(VERSION 3.1)
project(Communication)

set(HEADER_FILES
    config.h
    components/communicationBinder.h
    components/serverCommunication.h
    components/serverCommunication.inl
)

set(SOURCE_FILES
    initCommunication.cpp
    components/communicationBinder.cpp
    components/serverCommunication.cpp
)

set(EXTRA_FILES
    README.md
)

find_package(SofaFramework REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} -loscpack -lpthread ${}SofaCore SofaSimulationGraph)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_Communication")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

install(TARGETS ${PROJECT_NAME}
        COMPONENT Communication_libraries
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
