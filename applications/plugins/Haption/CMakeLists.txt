cmake_minimum_required(VERSION 3.1)
project(Haption)

find_package(SofaHaptics REQUIRED)
find_package(SofaOpenglVisual REQUIRED)

set(HEADER_FILES
    HaptionDriver.h
)

set(SOURCE_FILES
    initHaption.cpp 
    HaptionDriver.cpp
    initHaption.cpp
)

set(README_FILES
    PluginHaption.txt
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaHaptics SofaOpenglVisual)

install(TARGETS ${PROJECT_NAME}
        COMPONENT Haption_libraries
        EXPORT HaptionTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY exemple/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
