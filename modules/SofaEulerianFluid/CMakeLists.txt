cmake_minimum_required(VERSION 2.8.11)
project(SofaEulerianFluid)

set(HEADER_FILES

    initEulerianFluid.h
    Fluid2D.h
    Fluid3D.h
    Grid2D.h
    Grid3D.h

    )

set(SOURCE_FILES

    initEulerianFluid.cpp
    Fluid2D.cpp
    Fluid3D.cpp
    Grid2D.cpp
    Grid3D.cpp

    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries( ${PROJECT_NAME} SofaCore ) 
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAADVANCED_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAADVANCED_DEPENDENCY_DIR}>")

set( COMPILER_FLAGS "-DSOFA_BUILD_EULERIAN_FLUID")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAADVANCED_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        EXPORT  SofaAdvancedTargets
        RUNTIME DESTINATION "${SOFAADVANCED_INSTALL_BIN_DIR}" COMPONENT SofaAdvanced_libraries
        LIBRARY DESTINATION "${SOFAADVANCED_INSTALL_LIB_DIR}" COMPONENT SofaAdvanced_libraries
        ARCHIVE DESTINATION "${SOFAADVANCED_INSTALL_LIB_DIR}" COMPONENT SofaAdvanced_libraries
        PUBLIC_HEADER DESTINATION "${SOFAADVANCED_INSTALL_INCLUDE_DIR}/SofaEulerianFluid" COMPONENT SofaAdvanced_headers
        )

