cmake_minimum_required(VERSION 2.8)
project(SofaMiscTopology)

set(HEADER_FILES

    initMiscTopology.h
    TopologicalChangeProcessor.h
    )

set(SOURCE_FILES

    initMiscTopology.cpp
    TopologicalChangeProcessor.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAMISC_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAMISC_DEPENDENCY_DIR}>")

set( SOFAMISCTOPOLOGY_COMPILER_FLAGS "-DSOFA_BUILD_MISC_TOPOLOGY -DPOINT_DATA_VECTOR_ACCESS")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAMISC_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAMISCTOPOLOGY_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        EXPORT    SofaMiscTargets
        RUNTIME DESTINATION "${SOFAMISC_INSTALL_BIN_DIR}" COMPONENT SofaMisc_libraries
        LIBRARY DESTINATION "${SOFAMISC_INSTALL_LIB_DIR}" COMPONENT SofaMisc_libraries
        ARCHIVE DESTINATION "${SOFAMISC_INSTALL_LIB_DIR}" COMPONENT SofaMisc_libraries
        PUBLIC_HEADER DESTINATION "${SOFAMISC_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}" COMPONENT SofaMisc_headers
)