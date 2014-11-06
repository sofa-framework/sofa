cmake_minimum_required(VERSION 2.8)
project(SofaMisc)

set(HEADER_FILES

    initMisc.h
    MeshTetraStuffing.h

    )

set(SOURCE_FILES

    initMisc.cpp
    MeshTetraStuffing.cpp

    )

if(SOFA_HAVE_SMP)
	list(APPEND HEADER_FILES "ParallelCGLinearSolver.h")
	list(APPEND HEADER_FILES "ParallelCGLinearSolver.inl")

	list(APPEND SOURCE_FILES "ParallelCGLinearSolver.cpp")
endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaMeshCollision SofaMiscTopology SofaTopologyMapping)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAMISC_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAMISC_DEPENDENCY_DIR}>")

set( SOFAMISC_COMPILER_FLAGS "-DSOFA_BUILD_MISC")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAMISC_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAMISC_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaMisc_libraries
        EXPORT    SofaMiscTargets
        RUNTIME DESTINATION "${SOFAMISC_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFAMISC_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFAMISC_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${SOFAMISC_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}"
)
