cmake_minimum_required(VERSION 3.1)
project(PreassembledMass)


set(HEADER_FILES
    PreassembledMass.h
    PreassembledMass.inl
    config.h
)

set(SOURCE_FILES
    PreassembledMass.cpp
    initPlugin.cpp
)

find_package(SofaGeneral REQUIRED)
find_package(Compliant REQUIRED)
find_package(Flexible QUIET)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaEigen2Solver)
target_link_libraries(${PROJECT_NAME} Compliant)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")
set(PREASSEMBLEDMASS_COMPILER_FLAGS "-DSOFA_BUILD_PreassembledMass")

if(Flexible_FOUND)
    target_link_libraries(${PROJECT_NAME} Flexible)
    set(PREASSEMBLEDMASS_COMPILER_FLAGS "${PREASSEMBLEDMASS_COMPILER_FLAGS} -DSOFA_HAVE_FLEXIBLE")
else()
    message(STATUS "PreassembledMass: could not find Flexible")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS ${PREASSEMBLEDMASS_COMPILER_FLAGS})

install(TARGETS ${PROJECT_NAME}
        COMPONENT PreassembledMass_libraries
        EXPORT PreassembledMassTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
