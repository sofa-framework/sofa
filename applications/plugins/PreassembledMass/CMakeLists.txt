cmake_minimum_required(VERSION 2.8)
project(PreassembledMass)

set(PREASSEMBLEDMASS_INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(PREASSEMBLEDMASS_INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(PREASSEMBLEDMASS_INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")

set(HEADER_FILES
    PreassembledMass.h
    PreassembledMass.inl
    initPlugin.h
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
    RUNTIME DESTINATION "${PREASSEMBLEDMASS_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${PREASSEMBLEDMASS_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${PREASSEMBLEDMASS_INSTALL_LIB_DIR}")

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
