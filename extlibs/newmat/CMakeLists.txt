cmake_minimum_required(VERSION 2.8.12)
project(newmat)

if(CMAKE_VERSION GREATER 3.1)
    cmake_policy(SET CMP0054 NEW)   # CMake 3.2.3 warns if this is not set.
endif()

set(HEADER_FILES

    newmat/include.h
    newmat/newmat.h
    newmat/newmatrc.h
    newmat/boolean.h
    newmat/myexcept.h
    newmat/controlw.h
    newmat/newmatap.h
    newmat/newmatrm.h
    newmat/precisio.h)

set(SOURCE_FILES

    newmat/newmat1.cpp
    newmat/newmat2.cpp
    newmat/newmat3.cpp
    newmat/newmat4.cpp
    newmat/newmat5.cpp
    newmat/newmat6.cpp
    newmat/newmat7.cpp
    newmat/newmat8.cpp
    newmat/newmat9.cpp
    newmat/newmatex.cpp
    newmat/newmatrm.cpp
    newmat/bandmat.cpp
    newmat/submat.cpp
    newmat/myexcept.cpp
    newmat/cholesky.cpp
    newmat/evalue.cpp
    newmat/fft.cpp
    newmat/hholder.cpp
    newmat/jacobi.cpp
    newmat/newfft.cpp
    newmat/sort.cpp
    newmat/svd.cpp)

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "SofaExtlibs")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
target_compile_definitions(${PROJECT_NAME} PRIVATE "use_namespace")

# The code must be relocatable if we want to link a shared library against it
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
        OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE "-fPIC")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

include(CMakePackageConfigHelpers)

# NewmatConfig.cmake (build tree)
set(NEWMAT_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
configure_package_config_file(NewmatConfig.cmake.in
                              NewmatConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                              PATH_VARS NEWMAT_INCLUDE_DIR)

# NewmatConfig.cmake (installed)
set(NEWMAT_INCLUDE_DIR include)
configure_package_config_file(NewmatConfig.cmake.in
                              NewmatConfigInstall.cmake
                              INSTALL_DESTINATION lib/cmake/Newmat
                              PATH_VARS NEWMAT_INCLUDE_DIR)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/NewmatConfigInstall.cmake
        DESTINATION lib/cmake/Newmat
        RENAME NewmatConfig.cmake)

install(TARGETS ${PROJECT_NAME}
    EXPORT NewmatTargets
    RUNTIME DESTINATION bin COMPONENT Newmat_libraries
    LIBRARY DESTINATION lib COMPONENT Newmat_libraries
    ARCHIVE DESTINATION lib COMPONENT Newmat_libraries
    PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}" COMPONENT Newmat_headers)

# NewmatTargets.cmake (installed)
install(EXPORT NewmatTargets
        DESTINATION lib/cmake/Newmat)
