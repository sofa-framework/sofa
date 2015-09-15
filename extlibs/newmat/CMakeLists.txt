cmake_minimum_required(VERSION 2.8.12)
project(newmat)

set(HEADER_FILES
    newmat/boolean.h
    newmat/controlw.h
    newmat/include.h
    newmat/myexcept.h
    newmat/newmat.h
    newmat/newmatap.h
    newmat/newmatrc.h
    newmat/newmatrm.h
    newmat/precisio.h
)

set(SOURCE_FILES
    newmat/bandmat.cpp
    newmat/cholesky.cpp
    newmat/evalue.cpp
    newmat/fft.cpp
    newmat/hholder.cpp
    newmat/jacobi.cpp
    newmat/myexcept.cpp
    newmat/newfft.cpp
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
    newmat/sort.cpp
    newmat/submat.cpp
    newmat/svd.cpp
)

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "SofaExtlibs")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
target_compile_definitions(${PROJECT_NAME} PRIVATE "use_namespace")

# The code must be relocatable if we want to link a shared library against it
if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xGNU" OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xClang")
    target_compile_options(${PROJECT_NAME} PRIVATE "-fPIC")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")


install(TARGETS ${PROJECT_NAME}
        EXPORT NewmatTargets
        RUNTIME DESTINATION bin COMPONENT Newmat_libraries
        LIBRARY DESTINATION lib COMPONENT Newmat_libraries
        ARCHIVE DESTINATION lib COMPONENT Newmat_libraries
        PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}" COMPONENT Newmat_headers)


include(CMakePackageConfigHelpers)

### NewmatConfig.cmake
## Build tree
set(NEWMAT_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
configure_package_config_file(NewmatConfig.cmake.in
                              NewmatConfig.cmake
                              INSTALL_DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
                              PATH_VARS NEWMAT_INCLUDE_DIR)
## Install tree
set(NEWMAT_INCLUDE_DIR include)
configure_package_config_file(NewmatConfig.cmake.in
                              InstalledNewmatConfig.cmake
                              INSTALL_DESTINATION lib/cmake/Newmat
                              PATH_VARS NEWMAT_INCLUDE_DIR)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/InstalledNewmatConfig.cmake
        DESTINATION lib/cmake/Newmat
        RENAME NewmatConfig.cmake)

## NewmatTargets.cmake
install(EXPORT NewmatTargets DESTINATION lib/cmake/Newmat)
