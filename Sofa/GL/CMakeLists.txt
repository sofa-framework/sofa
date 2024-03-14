cmake_minimum_required(VERSION 3.12)
project(Sofa.GL LANGUAGES CXX)

set(SOFAGLSRC_ROOT "src/sofa/gl")

sofa_find_package(OpenGL REQUIRED BOTH_SCOPES)
sofa_find_package(GLEW BOTH_SCOPES REQUIRED)
sofa_find_package(Sofa.Helper REQUIRED)
sofa_find_package(Sofa.DefaultType REQUIRED)

set(HEADER_FILES
    ${SOFAGLSRC_ROOT}/config.h.in
    ${SOFAGLSRC_ROOT}/initSofa.GL.h
    ${SOFAGLSRC_ROOT}/gl.h
    ${SOFAGLSRC_ROOT}/glu.h
    ${SOFAGLSRC_ROOT}/Capture.h
    ${SOFAGLSRC_ROOT}/Color.h
    ${SOFAGLSRC_ROOT}/RAII.h
    ${SOFAGLSRC_ROOT}/template.h
    ${SOFAGLSRC_ROOT}/Axis.h
    ${SOFAGLSRC_ROOT}/BasicShapes.h
    ${SOFAGLSRC_ROOT}/BasicShapesGL.h
    ${SOFAGLSRC_ROOT}/BasicShapesGL.inl
    ${SOFAGLSRC_ROOT}/Cylinder.h
    ${SOFAGLSRC_ROOT}/Texture.h
    ${SOFAGLSRC_ROOT}/VideoRecorderFFMPEG.h
    ${SOFAGLSRC_ROOT}/glText.h
    ${SOFAGLSRC_ROOT}/glText.inl
    ${SOFAGLSRC_ROOT}/TransformationGL.h
    ${SOFAGLSRC_ROOT}/DrawToolGL.h
)

set(SOURCE_FILES
    ${SOFAGLSRC_ROOT}/initSofa.GL.cpp
    ${SOFAGLSRC_ROOT}/Axis.cpp
    ${SOFAGLSRC_ROOT}/BasicShapesGL.cpp
    ${SOFAGLSRC_ROOT}/Cylinder.cpp
    ${SOFAGLSRC_ROOT}/glText.cpp
    ${SOFAGLSRC_ROOT}/Capture.cpp
    ${SOFAGLSRC_ROOT}/Texture.cpp
    ${SOFAGLSRC_ROOT}/VideoRecorderFFMPEG.cpp
    ${SOFAGLSRC_ROOT}/gl.cpp
    ${SOFAGLSRC_ROOT}/TransformationGL.cpp
    ${SOFAGLSRC_ROOT}/DrawToolGL.cpp
)

if(GLEW_FOUND)
    list(APPEND HEADER_FILES
        ${SOFAGLSRC_ROOT}/FrameBufferObject.h
        ${SOFAGLSRC_ROOT}/GLSLShader.h
    )
    list(APPEND SOURCE_FILES
        ${SOFAGLSRC_ROOT}/FrameBufferObject.cpp
        ${SOFAGLSRC_ROOT}/GLSLShader.cpp
    )
    list(APPEND SHADER_FILES
        ${SOFAGLSRC_ROOT}/shaders/generateSphere.cppglsl
    )
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Helper Sofa.DefaultType)

if(TARGET OpenGL::GL AND TARGET OpenGL::GLU) # Imported targets defined since CMake 3.8
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenGL::GL OpenGL::GLU)
else()
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OPENGL_LIBRARIES})
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${OPENGL_INCLUDE_DIR})
endif()
if(CMAKE_SYSTEM_NAME STREQUAL Linux AND SOFA_BUILD_RELEASE_PACKAGE AND OPENGL_GLU_FOUND)
    # Add GLU to Linux binaries
    sofa_install_libraries(PATHS ${OPENGL_glu_LIBRARY})
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC GLEW::GLEW)
if (SOFA_BUILD_RELEASE_PACKAGE OR CMAKE_SYSTEM_NAME STREQUAL Windows)
    sofa_install_libraries(TARGETS GLEW::GLEW)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

sofa_add_subdirectory(module Component ${PROJECT_NAME}.Component ON)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA.GL_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA.GL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
