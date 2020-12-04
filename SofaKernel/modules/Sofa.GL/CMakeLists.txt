cmake_minimum_required(VERSION 3.12)
project(Sofa.GL LANGUAGES CXX)

set(SOFAGLSRC_ROOT "src/sofa/gl")

sofa_find_package(OpenGL REQUIRED BOTH_SCOPES)
sofa_find_package(GLEW BOTH_SCOPES)

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
    ${SOFAGLSRC_ROOT}/Trackball.h
    ${SOFAGLSRC_ROOT}/VideoRecorderFFMPEG.h
    ${SOFAGLSRC_ROOT}/glText.h
    ${SOFAGLSRC_ROOT}/glText.inl
    ${SOFAGLSRC_ROOT}/Trackball.h
    ${SOFAGLSRC_ROOT}/TransformationGL.h
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
    ${SOFAGLSRC_ROOT}/Color.cpp
    ${SOFAGLSRC_ROOT}/gl.cpp
    ${SOFAGLSRC_ROOT}/Trackball.cpp
    ${SOFAGLSRC_ROOT}/TransformationGL.cpp
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
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaDefaultType)

target_compile_definitions(${PROJECT_NAME} PRIVATE SOFA_BUILD_SOFA_GL) # To remove once sofa_add_targets_to_package remove the dot in the generated definition 

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

if(GLEW_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC GLEW::GLEW)
    if (SOFA_BUILD_RELEASE_PACKAGE OR CMAKE_SYSTEM_NAME STREQUAL Windows)
        sofa_install_libraries(TARGETS GLEW::GLEW)
    endif()
else()
    message("OpenGL advanced functions (e.g shaders, FBO) are disabled.")
endif()

sofa_add_targets_to_package(
    PACKAGE_NAME SofaFramework
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaFramework"
    )
