cmake_minimum_required(VERSION 3.1)
project(QGLViewer VERSION 2.7.1)

sofa_find_package(OpenGL REQUIRED)
sofa_find_package(Qt5 COMPONENTS Core Gui Xml OpenGL Widgets QUIET)
if(NOT Qt5Core_FOUND)
    message(SEND_ERROR "QGLViewer: Qt5 is required and was not found.\n"
        "Add Qt5 directory to CMAKE_PREFIX_PATH or disable QGLViewer by setting SOFAGUI_QGLVIEWER to OFF.")
endif()
set(QT_TARGETS Qt5::Core Qt5::Gui Qt5::Xml Qt5::OpenGL Qt5::Widgets)

set(MOC_HEADER_FILES
    camera.h
    frame.h
    keyFrameInterpolator.h
    manipulatedCameraFrame.h
    manipulatedFrame.h
    qglviewer.h
    )
# These header files don't need to be processed by the moc.
set(HEADER_FILES
    VRender/AxisAlignedBox.h
    VRender/Exporter.h
    VRender/NVector3.h
    VRender/Optimizer.h
    VRender/ParserGL.h
    VRender/Primitive.h
    VRender/PrimitivePositioning.h
    VRender/SortMethod.h
    VRender/Types.h
    VRender/VRender.h
    VRender/Vector2.h
    VRender/Vector3.h
    VRender/gpc.h
    config.h
    constraint.h
    domUtils.h
    mouseGrabber.h
    quaternion.h
    vec.h
    )
set(SOURCE_FILES
    VRender/BSPSortMethod.cpp
    VRender/BackFaceCullingOptimizer.cpp
    VRender/EPSExporter.cpp
    VRender/Exporter.cpp
    VRender/FIGExporter.cpp
    VRender/NVector3.cpp
    VRender/ParserGL.cpp
    VRender/Primitive.cpp
    VRender/PrimitivePositioning.cpp
    VRender/TopologicalSortMethod.cpp
    VRender/VRender.cpp
    VRender/Vector2.cpp
    VRender/Vector3.cpp
    VRender/VisibilityOptimizer.cpp
    VRender/gpc.cpp
    camera.cpp
    constraint.cpp
    frame.cpp
    keyFrameInterpolator.cpp
    manipulatedCameraFrame.cpp
    manipulatedFrame.cpp
    mouseGrabber.cpp
    qglviewer.cpp
    quaternion.cpp
    saveSnapshot.cpp
    vec.cpp
    )
set(UI_FILES
    ImageInterface.ui
    VRenderInterface.ui
    )

qt5_wrap_cpp(MOC_FILES ${MOC_HEADER_FILES})
qt5_wrap_ui(FORM_FILES ${UI_FILES})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${FORM_FILES} ${SOURCE_FILES})

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "SofaExtlibs")
target_link_libraries(${PROJECT_NAME} PUBLIC ${QT_TARGETS})
if(TARGET OpenGL::GL AND TARGET OpenGL::GLU) # Imported targets defined since CMake 3.8
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenGL::GL OpenGL::GLU)
else()
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OPENGL_LIBRARIES})
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${OPENGL_INCLUDE_DIR})
endif()

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "$<INSTALL_INTERFACE:include/extlibs/${PROJECT_NAME}>")

if(WIN32)
    target_compile_options(${PROJECT_NAME} PRIVATE "-DCREATE_QGLVIEWER_DLL")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")

sofa_generate_package(NAME ${PROJECT_NAME} VERSION ${PROJECT_VERSION} TARGETS ${PROJECT_NAME} INCLUDE_INSTALL_DIR "extlibs/${PROJECT_NAME}/${PROJECT_NAME}")

# Qt deployment
if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    include(windeployqt)
    windeployqt(${PROJECT_NAME} ${CMAKE_BINARY_DIR}/bin ${CMAKE_INSTALL_PREFIX}/bin)
endif()
