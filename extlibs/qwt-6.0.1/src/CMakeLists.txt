cmake_minimum_required(VERSION 2.8.12)
project(qwt)


set(QWT_VERSION 6.0.1)

file(GLOB HEADER_FILES qwt_*.h)

file(GLOB SOURCE_FILES qwt_*.cpp)

# Header files containing Q_OBJECT declarations
set(Q_OBJECT_HEADER_FILES
    qwt_abstract_slider.h
    qwt_analog_clock.h
    qwt_compass.h
    qwt_counter.h
    qwt_dial.h
    qwt_dyngrid_layout.h
    qwt_knob.h
    qwt_legend.h
    qwt_legend_item.h
    qwt_magnifier.h
    qwt_panner.h
    qwt_picker.h
    qwt_plot.h
    qwt_plot_canvas.h
    qwt_plot_magnifier.h
    qwt_plot_panner.h
    qwt_plot_picker.h
    qwt_plot_renderer.h
    qwt_plot_zoomer.h
    qwt_sampling_thread.h
    qwt_scale_widget.h
    qwt_slider.h
    qwt_text_label.h
    qwt_thermo.h
    qwt_wheel.h
)


find_package(Qt4)
include(${QT_USE_FILE})

qt4_wrap_cpp(MOC_FILES ${Q_OBJECT_HEADER_FILES})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
target_link_libraries(${PROJECT_NAME} PRIVATE ${QT_LIBRARIES})
if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE rt)
endif()
target_compile_options(${PROJECT_NAME} PUBLIC "-DQT_DLL" "-DQWT_DLL" "-DQWT_MAKEDLL" "-DQWT_NO_SVG" ${QT_DEFINITIONS})
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "SofaExtlibs")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${QWT_VERSION})

include(../../../SofaFramework/SofaPackageConfigHelpers.cmake)
sofa_create_package(Qwt 6.0.1 qwt)
