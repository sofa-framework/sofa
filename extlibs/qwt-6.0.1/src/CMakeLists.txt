include(${SOFA_CMAKE_DIR}/preProject.cmake)

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
    qwt_plot_canvas.h
    qwt_plot.h
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
    qwt_wheel.h)


UseQt()
# Declare moccable files, and get a list of the files to be generated
QT4_WRAP_CPP(MOC_FILES ${Q_OBJECT_HEADER_FILES})
# Convert absolute paths of moc-generated files to relative paths
ToRelativePath(MOC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_FILES}")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("QT_DLL;QWT_DLL;QWT_MAKEDLL;QWT_NO_SVG;QT_NO_QT_INCLUDE_WARN")
if(UNIX AND NOT APPLE)
    AddLinkerDependencies(rt)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)
