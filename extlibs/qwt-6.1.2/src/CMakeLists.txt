cmake_minimum_required(VERSION 2.8.12)
project(qwt)

 
set(QWT_VERSION 6.1.2)

file(GLOB HEADER_FILES qwt_*.h)

file(GLOB SOURCE_FILES qwt_*.cpp)

# Header files containing Q_OBJECT declarations
set(Q_OBJECT_HEADER_FILES
	qwt_abstract_legend.h
	qwt_abstract_scale.h
    qwt_abstract_slider.h
    qwt_analog_clock.h
    qwt_compass.h
    qwt_counter.h
    qwt_dial.h
    qwt_dyngrid_layout.h
    qwt_knob.h
    qwt_legend.h
    qwt_legend_label.h
    qwt_magnifier.h
    qwt_panner.h
    qwt_picker.h
    qwt_plot_canvas.h
    qwt_plot.h
    qwt_plot_glcanvas.h
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

set(HEADER_FILES
	./qwt.h
	./qwt_abstract_legend.h
	./qwt_abstract_scale.h
	./qwt_abstract_scale_draw.h
	./qwt_abstract_slider.h
	./qwt_analog_clock.h
	./qwt_arrow_button.h
	./qwt_clipper.h
	./qwt_color_map.h
	./qwt_column_symbol.h
	./qwt_compass.h
	./qwt_compass_rose.h
	./qwt_compat.h
	./qwt_counter.h
	./qwt_curve_fitter.h
	./qwt_date.h
	./qwt_date_scale_draw.h
	./qwt_date_scale_engine.h
	./qwt_dial.h
	./qwt_dial_needle.h
	./qwt_dyngrid_layout.h
	./qwt_event_pattern.h
	./qwt_global.h
	./qwt_graphic.h
	./qwt_interval.h
	./qwt_interval_symbol.h
	./qwt_knob.h
	./qwt_legend.h
	./qwt_legend_data.h
	./qwt_legend_label.h
	./qwt_magnifier.h
	./qwt_math.h
	./qwt_matrix_raster_data.h
	./qwt_null_paintdevice.h
	./qwt_painter.h
	./qwt_painter_command.h
	./qwt_panner.h
	./qwt_picker.h
	./qwt_picker_machine.h
	./qwt_pixel_matrix.h
	./qwt_plot.h
	./qwt_plot_abstract_barchart.h
	./qwt_plot_barchart.h
	./qwt_plot_canvas.h
	./qwt_plot_curve.h
	./qwt_plot_dict.h
	./qwt_plot_directpainter.h
	./qwt_plot_glcanvas.h
	./qwt_plot_grid.h
	./qwt_plot_histogram.h
	./qwt_plot_intervalcurve.h
	./qwt_plot_item.h
	./qwt_plot_layout.h
	./qwt_plot_legenditem.h
	./qwt_plot_magnifier.h
	./qwt_plot_marker.h
	./qwt_plot_multi_barchart.h
	./qwt_plot_panner.h
	./qwt_plot_picker.h
	./qwt_plot_rasteritem.h
	./qwt_plot_renderer.h
	./qwt_plot_rescaler.h
	./qwt_plot_scaleitem.h
	./qwt_plot_seriesitem.h
	./qwt_plot_shapeitem.h
	./qwt_plot_spectrocurve.h
	./qwt_plot_spectrogram.h
	#./qwt_plot_svgitem.h
	./qwt_plot_textlabel.h
	./qwt_plot_tradingcurve.h
	./qwt_plot_zoneitem.h
	./qwt_plot_zoomer.h
	./qwt_point_3d.h
	./qwt_point_data.h
	./qwt_point_mapper.h
	./qwt_point_polar.h
	./qwt_raster_data.h
	./qwt_round_scale_draw.h
	./qwt_samples.h
	./qwt_sampling_thread.h
	./qwt_scale_div.h
	./qwt_scale_draw.h
	./qwt_scale_engine.h
	./qwt_scale_map.h
	./qwt_scale_widget.h
	./qwt_series_data.h
	./qwt_series_store.h
	./qwt_slider.h
	./qwt_spline.h
	./qwt_symbol.h
	./qwt_system_clock.h
	./qwt_text.h
	./qwt_text_engine.h
	./qwt_text_label.h
	./qwt_thermo.h
	./qwt_transform.h
	./qwt_wheel.h
	./qwt_widget_overlay.h
)
# Header files containing Q_OBJECT declarations
set(SOURCE_FILES
    qwt_abstract_legend.cpp
	qwt_abstract_scale.cpp
	qwt_abstract_scale_draw.cpp
	qwt_abstract_slider.cpp
	qwt_analog_clock.cpp
	qwt_arrow_button.cpp
	qwt_clipper.cpp
	qwt_color_map.cpp
	qwt_column_symbol.cpp
	qwt_compass.cpp
	qwt_compass_rose.cpp
	qwt_counter.cpp
	qwt_curve_fitter.cpp
	qwt_date.cpp
	qwt_date_scale_draw.cpp
	qwt_date_scale_engine.cpp
	qwt_dial.cpp
	qwt_dial_needle.cpp
	qwt_dyngrid_layout.cpp
	qwt_event_pattern.cpp
	qwt_graphic.cpp
	qwt_interval.cpp
	qwt_interval_symbol.cpp
	qwt_knob.cpp
	qwt_legend.cpp
	qwt_legend_data.cpp
	qwt_legend_label.cpp
	qwt_magnifier.cpp
	qwt_math.cpp
	qwt_matrix_raster_data.cpp
	qwt_null_paintdevice.cpp
	qwt_painter.cpp
	qwt_painter_command.cpp
	qwt_panner.cpp
	qwt_picker.cpp
	qwt_picker_machine.cpp
	qwt_pixel_matrix.cpp
	qwt_plot.cpp
	qwt_plot_abstract_barchart.cpp
	qwt_plot_axis.cpp
	qwt_plot_barchart.cpp
	qwt_plot_canvas.cpp
	qwt_plot_curve.cpp
	qwt_plot_dict.cpp
	qwt_plot_directpainter.cpp
	qwt_plot_glcanvas.cpp
	qwt_plot_grid.cpp
	qwt_plot_histogram.cpp
	qwt_plot_intervalcurve.cpp
	qwt_plot_item.cpp
	qwt_plot_layout.cpp
	qwt_plot_legenditem.cpp
	qwt_plot_magnifier.cpp
	qwt_plot_marker.cpp
	qwt_plot_multi_barchart.cpp
	qwt_plot_panner.cpp
	qwt_plot_picker.cpp
	qwt_plot_rasteritem.cpp
	qwt_plot_renderer.cpp
	qwt_plot_rescaler.cpp
	qwt_plot_scaleitem.cpp
	qwt_plot_seriesitem.cpp
	qwt_plot_shapeitem.cpp
	qwt_plot_spectrocurve.cpp
	qwt_plot_spectrogram.cpp
	#qwt_plot_svgitem.cpp
	qwt_plot_textlabel.cpp
	qwt_plot_tradingcurve.cpp
	qwt_plot_xml.cpp
	qwt_plot_zoneitem.cpp
	qwt_plot_zoomer.cpp
	qwt_point_3d.cpp
	qwt_point_data.cpp
	qwt_point_mapper.cpp
	qwt_point_polar.cpp
	qwt_raster_data.cpp
	qwt_round_scale_draw.cpp
	qwt_sampling_thread.cpp
	qwt_scale_div.cpp
	qwt_scale_draw.cpp
	qwt_scale_engine.cpp
	qwt_scale_map.cpp
	qwt_scale_widget.cpp
	qwt_series_data.cpp
	qwt_slider.cpp
	qwt_spline.cpp
	qwt_symbol.cpp
	qwt_system_clock.cpp
	qwt_text.cpp
	qwt_text_engine.cpp
	qwt_text_label.cpp
	qwt_thermo.cpp
	qwt_transform.cpp
	qwt_wheel.cpp
	qwt_widget_overlay.cpp
    )

find_package(Qt5 COMPONENTS Core QUIET)

if(Qt5Core_FOUND)
	find_package(Qt5 COMPONENTS Concurrent Gui Xml OpenGL PrintSupport Quick Qml REQUIRED)

    include_directories(${Qt5Core_INCLUDE_DIRS})
    include_directories(${Qt5Widgets_INCLUDE_DIRS})
    include_directories(${Qt5OpenGL_INCLUDE_DIRS})
    include_directories(${Qt5Concurrent_INCLUDE_DIRS})
    include_directories(${Qt5PrintSupport_INCLUDE_DIRS})
    include_directories(${Qt5Qml_INCLUDE_DIRS})
    include_directories(${Qt5Quick_INCLUDE_DIRS})
    include_directories(${Qt5Xml_INCLUDE_DIRS})

    set(QT_LIBRARIES ${Qt5Core_LIBRARIES})
    set(QT_LIBRARIES ${QT_LIBRARIES} ${Qt5Widgets_LIBRARIES})
    set(QT_LIBRARIES ${QT_LIBRARIES} ${Qt5OpenGL_LIBRARIES})
    set(QT_LIBRARIES ${QT_LIBRARIES} ${Qt5Concurrent_LIBRARIES})
    set(QT_LIBRARIES ${QT_LIBRARIES} ${Qt5Xml_LIBRARIES})
    set(QT_LIBRARIES ${QT_LIBRARIES} ${Qt5PrintSupport_LIBRARIES})

    qt5_wrap_cpp(MOC_FILES ${Q_OBJECT_HEADER_FILES})
else()

	find_package(Qt4 COMPONENTS qtcore qtgui qtopengl qtxml qtnetwork REQUIRED)

	include(${QT_USE_FILE})
	include_directories(${QT_INCLUDE_DIR})

	qt4_wrap_cpp(MOC_FILES ${Q_OBJECT_HEADER_FILES})
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
target_link_libraries(${PROJECT_NAME} PRIVATE ${QT_LIBRARIES})
if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE rt)
endif()
target_compile_options(${PROJECT_NAME} PRIVATE "-DQT_DLL" "-DQWT_DLL" "-DQWT_MAKEDLL" "-DQWT_NO_SVG" ${QT_DEFINITIONS})
target_compile_options(${PROJECT_NAME} INTERFACE "-DQT_DLL" "-DQWT_DLL"  "-DQWT_NO_SVG" ${QT_DEFINITIONS})
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "SofaExtlibs")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${QWT_VERSION})

include(../../../SofaFramework/SofaMacros.cmake)
sofa_create_package(Qwt ${QWT_VERSION} qwt .)
