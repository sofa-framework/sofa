include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

	qwt.h
    qwt_abstract_scale_draw.h
    qwt_interval_symbol.h
    qwt_clipper.h
    qwt_color_map.h
    qwt_compat.h
    qwt_column_symbol.h
    qwt_interval.h
    qwt_dyngrid_layout.h
    qwt_global.h
    qwt_math.h
    qwt_magnifier.h
    qwt_null_paintdevice.h
    qwt_painter.h
    qwt_panner.h
    qwt_picker.h
    qwt_picker_machine.h
    qwt_point_3d.h
    qwt_point_polar.h
    qwt_round_scale_draw.h
    qwt_scale_div.h
    qwt_scale_draw.h
    qwt_scale_engine.h
    qwt_scale_map.h
    qwt_spline.h
    qwt_symbol.h
    qwt_system_clock.h
    qwt_text_engine.h
    qwt_text_label.h
    qwt_text.h
	
	# QwtPlot
	qwt_curve_fitter.h
	qwt_event_pattern.h
	qwt_legend.h
	qwt_legend_item.h
	qwt_legend_itemmanager.h
	qwt_plot.h
	qwt_plot_renderer.h
	qwt_plot_curve.h
	qwt_plot_dict.h
	qwt_plot_directpainter.h
	qwt_plot_grid.h
	qwt_plot_histogram.h
	qwt_plot_item.h
	qwt_plot_intervalcurve.h
	qwt_plot_layout.h
	qwt_plot_marker.h
	qwt_plot_rasteritem.h
	qwt_plot_spectrogram.h
	qwt_plot_spectrocurve.h
	qwt_plot_scaleitem.h
	qwt_plot_seriesitem.h
	qwt_plot_canvas.h
	qwt_plot_panner.h
	qwt_plot_picker.h
	qwt_plot_zoomer.h
	qwt_plot_magnifier.h
	qwt_plot_rescaler.h
	qwt_raster_data.h
	qwt_matrix_raster_data.h
	qwt_sampling_thread.h
	qwt_series_data.h
	qwt_scale_widget.h 
	
	# QwtWidgets
	qwt_abstract_slider.h
	qwt_abstract_scale.h
	qwt_arrow_button.h
	qwt_analog_clock.h
	qwt_compass.h
	qwt_compass_rose.h
	qwt_counter.h
	qwt_dial.h
	qwt_dial_needle.h
	qwt_double_range.h
	qwt_knob.h
	qwt_slider.h
	qwt_thermo.h
	qwt_wheel.h
	)

set(SOURCE_FILES

	qwt_abstract_scale_draw.cpp
    qwt_interval_symbol.cpp
    qwt_clipper.cpp
    qwt_color_map.cpp
    qwt_column_symbol.cpp
    qwt_interval.cpp
    qwt_dyngrid_layout.cpp
    qwt_math.cpp
    qwt_magnifier.cpp
    qwt_panner.cpp
    qwt_null_paintdevice.cpp
    qwt_painter.cpp
    qwt_picker.cpp
    qwt_round_scale_draw.cpp
    qwt_scale_div.cpp
    qwt_scale_draw.cpp
    qwt_scale_map.cpp
    qwt_spline.cpp
    qwt_text_engine.cpp
    qwt_text_label.cpp
    qwt_text.cpp
    qwt_event_pattern.cpp
    qwt_picker_machine.cpp
    qwt_point_3d.cpp
    qwt_point_polar.cpp
    qwt_scale_engine.cpp
    qwt_symbol.cpp
    qwt_system_clock.cpp
	
	# QwtPlot
	qwt_curve_fitter.cpp
	qwt_legend.cpp
	qwt_legend_item.cpp
	qwt_plot.cpp
	qwt_plot_renderer.cpp
	qwt_plot_xml.cpp
	qwt_plot_axis.cpp
	qwt_plot_curve.cpp
	qwt_plot_dict.cpp
	qwt_plot_directpainter.cpp
	qwt_plot_grid.cpp
	qwt_plot_histogram.cpp
	qwt_plot_item.cpp
	qwt_plot_intervalcurve.cpp
	qwt_plot_spectrogram.cpp
	qwt_plot_spectrocurve.cpp
	qwt_plot_scaleitem.cpp
	qwt_plot_seriesitem.cpp
	qwt_plot_marker.cpp
	qwt_plot_layout.cpp
	qwt_plot_canvas.cpp
	qwt_plot_panner.cpp
	qwt_plot_rasteritem.cpp
	qwt_plot_picker.cpp
	qwt_plot_zoomer.cpp
	qwt_plot_magnifier.cpp
	qwt_plot_rescaler.cpp
	qwt_raster_data.cpp
	qwt_matrix_raster_data.cpp
	qwt_sampling_thread.cpp
	qwt_series_data.cpp
	qwt_scale_widget.cpp 
	
	# QwtWidgets
	qwt_abstract_slider.cpp
	qwt_abstract_scale.cpp
	qwt_arrow_button.cpp
	qwt_analog_clock.cpp
	qwt_compass.cpp
	qwt_compass_rose.cpp
	qwt_counter.cpp
	qwt_dial.cpp
	qwt_dial_needle.cpp
	qwt_double_range.cpp
	qwt_knob.cpp
	qwt_slider.cpp
	qwt_thermo.cpp
	qwt_wheel.cpp
	)

UseQt()
if(SOFA-EXTERNAL_PREFER_QT4)	
	SOFA_QT4_WRAP_CPP(MOC_FILES ${HEADER_FILES})
else()	
	SOFA_QT3_WRAP_CPP(MOC_FILES ${HEADER_FILES})
endif()

# make mocced header files path from absolute to relative
ToRelativePath(MOC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_FILES}")

set(MOC_FILES_GROUP "Generated")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("QT_DLL;QWT_DLL;QWT_MAKEDLL;QWT_NO_SVG")
if(UNIX AND NOT APPLE)
	AddLinkerDependencies(rt)
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

