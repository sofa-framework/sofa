cmake_minimum_required(VERSION 2.8.12)
project(qwt)


set(QWT_MAJOR_VERSION 6)
set(QWT_MINOR_VERSION 0)
set(QWT_PATCH_VERSION 1)
set(QWT_VERSION ${QWT_MAJOR_VERSION}.${QWT_MINOR_VERSION}.${QWT_PATCH_VERSION})

set(QWT_BUILD_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(QWT_BUILD_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(QWT_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(QWT_BUILD_CMAKE_DIR ${CMAKE_BINARY_DIR}/Qwt/)

# Offer the user the choice of overriding the installation directories
set(QWT_INSTALL_LIB_DIR     lib                 CACHE PATH "Installation directory for libraries")
set(QWT_INSTALL_BIN_DIR     bin                 CACHE PATH "Installation directory for executables")
set(QWT_INSTALL_INCLUDE_DIR include             CACHE PATH "Installation directory for header files")
set(QWT_INSTALL_CMAKE_DIR   lib/cmake/Qwt   CACHE PATH "Installation directory for CMake files")

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


find_package(Qt4)

include(${QT_USE_FILE})

# Declare moccable files, and get a list of the files to be generated
QT4_WRAP_CPP(MOC_FILES ${Q_OBJECT_HEADER_FILES})

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")
target_link_libraries(${PROJECT_NAME} PRIVATE ${QT_LIBRARIES})

if(UNIX AND NOT APPLE)
    target_link_libraries(${PROJECT_NAME} PRIVATE rt)
endif()

target_compile_options(${PROJECT_NAME} PUBLIC "-DQT_DLL" "-DQWT_DLL" "-DQWT_MAKEDLL" "-DQWT_NO_SVG" ${QT_DEFINITIONS} )

set_target_properties( ${PROJECT_NAME}  PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME}  PROPERTIES VERSION ${QWT_VERSION} )
set_target_properties( ${PROJECT_NAME}  PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )
		  
install(TARGETS ${PROJECT_NAME}
        EXPORT  QwtTargets
        RUNTIME DESTINATION "${QWT_INSTALL_BIN_DIR}" 
        LIBRARY DESTINATION "${QWT_INSTALL_LIB_DIR}" 
        ARCHIVE DESTINATION "${QWT_INSTALL_LIB_DIR}" 
        PUBLIC_HEADER DESTINATION "${QWT_INSTALL_INCLUDE_DIR}" )

include(CMakePackageConfigHelpers)

# QwtConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${QWT_BUILD_CMAKE_DIR}/QwtConfigVersion.cmake
                                 VERSION ${QWT_VERSION}
                                 COMPATIBILITY AnyNewerVersion)
                                
install(FILES ${QWT_BUILD_CMAKE_DIR}/QwtConfigVersion.cmake
        DESTINATION ${QWT_INSTALL_CMAKE_DIR})


# QwtConfig.cmake (build tree)
set(QWT_INCLUDE_DIR "${QWT_BUILD_INCLUDE_DIR}")
configure_package_config_file(QwtConfig.cmake.in 
                              ${QWT_BUILD_CMAKE_DIR}/QwtConfig.cmake
                              INSTALL_DESTINATION ${QWT_BUILD_CMAKE_DIR}
                              PATH_VARS QWT_INCLUDE_DIR )

# QwtConfig.cmake (installed)
set(QWT_INCLUDE_DIR "${QWT_INSTALL_INCLUDE_DIR}")
configure_package_config_file(QwtConfig.cmake.in 
                              ${QWT_BUILD_CMAKE_DIR}/QwtConfigInstall.cmake
                              INSTALL_DESTINATION ${QWT_INSTALL_CMAKE_DIR}
                              PATH_VARS QWT_INCLUDE_DIR )

install(FILES ${QWT_BUILD_CMAKE_DIR}/QwtConfigInstall.cmake
        DESTINATION ${QWT_INSTALL_CMAKE_DIR}
        RENAME QwtConfig.cmake)
	   
# QwtTargets.cmake (installed)
install(EXPORT QwtTargets
        DESTINATION ${QWT_INSTALL_CMAKE_DIR})
