cmake_minimum_required(VERSION 3.12)

add_subdirectory(Component)
add_subdirectory(Common)
add_subdirectory(Batch)

option(SOFA_GUI_ENABLE_QT "Compile the Qt GUI" ON)
if(SOFA_GUI_ENABLE_QT)
    add_subdirectory(Qt)
endif()

option(SOFA_GUI_ENABLE_HEADLESS_RECORDER "Compile the headless recorder (Linux only)" OFF)
if(SOFA_GUI_ENABLE_HEADLESS_RECORDER)
    add_subdirectory(HeadlessRecorder)
endif()
