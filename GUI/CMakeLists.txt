cmake_minimum_required(VERSION 3.12)

add_subdirectory(Common)
add_subdirectory(Batch)

option(SOFAGUI_QT "Compile the Qt GUI" ON)
if(SOFAGUI_QT)
    add_subdirectory(Qt)
endif()

option(SOFAGUI_HEADLESS_RECORDER "Compile the headless recorder (Linux only)" OFF)
if(SOFAGUI_HEADLESS_RECORDER)
    add_subdirectory(HeadlessRecorder)
endif()
