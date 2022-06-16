cmake_minimum_required(VERSION 3.12)

add_subdirectory(Component)
add_subdirectory(Common)
add_subdirectory(Batch)

option(SOFA_GUI_ENABLE_QT "Compile the Qt GUI" ON)
if(SOFA_GUI_ENABLE_QT)
    add_subdirectory(Qt)
endif()

option(SOFA_GUI_ENABLE_HEADLESSRECORDER "Compile the headless recorder (Linux only)" OFF)
if(SOFA_GUI_ENABLE_HEADLESSRECORDER)
    add_subdirectory(HeadlessRecorder)
endif()
