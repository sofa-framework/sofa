cmake_minimum_required(VERSION 3.12)

sofa_add_module(Component ${PROJECT_NAME}.Component ON)
sofa_add_module(Common ${PROJECT_NAME}.Common ON)
sofa_add_module(Batch ${PROJECT_NAME}.Batch ON)

option(SOFA_GUI_ENABLE_QT "Compile the Qt GUI" ON)
if(SOFA_GUI_ENABLE_QT)
    sofa_add_module(Qt Sofa.GUI.Qt ON)
endif()

option(SOFA_GUI_ENABLE_HEADLESSRECORDER "Compile the headless recorder (Linux only)" OFF)
if(SOFA_GUI_ENABLE_HEADLESSRECORDER)
    sofa_add_module(HeadlessRecorder Sofa.GUI.HeadlessRecorder ON)
endif()
