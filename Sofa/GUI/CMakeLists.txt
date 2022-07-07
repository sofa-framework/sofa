cmake_minimum_required(VERSION 3.12)

sofa_add_module(Component Sofa.GUI.Component)
sofa_add_module(Common Sofa.GUI.Common)
sofa_add_module(Batch Sofa.GUI.Batch)

option(SOFA_GUI_ENABLE_QT "Compile the Qt GUI" ON)
if(SOFA_GUI_ENABLE_QT)
    sofa_add_module(Qt Sofa.GUI.Qt)
endif()

option(SOFA_GUI_ENABLE_HEADLESSRECORDER "Compile the headless recorder (Linux only)" OFF)
if(SOFA_GUI_ENABLE_HEADLESSRECORDER)
    sofa_add_module(HeadlessRecorder Sofa.GUI.HeadlessRecorder)
endif()
