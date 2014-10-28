include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
		Window.cpp
		Viewer.cpp
		Camera.cpp
        Scene.cpp
		
		Main.cpp
)

set(HEADER_FILES
		Window.h
		Viewer.h
		Camera.h
        Scene.h
)

set(QML_FILES
		data/qml/Main.qml
		data/qml/FileBrowser.qml
		data/qml/Footer.qml
		data/qml/Header.qml
		data/qml/Scene.qml
		data/qml/Viewer.qml
		data/qml/SimulationControl.qml
		data/qml/FPSDisplay.qml
)

set(QRC_FILES
		resource.qrc
)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# on Window, Qt packages need the glu32 lib full path, but they do not find it without help
if(WIN32)
	if(CMAKE_CL_64)
		set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "C:/Program Files (x86)/Windows Kits/8.0/Lib/win8/um/x64")
	else()
		set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "C:/Program Files (x86)/Windows Kits/8.0/Lib/win8/um/x86")
	endif()
endif()

find_package(Qt5 COMPONENTS Core Gui Widgets Quick Qml REQUIRED PATHS "${SOFA-EXTERNAL_QT5_PATH}")
# make sure to use QT > 5.0

qt5_wrap_cpp(MOC_FILES ${HEADER_FILES}) # do not use "set(CMAKE_AUTOMOC ON)" since all the mocced files will be compiled by a single compilation unit leading to a "out of heap space" issue on MSVC
qt5_add_resources(RESOURCE_FILES ${QRC_FILES})

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${QRC_FILES} ${RESOURCE_FILES} ${QML_FILES})

AddLinkerDependencies(SofaSimulationGraph SofaComponentMain Qt5::Core Qt5::Gui Qt5::Widgets Qt5::Qml Qt5::Quick)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

