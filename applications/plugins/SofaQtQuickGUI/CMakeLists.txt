include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
		SofaQtQuickGUI.cpp
                Tools.cpp
		Viewer.cpp
		Camera.cpp
                Scene.cpp
                SceneListModel.cpp
                DataListModel.cpp
		PythonInteractor.cpp
		PickingInteractor.cpp
)

set(HEADER_FILES
		SofaQtQuickGUI.h
                Tools.h
		Viewer.h
		Camera.h
                Scene.h
                SceneListModel.h
                DataListModel.h
		PythonInteractor.h
		PickingInteractor.h
)

set(QML_FILES
                data/qml/SofaDataTypes/qmldir
                data/qml/SofaDataTypes/DataType_boolean.qml
                data/qml/SofaDataTypes/DataType_0to1RatioWidget.qml
                data/qml/SofaDataTypes/DataType_array.qml
                data/qml/SofaDataTypes/DataType_boolean.qml
                data/qml/SofaDataTypes/DataType_BoundingBox.qml
                data/qml/SofaDataTypes/DataType_Material.qml
                data/qml/SofaDataTypes/DataType_OptionsGroup.qml
                data/qml/SofaDataTypes/DataType_graph_linear.qml
                data/qml/SofaDataTypes/DataType_graph.qml
                data/qml/SofaDataTypes/DataType_number.qml
                data/qml/SofaDataTypes/DataType_poissonRatio.qml
                data/qml/SofaDataTypes/DataType_string.qml
                data/qml/SofaDataTypes/DataType_widget_displayFlags.qml
                data/qml/SofaCommon/SofaCommonScript.js
                data/qml/SofaCommon/SofaApplicationScript.js
                data/qml/SofaCommon/SofaToolsScript.js
                data/qml/SofaCommon/SofaSettingsScript.js
                data/qml/SofaBasics/qmldir
                data/qml/SofaBasics/Application.qml
                data/qml/SofaBasics/RecentSettings.qml
                data/qml/SofaBasics/UISettings.qml
                data/qml/SofaBasics/Tools.qml
                data/qml/SofaBasics/ToolTip.qml
                data/qml/SofaBasics/DefaultStatusBar.qml
                data/qml/SofaBasics/DefaultToolBar.qml
		data/qml/SofaBasics/DefaultMenuBar.qml
		data/qml/SofaBasics/Scene.qml
                data/qml/SofaBasics/SceneListModel.qml
                data/qml/SofaBasics/Data.qml
                data/qml/SofaBasics/DataListModel.qml
		data/qml/SofaBasics/PythonInteractor.qml
                data/qml/SofaBasics/UserInteractor.qml
		data/qml/SofaBasics/Camera.qml
		data/qml/SofaBasics/FPSDisplay.qml
                data/qml/SofaBasics/ContentItem.qml
                data/qml/SofaBasics/CollapsibleGroupBox.qml
		data/qml/SofaBasics/Vector3DSpinBox.qml
		data/qml/SofaBasics/DynamicSplitView.qml
                data/qml/SofaBasics/DynamicContent.qml
                data/qml/SofaInteractor/qmldir
                data/qml/SofaInteractor/UserInteractor_Selection.qml
		data/qml/SofaTools/qmldir
		data/qml/SofaTools/SimulationControl.qml
                data/qml/SofaTools/SceneGraphView.qml
                data/qml/SofaTools/DataListView.qml
		data/qml/SofaTools/SceneGUILoader.qml
		data/qml/SofaWidgets/qmldir
		data/qml/SofaWidgets/Viewer.qml
                data/qml/SofaWidgets/DataView.qml
		data/qml/SofaWidgets/ImageViewer.qml
		data/qml/SofaWidgets/ToolPanel.qml
)

set(CONFIG_FILES
		data/config/default.ini
)

set(QRC_FILES
                data/resources.qrc
                data/qml/qml.qrc
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

if(NOT MSVC)
    add_definitions(-std=gnu++11)
endif()

find_package(Qt5 COMPONENTS Core Gui Widgets Quick Qml REQUIRED PATHS "${SOFA-EXTERNAL_QT5_PATH}")
# make sure to use QT > 5.0

include_directories(${Qt5Core_INCLUDE_DIRS})
include_directories(${Qt5Gui_INCLUDE_DIRS})
include_directories(${Qt5Widgets_INCLUDE_DIRS})
include_directories(${Qt5Qml_INCLUDE_DIRS})
include_directories(${Qt5Quick_INCLUDE_DIRS})

add_definitions(${Qt5Core_DEFINITIONS})
add_definitions(${Qt5Gui_DEFINITIONS})
add_definitions(${Qt5Widgets_DEFINITIONS})
add_definitions(${Qt5Qml_DEFINITIONS})
add_definitions(${Qt5Quick_DEFINITIONS})

if (WIN32)
	if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
		add_definitions(-DQT_NO_DEBUG)
	endif()
endif()
	
qt5_wrap_cpp(MOC_FILES ${HEADER_FILES}) # do not use "set(CMAKE_AUTOMOC ON)" since all the mocced files will be compiled by a single compilation unit leading to a "out of heap space" issue on MSVC
qt5_add_resources(RESOURCE_FILES ${QRC_FILES})

add_definitions(-DQT_PLUGIN)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${QRC_FILES} ${RESOURCE_FILES} ${QML_FILES} ${CONFIG_FILES})

AddCompilerDefinitions("SOFA_BUILD_SOFAQTQUICKGUI")
AddLinkerDependencies(SofaPython SofaSimulationGraph SofaComponentMain Qt5::Core Qt5::Gui Qt5::Widgets Qt5::Qml Qt5::Quick)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

