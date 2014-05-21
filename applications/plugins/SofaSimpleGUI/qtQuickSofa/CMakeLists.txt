include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
		Window.cpp
		Viewer.cpp
        Scene.cpp
		
		Main.cpp
)

set(HEADER_FILES
		Window.h
		Viewer.h
        Scene.h
)

set(QRC_FILES
		qtQuickSofa.qrc
)

# find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)

if(WIN32)
	if(CMAKE_CL_64)
		set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "C:/Program Files (x86)/Windows Kits/8.0/Lib/win8/um/x64")
	else()
		set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} "C:/Program Files (x86)/Windows Kits/8.0/Lib/win8/um/x86")
	endif()
endif()

find_package(Qt5Core HINTS "${SOFA-EXTERNAL_QT5_PATH}/lib/cmake/Qt5Core")
find_package(Qt5Gui HINTS "${SOFA-EXTERNAL_QT5_PATH}/lib/cmake/Qt5Gui")
find_package(Qt5Widgets HINTS "${SOFA-EXTERNAL_QT5_PATH}/lib/cmake/Qt5Widgets")
find_package(Qt5Quick HINTS "${SOFA-EXTERNAL_QT5_PATH}/lib/cmake/Qt5Quick")
find_package(Qt5Qml HINTS "${SOFA-EXTERNAL_QT5_PATH}/lib/cmake/Qt5Qml")

qt5_add_resources(RESOURCE_FILES ${QRC_FILES})


# make mocced header / resource files path from absolute to relative
#ToRelativePath(RESOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${RESOURCE_FILES}")

#set(MOC_FILES_GROUP "Generated")

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES} ${QRC_FILES} ${RESOURCE_FILES})

AddLinkerDependencies(SofaSimpleGUI Qt5::Core Qt5::Gui Qt5::Widgets Qt5::Qml Qt5::Quick)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

