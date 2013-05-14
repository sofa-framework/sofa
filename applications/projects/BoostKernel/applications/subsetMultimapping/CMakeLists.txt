include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    subsetMultimapping.cpp
    )

if(SOFA-LIB_GUI_QTVIEWER)
	UseQt()
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaComponentMain ObjectCreator SofaGuiMain SofaSimulationBGL)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

