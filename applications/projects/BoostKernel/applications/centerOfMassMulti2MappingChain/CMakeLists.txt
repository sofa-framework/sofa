include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    centerOfMassMulti2MappingChain.cpp
    )

if(SOFA-LIB_GUI_QTVIEWER)
	UseQt()
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

AddLinkerDependencies(SofaComponentMain ObjectCreator SofaGuiMain SofaSimulationBGL)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

