include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    
    SofaPhysicsAPI.h 
    SofaPhysicsSimulation_impl.h 
    SofaPhysicsOutputMesh_impl.h 
    SofaPhysicsDataMonitor_impl.h 
    SofaPhysicsDataController_impl.h 
    fakegui.h
    
    )

set(SOURCE_FILES

    main.cpp
    SofaPhysicsSimulation.cpp 
    SofaPhysicsOutputMesh.cpp 
    SofaPhysicsDataMonitor.cpp 
    SofaPhysicsDataController.cpp 
    fakegui.cpp

	)

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})

AddLinkerDependencies(SofaGuiMain SofaComponentMain)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

