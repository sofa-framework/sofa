cmake_minimum_required(VERSION 2.8)

project("SofaPhysicsAPI")

include(${SOFA_CMAKE_DIR}/pre.cmake)

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

#set(LINKER_DEPENDENCIES )

include(${SOFA_CMAKE_DIR}/post.cmake)

