cmake_minimum_required(VERSION 3.1)
project(SofaExporter)

################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES
    config.h
    initExporter.h
)

set(SOURCE_FILES
    initExporter.cpp
)

set(EXTRA_FILES
    README.md
    examples/MeshExporter.scn
    examples/STLExporter.scn
    examples/OBJExporter.scn
)


############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND HEADER_FILES
         BlenderExporter.h
         BlenderExporter.inl
         MeshExporter.h
         OBJExporter.h
         STLExporter.h
         VTKExporter.h
         WriteState.h
         WriteState.inl
         WriteTopology.h
         WriteTopology.inl)

    list(APPEND SOURCE_FILES
         BlenderExporter.cpp
         MeshExporter.cpp
         OBJExporter.cpp
         STLExporter.cpp
         VTKExporter.cpp
         WriteState.cpp
         WriteTopology.cpp)
    message(STATUS "SofaExporter: build all maintained sofa components.")
else()
    message(STATUS "SofaExporter: build with only the minimal set of components.")
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_DEPRECATEDCOMPONENTS)
    message(STATUS "SofaExporter: build with deprecated components.")
else()
    message(STATUS "SofaExporter: build without deprecated components.")
endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationTree)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_EXPORTER")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
