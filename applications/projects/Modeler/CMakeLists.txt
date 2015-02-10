set(SOFA_PROJECT_FOLDER "SofaLib")
RegisterProjects("SofaModeler" PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib")

set(SOFA_PROJECT_FOLDER "SofaApplication")
RegisterProjects("Modeler" PATH "${CMAKE_CURRENT_SOURCE_DIR}/exec")

add_subdirectory("exec")

# Configuration file for the Modeler
set(SHARE_DIR ${SOFA_SRC_DIR}/share)
set(EXAMPLES_DIR ${SOFA_SRC_DIR}/examples)
if(WIN32)
    set(PLUGINS_DIR ../bin)
else()
    set(PLUGINS_DIR ../lib)
endif()
configure_file(Modeler-config.xml.in ${CMAKE_BINARY_DIR}/etc/Modeler-config.xml)
