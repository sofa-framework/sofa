set(SOFA_PROJECT_FOLDER "SofaLib")
RegisterProjects("SofaModeler" PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib")

set(SOFA_PROJECT_FOLDER "SofaApplication")
RegisterProjects("Modeler" PATH "${CMAKE_CURRENT_SOURCE_DIR}/exec")

add_subdirectory("exec")

# Configuration file for the Modeler
set(SOFA_SHARE_DIR ${SOFA_SRC_DIR}/share)
set(SOFA_EXAMPLES_DIR ${SOFA_SRC_DIR}/examples)
configure_file(Modeler-config.xml.in ${CMAKE_BINARY_DIR}/etc/Modeler-config.xml)
