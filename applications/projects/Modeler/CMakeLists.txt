set(SOFA_PROJECT_FOLDER "SofaLib")
RegisterDependencies("SofaModeler" PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib")

set(SOFA_PROJECT_FOLDER "SofaApplication")
RegisterDependencies("Modeler" PATH "${CMAKE_CURRENT_SOURCE_DIR}/exec")

add_subdirectory("exec")