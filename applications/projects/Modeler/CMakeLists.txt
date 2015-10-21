#set(SOFA_PROJECT_FOLDER "SofaLib")
#RegisterProjects("SofaModeler" PATH "${CMAKE_CURRENT_SOURCE_DIR}/lib")

#set(SOFA_PROJECT_FOLDER "SofaApplication")
#RegisterProjects("Modeler" PATH "${CMAKE_CURRENT_SOURCE_DIR}/exec")

add_subdirectory("exec")
add_subdirectory("lib")
