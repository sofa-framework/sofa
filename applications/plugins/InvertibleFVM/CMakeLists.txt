include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    InvertibleFVMForceField.h
    InvertibleFVMForceField.inl
    initPlugin.h
)

set(SOURCE_FILES
    InvertibleFVMForceField.cpp
    initPlugin.cpp
)

set(README_FILES InvertibleFVM.txt)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_INVERTIBLEFVM")
AddLinkerDependencies(SofaGuiQt )

include(${SOFA_CMAKE_DIR}/postProject.cmake)