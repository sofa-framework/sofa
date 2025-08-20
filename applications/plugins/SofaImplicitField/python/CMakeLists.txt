project(SofaImplicitField.Python)

set(SOURCE_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Binding_ScalarField.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Module_SofaImplicitField.cpp
)

set(HEADER_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Binding_ScalarField.h
)

if (NOT TARGET SofaPython3::Plugin)
    find_package(SofaPython3 REQUIRED COMPONENTS Plugin Bindings.Sofa.Core)
endif()

sofa_find_package(SofaImplicitField REQUIRED)

SP3_add_python_module(
    TARGET       ${PROJECT_NAME}
    PACKAGE      SofaImplicitField.Python
    MODULE       SofaImplicitField
    DESTINATION  .
    SOURCES      ${SOURCE_FILES}
    HEADERS      ${HEADER_FILES}
    DEPENDS      SofaImplicitField SofaPython3::Plugin SofaPython3::Bindings.Sofa.Core
)
