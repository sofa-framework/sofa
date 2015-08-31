include(${SOFA_CMAKE_DIR}/preProject.cmake)


set(HEADER_FILES
    ClosestPointRegistrationForceField.h
    ClosestPointRegistrationForceField.inl
    GroupwiseRegistrationEngine.h
    InertiaAlign.h
    RegistrationContact.h
    RegistrationContact.inl
    RegistrationContactForceField.h
    RegistrationContactForceField.inl
    RegistrationExporter.h
    initRegistration.h
)

set(SOURCE_FILES
    ClosestPointRegistrationForceField.cpp
    GroupwiseRegistrationEngine.cpp
    InertiaAlign.cpp
    RegistrationContact.cpp
    RegistrationContactForceField.cpp
    RegistrationExporter.cpp
    initRegistration.cpp
)

set(PYTHON_FILES

    python/Registration/__init__.py
    python/Registration/sml.py

)

set(README_FILES README.txt)

if(SOFA-PLUGIN_IMAGE)
    list(APPEND HEADER_FILES "IntensityProfileRegistrationForceField.h")
    list(APPEND HEADER_FILES "IntensityProfileRegistrationForceField.inl")
    list(APPEND SOURCE_FILES "IntensityProfileRegistrationForceField.cpp")
    AddLinkerDependencies(image)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES} ${PYTHON_FILES})
AddCompilerDefinitions("SOFA_BUILD_REGISTRATION")
AddLinkerDependencies(SofaBaseCollision SofaGuiCommon SofaBaseVisual SofaExporter SofaLoader SofaOpenglVisual SofaMiscForceField)
#AddLinkerDependencies(SofaGuiQt)


include(${SOFA_CMAKE_DIR}/postProject.cmake)
