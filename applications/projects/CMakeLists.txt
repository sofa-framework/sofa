cmake_minimum_required(VERSION 3.1)

# Check if Qt(4/5) is present
set(SOFA_HAVE_QT_LIBRARIES 1)
find_package(Qt5 COMPONENTS Core QUIET)
if(NOT Qt5Core_FOUND)
    find_package(Qt4 QUIET COMPONENTS qtcore)
    if(NOT Qt4_FOUND)
        set(SOFA_HAVE_QT_LIBRARIES 0)
    endif()
endif()

if(SOFA_HAVE_QT_LIBRARIES)
    sofa_add_application(Modeler Modeler ON)
endif()

sofa_add_application(getDeprecatedComponents getDeprecatedComponents OFF)

sofa_add_application(GenerateRigid GenerateRigid)
sofa_add_application(meshconv meshconv OFF)

sofa_add_application(SofaPhysicsAPI SofaPhysicsAPI)
sofa_add_application(SofaGuiGlut SofaGuiGlut OFF)

sofa_add_application(runSofa runSofa ON)
