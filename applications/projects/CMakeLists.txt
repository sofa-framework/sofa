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

############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    sofa_add_application(GenerateRigid GenerateRigid)
    sofa_add_application(meshconv meshconv OFF)
    sofa_add_application(SofaPhysicsAPI SofaPhysicsAPI)
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_DEPRECATEDCOMPONENTS)
    message(STATUS "   With deprecated components.")
    sofa_add_application(runSofa runSofa ON)
else()
    message(STATUS "   Without deprecated components.")
endif()


