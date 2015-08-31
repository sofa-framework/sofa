include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    THMPGHashTable.h
    THMPGSpatialHashing.h
    initTHMPGSpatialHashingPlugin.h
)

set(SOURCE_FILES
    THMPGHashTable.cpp
    THMPGSpatialHashing.cpp
    initTHMPGSpatialHashingPlugin.cpp
)

#set(README_FILES Compliant.txt)


#if(WIN32)
# include_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Include")
# link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
#else()
# include_directories("/usr/include/suitesparse/")
#endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES} ${PYTHON_FILES})

#AddCompilerDefinitions("SOFA_BUILD_THMPGSpatialHashing")
#AddLinkerDependencies(SofaEigen2Solver SofaUserInteraction SofaComponentMisc SofaHelper SofaSimulationGraph)

# is this really useful?
#if(NOT APPLE)
# AddLinkerDependencies(amd lapack blas camd colamd cholmod)
#endif()

# WIN32: copy the needed dlls from the lib dir to the Sofa bin dir
#if(WIN32)
# file(GLOB sharedObjects "${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib/*.dll")
# foreach(sharedObject ${sharedObjects})
#  file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
# endforeach()
#endif()

AddCompilerDefinitions("SOFA_BUILD_THMPGSPATIALHASHING")






#### BOOST ######

# linking with boost-system is needed
# that means an installed version of boost is mandatory (sadly)
# I loved the header-only boost
if(NOT SOFA-EXTERNAL_BOOST)
    message(FATAL_ERROR "THMPGSpatialHashing plugin needs a compiled Boost library >= 1.48. Please modify the SOFA-EXTERNAL_BOOST_PATH option to fix it...")
endif()

find_package(Boost COMPONENTS system REQUIRED)
AddLinkerDependencies( ${Boost_SYSTEM_LIBRARY} )

if(LINUX OR UNIX AND NOT APPLE)
    AddLinkerDependencies( rt ) # clock_gettime
endif()


include(${SOFA_CMAKE_DIR}/postProject.cmake)
include(custom.cmake OPTIONAL)
