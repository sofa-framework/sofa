cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

project("SixenseHydra")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES 
    initSixenseHydra.h 
    RazerHydraDriver.h 
    )

set(SOURCE_FILES
    initSixenseHydra.cpp 
    RazerHydraDriver.cpp 
    )



add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set(COMPILER_DEFINES "SOFA_BUILD_SIXENSE_HYDRA")
set(LINKER_DEPENDENCIES SofaBaseLinearSolver SofaConstraint SofaUserInteraction SofaSimpleFem SofaBaseTopology SofaGuiCommon SofaMeshCollision)

# add Sixense SDK
find_package(Sixense REQUIRED)
include_directories(${SIXENSE_INCLUDE_DIR})
if(SIXENSE_LIBRARY_DYNAMIC)
  set(LIBRARIES
    ${LIBRARIES}
    ${SIXENSE_LIBRARY_DYNAMIC} ${SIXENSE_UTILS_LIBRARY}
    )
else(SIXENSE_LIBRARY_DYNAMIC)
  set(LIBRARIES
    ${LIBRARIES}
    ${SIXENSE_LIBRARY_STATIC} ${SIXENSE_UTILS_LIBRARY}
    )
endif(SIXENSE_LIBRARY_DYNAMIC)

include(${SOFA_CMAKE_DIR}/postProject.cmake)