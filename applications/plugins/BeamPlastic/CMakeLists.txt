cmake_minimum_required(VERSION 2.8.12)
project(BeamPlastic)

set(BEAMPLASTIC_MAJOR_VERSION 0)
set(BEAMPLASTIC_MINOR_VERSION 1)
set(BEAMPLASTIC_VERSION ${BEAMPLASTIC_MAJOR_VERSION}.${BEAMPLASTIC_MINOR_VERSION})

set(HEADER_FILES
	initBeamPlastic.h
	config.h
	forceField/BeamPlasticFEMForceField.h
	forceField/BeamPlasticFEMForceField.inl
    forceField/PlasticConstitutiveLaw.h
    forceField/RambergOsgood.h
	quadrature/gaussian.h
	quadrature/quadrature.h
	PoissonContainer.h
	StiffnessContainer.h
)

set(SOURCE_FILES
	initBeamPlastic.cpp
	forceField/BeamPlasticFEMForceField.cpp
)

find_package(SofaFramework REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaCore SofaBaseTopology SofaSimpleFem)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_BEAMPLASTIC"
	                                             DEBUG_POSTFIX "_d")


#include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")

install(TARGETS ${PROJECT_NAME}
        #COMPONENT PluginExample_libraries
        #EXPORT PluginExampleTargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
