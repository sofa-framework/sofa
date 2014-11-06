cmake_minimum_required(VERSION 2.8)
project(SofaMiscFem)

set(HEADER_FILES

    initMiscFEM.h
    FastTetrahedralCorotationalForceField.h
    FastTetrahedralCorotationalForceField.inl
    TetrahedralTensorMassForceField.h
    TetrahedralTensorMassForceField.inl
	StandardTetrahedralFEMForceField.h
	StandardTetrahedralFEMForceField.inl
	BoyceAndArruda.h
	Costa.h
	HyperelasticMaterial.h
	MooneyRivlin.h
	STVenantKirchhoff.h
	VerondaWestman.h
    material/BaseMaterial.h
    material/NeoHookean.h
    material/Ogden.h
    )

set(SOURCE_FILES

    initMiscFEM.cpp
    FastTetrahedralCorotationalForceField.cpp
    TetrahedralTensorMassForceField.cpp
 	StandardTetrahedralFEMForceField.cpp

    )

if(SOFA_HAVE_NEWMAT)
  list(APPEND HEADER_FILES
	PlasticMaterial.h
  )
  list(APPEND SOURCE_FILES
	PlasticMaterial.cpp
  )
endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaSimpleFem)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAMISC_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAMISC_DEPENDENCY_DIR}>")

set( SOFAMISCFEM_COMPILER_FLAGS "-DSOFA_BUILD_MISC_FEM")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAMISC_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAMISCFEM_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaMiscFem_libraries
        EXPORT    SofaMiscTargets
        RUNTIME DESTINATION "${SOFAMISC_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFAMISC_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFAMISC_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${SOFAMISC_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}"
)
