cmake_minimum_required(VERSION 3.22)
project(SofaCUDA.Component LANGUAGES CUDA CXX)

set(SOFACUDA_COMPONENT_MAJOR_VERSION 0)
set(SOFACUDA_COMPONENT_MINOR_VERSION 1)
set(SOFACUDA_COMPONENT_VERSION ${SOFACUDA_COMPONENT_MAJOR_VERSION}.${SOFACUDA_COMPONENT_MINOR_VERSION})

set(SOFACUDA_COMPONENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(HEADER_FILES
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/config.h.in
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/init.h
    
    ### Mechanical
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/statecontainer/CudaMechanicalObject.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/statecontainer/CudaMechanicalObject.inl

    ### Mappings
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaBarycentricMapping.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaBarycentricMapping.inl
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaBarycentricMappingRigid.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaBeamLinearMapping.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaIdentityMapping.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaIdentityMapping.inl
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/nonlinear/CudaRigidMapping.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/nonlinear/CudaRigidMapping.inl
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaSubsetMapping.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaSubsetMapping.inl
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaSubsetMultiMapping.h


    ### Mass
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mass/CudaDiagonalMass.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mass/CudaDiagonalMass.inl
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mass/CudaMeshMatrixMass.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mass/CudaMeshMatrixMass.inl
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mass/CudaUniformMass.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mass/CudaUniformMass.inl
    

    ### solidmechanics
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/fem/elastic/CudaHexahedronFEMForceField.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/fem/elastic/CudaHexahedronFEMForceField.inl
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/fem/hyperelastic/CudaStandardTetrahedralFEMForceField.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/fem/hyperelastic/CudaStandardTetrahedralFEMForceField.inl
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/tensormass/CudaTetrahedralTensorMassForceField.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/tensormass/CudaTetrahedralTensorMassForceField.inl
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/fem/elastic/CudaTetrahedronFEMForceField.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/fem/elastic/CudaTetrahedronFEMForceField.inl
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/fem/elastic/CudaTriangularFEMForceFieldOptim.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/fem/elastic/CudaTriangularFEMForceFieldOptim.inl
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/spring/CudaSpringForceField.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/spring/CudaSpringForceField.inl
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/tled/CudaHexahedronTLEDForceField.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/tled/CudaTetrahedronTLEDForceField.h

    ### MechanicalLoad
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mechanicalload/CudaConstantForceField.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mechanicalload/CudaEllipsoidForceField.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mechanicalload/CudaEllipsoidForceField.inl
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mechanicalload/CudaLinearForceField.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mechanicalload/CudaLinearForceField.inl
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mechanicalload/CudaPlaneForceField.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mechanicalload/CudaPlaneForceField.inl
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mechanicalload/CudaSphereForceField.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mechanicalload/CudaSphereForceField.inl

    ### Collisions
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/collision/geometry/CudaLineModel.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/collision/geometry/CudaPointModel.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/collision/geometry/CudaSphereModel.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/collision/geometry/CudaTriangleModel.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/collision/response/mapper/CudaContactMapper.h

    ### Constraints
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/constraint/projective/CudaFixedProjectiveConstraint.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/constraint/projective/CudaFixedProjectiveConstraint.inl
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/constraint/projective/CudaLinearMovementProjectiveConstraint.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/constraint/projective/CudaLinearMovementProjectiveConstraint.inl
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/collision/response/contact/CudaPenalityContactForceField.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/collision/response/contact/CudaPenalityContactForceField.inl

    ### Visual
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/visual/CudaVisualModel.h
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/visual/CudaVisualModel.inl
)

set(SOURCE_FILES
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/init.cpp

    ### Mechanical
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/statecontainer/CudaMechanicalObject.cpp

    ### Mappings
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaBarycentricMapping-3f.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaBarycentricMapping-3f1-3f.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaBarycentricMapping-3f1-d.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaBarycentricMapping-3f1-f.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaBarycentricMapping-3f1.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaBarycentricMapping-f.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaBarycentricMapping.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaBarycentricMappingRigid.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaIdentityMapping.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/nonlinear/CudaRigidMapping.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaSubsetMapping.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaSubsetMultiMapping.cpp

    ### Mass
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mass/CudaDiagonalMass.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mass/CudaMeshMatrixMass.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mass/CudaUniformMass.cpp

    ### Solidmechanics
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/fem/elastic/CudaHexahedronFEMForceField.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/fem/hyperelastic/CudaStandardTetrahedralFEMForceField.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/tensormass/CudaTetrahedralTensorMassForceField.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/fem/elastic/CudaTetrahedronFEMForceField.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/fem/elastic/CudaTriangularFEMForceFieldOptim.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/spring/CudaSpringForceField.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/spring/CudaRestShapeSpringsForceField.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/tled/CudaHexahedronTLEDForceField.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/tled/CudaTetrahedronTLEDForceField.cpp

    ### MechanicalLoad
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mechanicalload/CudaConstantForceField.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mechanicalload/CudaEllipsoidForceField.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mechanicalload/CudaLinearForceField.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mechanicalload/CudaPlaneForceField.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mechanicalload/CudaSphereForceField.cpp
    

    ### Collisions
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/collision/geometry/CudaLineModel.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/collision/geometry/CudaPointModel.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/collision/geometry/CudaSphereModel.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/collision/geometry/CudaTriangleModel.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/collision/response/contact/CudaPenalityContactForceField.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/collision/response/mapper/CudaContactMapper.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/collision/detection/intersection/CudaProximityIntersection.cpp

    ### Constraints  
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/constraint/lagrangian/model/CudaBilateralLagrangianConstraint.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/constraint/projective/CudaFixedProjectiveConstraint.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/constraint/projective/CudaFixedTranslationProjectiveConstraint.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/constraint/projective/CudaLinearMovementProjectiveConstraint.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/constraint/projective/CudaLinearVelocityProjectiveConstraint.cpp

    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaBeamLinearMapping.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/engine/select/CudaBoxROI.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/engine/select/CudaNearestPointROI.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/engine/select/CudaSphereROI.cpp

    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/engine/transform/CudaIndexValueMapper.cpp

    ### ConstraintCorrection
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/constraint/lagrangian/correction/CudaLinearSolverConstraintCorrection.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/constraint/lagrangian/correction/CudaPrecomputedConstraintCorrection.cpp
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/constraint/lagrangian/correction/CudaUncoupledConstraintCorrection.cpp

    ### Topology
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/topology/container/dynamic/CudaSetTopology.cpp

    ### Visual 
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/visual/CudaVisualModel.cpp
  
)

set(CUDA_SOURCES
    
    ### Mechanical
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/statecontainer/CudaMechanicalObject.cu

    ### Mappings
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaBarycentricMapping.cu
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/nonlinear/CudaRigidMapping.cu
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mapping/linear/CudaSubsetMapping.cu

    ### Mass
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mass/CudaDiagonalMass.cu
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mass/CudaMeshMatrixMass.cu
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mass/CudaUniformMass.cu

    ### solidmechanics
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/fem/elastic/CudaHexahedronFEMForceField.cu
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/fem/hyperelastic/CudaStandardTetrahedralFEMForceField.cu
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/tensormass/CudaTetrahedralTensorMassForceField.cu
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/fem/elastic/CudaTetrahedronFEMForceField.cu
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/fem/elastic/CudaTriangularFEMForceFieldOptim.cu
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/spring/CudaSpringForceField.cu
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/tled/CudaHexahedronTLEDForceField.cu
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/solidmechanics/tled/CudaTetrahedronTLEDForceField.cu

    ### mechanicalload
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mechanicalload/CudaEllipsoidForceField.cu
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mechanicalload/CudaLinearForceField.cu
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mechanicalload/CudaPlaneForceField.cu
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/mechanicalload/CudaSphereForceField.cu

    ### Collisions
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/collision/response/contact/CudaPenalityContactForceField.cu
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/collision/response/mapper/CudaContactMapper.cu
    
    ### Constraints
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/constraint/projective/CudaFixedProjectiveConstraint.cu
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/constraint/projective/CudaLinearMovementProjectiveConstraint.cu

    ### Visual
    ${SOFACUDA_COMPONENT_SOURCE_DIR}/SofaCUDA/component/visual/CudaVisualModel.cu
    
    
)

find_package(Sofa.Config REQUIRED)
sofa_find_package(SofaCUDA.Core REQUIRED)
sofa_find_package(Sofa.Component.Mass REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.FEM.Elastic REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.FEM.HyperElastic REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.TensorMass REQUIRED)
sofa_find_package(Sofa.Component.Collision.Response.Contact REQUIRED)
sofa_find_package(Sofa.Component.Collision.Detection.Algorithm REQUIRED)
sofa_find_package(Sofa.Component.Collision.Detection.Intersection REQUIRED)
sofa_find_package(Sofa.Component.StateContainer REQUIRED)
sofa_find_package(Sofa.Component.Constraint.Projective REQUIRED)
sofa_find_package(Sofa.Component.Mapping.Linear REQUIRED)
sofa_find_package(Sofa.Component.Mapping.NonLinear REQUIRED)
sofa_find_package(Sofa.Component.Engine.Select REQUIRED)
sofa_find_package(Sofa.Component.Engine.Transform REQUIRED)
sofa_find_package(Sofa.Component.MechanicalLoad REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${CUDA_SOURCES})

target_link_libraries(${PROJECT_NAME} PUBLIC SofaCUDA.Core)
target_link_libraries(${PROJECT_NAME} PUBLIC
    Sofa.Component.Mass
    Sofa.Component.SolidMechanics.FEM.Elastic
    Sofa.Component.SolidMechanics.FEM.HyperElastic
    Sofa.Component.SolidMechanics.TensorMass
    Sofa.Component.MechanicalLoad
    Sofa.Component.Collision.Response.Contact
    Sofa.Component.Collision.Detection.Algorithm
    Sofa.Component.Collision.Detection.Intersection
    Sofa.Component.StateContainer
    Sofa.Component.Constraint.Projective
    Sofa.Component.Mapping.Linear
    Sofa.Component.Mapping.NonLinear
    Sofa.Component.Engine.Select
    Sofa.Component.Engine.Transform
    Sofa.Component.MechanicalLoad
)

# see (I guess) https://github.com/sofa-framework/sofa/blob/314b95cbfba411bf8431486ea75b7c67c0bbcc76/Sofa/framework/Config/cmake/SofaMacrosInstall.cmake#L695
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>/include/${PROJECT_NAME}/${PROJECT_NAME}")

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${SOFACUDA_COMPONENT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    INCLUDE_SOURCE_DIR "src"
    RELOCATABLE "plugins"
)
