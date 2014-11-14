include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initFlexible.h
    BaseJacobian.h 
    helper.h 
    types/DeformationGradientTypes.h 
    types/StrainTypes.h 
    types/AffineTypes.h 
    types/AffineComponents.h 
    types/QuadraticTypes.h 
    types/QuadraticComponents.h 
    types/PolynomialBasis.h 
    types/ComponentSpecializations.h.inl
    types/ComponentSpecializations.cpp.inl
    types/ComponentSpecializationsDefines.h
    types/ComponentSpecializationsUndef.h
    types/DeformableFrameMass.h
    types/RigidConstraint.h
    quadrature/BaseGaussPointSampler.h
    quadrature/TopologyGaussPointSampler.h 
    quadrature/GaussPointContainer.h
    shapeFunction/BaseShapeFunction.h
    shapeFunction/ImageShapeFunctionSelectNode.h
    shapeFunction/ShepardShapeFunction.h 
    shapeFunction/HatShapeFunction.h 
    shapeFunction/BarycentricShapeFunction.h 
    shapeFunction/BezierShapeFunction.h
    deformationMapping/BaseDeformationImpl.inl 
    deformationMapping/BaseDeformationMapping.h
    deformationMapping/BaseDeformationMapping.inl 
    deformationMapping/CustomJacobianBlock.h
    deformationMapping/LinearMapping.h
    deformationMapping/LinearJacobianBlock.h
    deformationMapping/LinearJacobianBlock_point.inl 
    deformationMapping/LinearJacobianBlock_affine.inl 
    deformationMapping/LinearJacobianBlock_rigid.inl 
    deformationMapping/LinearJacobianBlock_quadratic.inl 
    deformationMapping/BaseDeformationMultiMapping.h
    deformationMapping/BaseDeformationMultiMapping.inl
    deformationMapping/LinearMultiMapping.h
    deformationMapping/MLSMapping.h
    deformationMapping/MLSJacobianBlock.h 
    deformationMapping/MLSJacobianBlock_point.inl 
    deformationMapping/MLSJacobianBlock_affine.inl 
    deformationMapping/MLSJacobianBlock_rigid.inl 
    deformationMapping/MLSJacobianBlock_quadratic.inl 
    deformationMapping/TriangleDeformationMapping.h 
    deformationMapping/TriangleDeformationMapping.inl 
    deformationMapping/TriangleStrainAverageMapping.h 
    deformationMapping/TriangleStrainAverageMapping.inl 
    deformationMapping/VolumeMapping.h
    deformationMapping/CorotationalMeshMapping.h
    engine/ComputeDualQuatEngine.h
    engine/ComputeDualQuatEngine.inl
    engine/ComputeWeightEngine.h
    strainMapping/BaseStrainMapping.h
    strainMapping/CauchyStrainMapping.h
    strainMapping/CauchyStrainJacobianBlock.h
    strainMapping/GreenStrainMapping.h
    strainMapping/GreenStrainJacobianBlock.h
    strainMapping/CorotationalStrainMapping.h
    strainMapping/CorotationalStrainJacobianBlock.h 
    strainMapping/CorotationalStrainJacobianBlock.inl 
    strainMapping/PrincipalStretchesMapping.h 
    strainMapping/PrincipalStretchesJacobianBlock.h 
    strainMapping/InvariantMapping.h 
    strainMapping/InvariantJacobianBlock.h 
    strainMapping/InvariantJacobianBlock.inl 
    strainMapping/PlasticStrainMapping.h 
    strainMapping/PlasticStrainJacobianBlock.h 
    strainMapping/RelativeStrainMapping.h 
    strainMapping/RelativeStrainJacobianBlock.h 
    material/BaseMaterial.h 
    material/BaseMaterialForceField.h 
    material/HookeForceField.h 
    material/HookeMaterialBlock.h 
    material/HookeMaterialBlock.inl 
    material/StabilizedHookeForceField.h 
    material/StabilizedHookeMaterialBlock.h 
    material/NeoHookeanForceField.h 
    material/NeoHookeanMaterialBlock.h 
    material/StabilizedNeoHookeanForceField.h 
    material/StabilizedNeoHookeanMaterialBlock.h 
    material/MooneyRivlinForceField.h 
    material/MooneyRivlinMaterialBlock.h 
    material/OgdenForceField.h 
    material/OgdenMaterialBlock.h 
    material/VolumePreservationForceField.h 
    material/VolumePreservationMaterialBlock.h 
    material/VolumePreservationMaterialBlock.inl 
    material/PolynomialMaterialForceField.h
    material/PolynomialMaterialBlock.h
    material/MuscleMaterialForceField.h
    material/MuscleMaterialBlock.h
    material/TendonMaterialForceField.h
    material/TendonMaterialBlock.h
    material/ProjectiveForceField.h
    material/ProjectiveMaterialBlock.h
    #forceField/FlexibleTetrahedronFEMForceField.h

	)

set(SOURCE_FILES

    initFlexible.cpp 
    types/DeformationGradientTypes.cpp 
    types/StrainTypes.cpp 
    types/AffineComponents.cpp 
    types/QuadraticComponents.cpp 
    types/RigidConstraint.cpp
    quadrature/BaseGaussPointSampler.cpp
    quadrature/TopologyGaussPointSampler.cpp 
    quadrature/GaussPointContainer.cpp
    shapeFunction/BaseShapeFunction.cpp
    shapeFunction/ImageShapeFunctionSelectNode.cpp
    shapeFunction/ShepardShapeFunction.cpp 
    shapeFunction/HatShapeFunction.cpp 
    shapeFunction/BarycentricShapeFunction.cpp 
    shapeFunction/BezierShapeFunction.cpp
    deformationMapping/LinearMapping_point.cpp
    deformationMapping/LinearMapping_affine.cpp 
    deformationMapping/LinearMapping_rigid.cpp 
    deformationMapping/LinearMapping_quadratic.cpp 
    deformationMapping/LinearMultiMapping_rigidaffine.cpp
    deformationMapping/MLSMapping_point.cpp
    deformationMapping/MLSMapping_affine.cpp 
    deformationMapping/MLSMapping_rigid.cpp 
    deformationMapping/MLSMapping_quadratic.cpp 
    deformationMapping/TriangleDeformationMapping.cpp 
    deformationMapping/TriangleStrainAverageMapping.cpp 
    deformationMapping/VolumeMapping.cpp
    deformationMapping/CorotationalMeshMapping.cpp
    engine/ComputeDualQuatEngine.cpp
    engine/ComputeWeightEngine.cpp
    strainMapping/CauchyStrainMapping.cpp
    strainMapping/GreenStrainMapping.cpp
    strainMapping/CorotationalStrainMapping.cpp 
    strainMapping/PrincipalStretchesMapping.cpp 
    strainMapping/InvariantMapping.cpp 
    strainMapping/PlasticStrainMapping.cpp 
    strainMapping/RelativeStrainMapping.cpp 
    material/HookeForceField.cpp 
    material/StabilizedHookeForceField.cpp 
    material/NeoHookeanForceField.cpp 
    material/StabilizedNeoHookeanForceField.cpp 
    material/MooneyRivlinForceField.cpp 
    material/OgdenForceField.cpp 
    material/VolumePreservationForceField.cpp 
    material/PolynomialMaterialForceField.cpp
    material/MuscleMaterialForceField.cpp
    material/TendonMaterialForceField.cpp
    material/ProjectiveForceField.cpp
    #forceField/FlexibleTetrahedronFEMForceField.cpp

	)

if(SOFA-PLUGIN_IMAGE)
    
   	list(APPEND HEADER_FILES "quadrature/ImageGaussPointSampler.h")
   	list(APPEND HEADER_FILES "shapeFunction/BaseImageShapeFunction.h")
   	list(APPEND HEADER_FILES "shapeFunction/VoronoiShapeFunction.h")
   	list(APPEND HEADER_FILES "shapeFunction/DiffusionShapeFunction.h")
   	list(APPEND HEADER_FILES "shapeFunction/ShapeFunctionDiscretizer.h")
   	list(APPEND HEADER_FILES "deformationMapping/ImageDeformation.h")
   	list(APPEND HEADER_FILES "mass/ImageDensityMass.h")
   	list(APPEND HEADER_FILES "mass/ImageDensityMass.inl")
        list(APPEND SOURCE_FILES "shapeFunction/ImageShapeFunctionContainer.h")

   	list(APPEND SOURCE_FILES "quadrature/ImageGaussPointSampler.cpp")
   	list(APPEND SOURCE_FILES "shapeFunction/VoronoiShapeFunction.cpp")
   	list(APPEND SOURCE_FILES "shapeFunction/DiffusionShapeFunction.cpp")
   	list(APPEND SOURCE_FILES "shapeFunction/ShapeFunctionDiscretizer.cpp")
   	list(APPEND SOURCE_FILES "deformationMapping/ImageDeformation.cpp")
   	list(APPEND SOURCE_FILES "mass/ImageDensityMass.cpp")
        list(APPEND SOURCE_FILES "shapeFunction/ImageShapeFunctionContainer.cpp")

        AddLinkerDependencies(image)
        AddCompilerDefinitions("SOFA_HAVE_IMAGE" )
endif()


# include python files
include(./python/python.cmake)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${PYTHON_FILES})

AddCompilerDefinitions("SOFA_BUILD_Flexible")
AddLinkerDependencies(SofaEigen2Solver SofaBaseVisual)

if(WIN32)
	include_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Include")

	AddCompilerDefinitions("EIGEN_DONT_ALIGN")
else()
	include_directories("/usr/include/suitesparse/")
endif()
if(UNIX)
    link_directories("/usr/lib/x86_64-linux-gnu/")
    AddLinkerDependencies("pthread")
endif()



sofa_option(SOFA-EXTERNAL_MGDIFFUSI0N_PATH PATH "" "Path to the MGDIFFUSI0N library")
if(NOT SOFA-EXTERNAL_MGDIFFUSI0N_PATH STREQUAL "")
    AddCompilerDefinitions("SOFA_HAVE_MGDIFFUSI0N")
    include_directories( "${SOFA-EXTERNAL_MGDIFFUSI0N_PATH}/mgdiffusion/DiffusionSolver/" )

#    link_directories( ${SOFA-EXTERNAL_MGDIFFUSI0N_PATH}/mgdiffusion/DiffusionSolver/ )
#    AddLinkerDependencies("DiffusionSolver")
#    link_directories( ${SOFA-EXTERNAL_MGDIFFUSI0N_PATH}/PhysBAM/build/nocona/release/Public_Library/ )
#    link_directories( ${SOFA-EXTERNAL_MGDIFFUSI0N_PATH}/PhysBAM/build/nocona/release/Common_Libraries/ )
#    AddLinkerDependencies("PhysBAM_PhysBAM_Tools" "Threading_Tools")


# TODO fix this!
# link_directories does not add a -Lpath to the link command so I found this workaround for now

    target_link_libraries (${PROJECT_NAME} ${SOFA-EXTERNAL_MGDIFFUSI0N_PATH}/mgdiffusion/DiffusionSolver/libDiffusionSolver.so
                                            ${SOFA-EXTERNAL_MGDIFFUSI0N_PATH}/PhysBAM/build/nocona/release/Public_Library/libPhysBAM_PhysBAM_Tools.so
                                            ${SOFA-EXTERNAL_MGDIFFUSI0N_PATH}/PhysBAM/build/nocona/release/Common_Libraries/libThreading_Tools.so )


endif()





include(${SOFA_CMAKE_DIR}/postProject.cmake)
