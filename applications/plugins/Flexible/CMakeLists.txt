include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    BaseJacobian.h
    config.h
    deformationMapping/BaseDeformationImpl.inl
    deformationMapping/BaseDeformationMapping.h
    deformationMapping/BaseDeformationMapping.inl
    deformationMapping/BaseDeformationMultiMapping.h
    deformationMapping/BaseDeformationMultiMapping.inl
    deformationMapping/CorotationalMeshMapping.h
    deformationMapping/CustomJacobianBlock.h
    deformationMapping/LinearJacobianBlock.h
    deformationMapping/LinearJacobianBlock_affine.inl
    deformationMapping/LinearJacobianBlock_point.inl
    deformationMapping/LinearJacobianBlock_quadratic.inl
    deformationMapping/LinearJacobianBlock_rigid.inl
    deformationMapping/LinearMapping.h
    deformationMapping/LinearMultiMapping.h
    deformationMapping/MLSJacobianBlock.h
    deformationMapping/MLSJacobianBlock_affine.inl
    deformationMapping/MLSJacobianBlock_point.inl
    deformationMapping/MLSJacobianBlock_quadratic.inl
    deformationMapping/MLSJacobianBlock_rigid.inl
    deformationMapping/MLSMapping.h
    deformationMapping/TetrahedronVolumeMapping.h
    deformationMapping/TriangleDeformationMapping.h
    deformationMapping/TriangleDeformationMapping.inl
    deformationMapping/TriangleStrainAverageMapping.h
    deformationMapping/TriangleStrainAverageMapping.inl
    deformationMapping/VolumeMapping.h
    engine/ComputeDualQuatEngine.h
    engine/ComputeDualQuatEngine.inl
    engine/ComputeWeightEngine.h
    # forceField/FlexibleTetrahedronFEMForceField.h
    helper.h
    mass/AffineMass.h
    material/BaseMaterial.h
    material/BaseMaterialForceField.h
    material/HookeForceField.h
    material/HookeMaterialBlock.h
    material/HookeMaterialBlock.inl
    material/MooneyRivlinForceField.h
    material/MooneyRivlinMaterialBlock.h
    material/MuscleMaterialBlock.h
    material/MuscleMaterialForceField.h
    material/NeoHookeanForceField.h
    material/NeoHookeanMaterialBlock.h
    material/OgdenForceField.h
    material/OgdenMaterialBlock.h
    material/PolynomialMaterialBlock.h
    material/PolynomialMaterialForceField.h
    material/ProjectiveForceField.h
    material/ProjectiveMaterialBlock.h
    material/StabilizedHookeForceField.h
    material/StabilizedHookeMaterialBlock.h
    material/StabilizedNeoHookeanForceField.h
    material/StabilizedNeoHookeanMaterialBlock.h
    material/TendonMaterialBlock.h
    material/TendonMaterialForceField.h
    material/VolumePreservationForceField.h
    material/VolumePreservationMaterialBlock.h
    material/VolumePreservationMaterialBlock.inl
    quadrature/BaseGaussPointSampler.h
    quadrature/GaussPointContainer.h
    quadrature/TopologyGaussPointSampler.h
    shapeFunction/BarycentricShapeFunction.h
    shapeFunction/BaseShapeFunction.h
    shapeFunction/BezierShapeFunction.h
    shapeFunction/HatShapeFunction.h
    shapeFunction/ImageShapeFunctionSelectNode.h
    shapeFunction/ShepardShapeFunction.h
    strainMapping/BaseStrainMapping.h
    strainMapping/CauchyStrainJacobianBlock.h
    strainMapping/CauchyStrainMapping.h
    strainMapping/CorotationalStrainJacobianBlock.h
    strainMapping/CorotationalStrainJacobianBlock.inl
    strainMapping/CorotationalStrainMapping.h
    strainMapping/GreenStrainJacobianBlock.h
    strainMapping/GreenStrainMapping.h
    strainMapping/InvariantJacobianBlock.h
    strainMapping/InvariantJacobianBlock.inl
    strainMapping/InvariantMapping.h
    strainMapping/PlasticStrainJacobianBlock.h
    strainMapping/PlasticStrainMapping.h
    strainMapping/PrincipalStretchesJacobianBlock.h
    strainMapping/PrincipalStretchesMapping.h
    strainMapping/RelativeStrainJacobianBlock.h
    strainMapping/RelativeStrainMapping.h
    types/AffineComponents.h
    types/AffineTypes.h
    types/ComponentSpecializations.cpp.inl
    types/ComponentSpecializations.h.inl
    types/ComponentSpecializationsDefines.h
    types/ComponentSpecializationsUndef.h
    types/DeformableFrameMass.h
    types/DeformationGradientTypes.h
    types/PolynomialBasis.h
    types/QuadraticComponents.h
    types/QuadraticTypes.h
    types/RigidConstraint.h
    types/StrainTypes.h
)

set(SOURCE_FILES
    deformationMapping/BaseDeformationMapping.cpp
    deformationMapping/BaseDeformationMultiMapping.cpp
    deformationMapping/CorotationalMeshMapping.cpp
    deformationMapping/LinearMapping_affine.cpp
    deformationMapping/LinearMapping_point.cpp
    deformationMapping/LinearMapping_quadratic.cpp
    deformationMapping/LinearMapping_rigid.cpp
    deformationMapping/LinearMultiMapping_rigidaffine.cpp
    deformationMapping/MLSMapping_affine.cpp
    deformationMapping/MLSMapping_point.cpp
    deformationMapping/MLSMapping_quadratic.cpp
    deformationMapping/MLSMapping_rigid.cpp
    deformationMapping/TetrahedronVolumeMapping.cpp
    deformationMapping/TriangleDeformationMapping.cpp
    deformationMapping/TriangleStrainAverageMapping.cpp
    deformationMapping/VolumeMapping.cpp
    engine/ComputeDualQuatEngine.cpp
    engine/ComputeWeightEngine.cpp
    # forceField/FlexibleTetrahedronFEMForceField.cpp
    initFlexible.cpp
    mass/AffineMass.cpp
    material/BaseMaterialForceField.cpp
    material/HookeForceField.cpp
    material/MooneyRivlinForceField.cpp
    material/MuscleMaterialForceField.cpp
    material/NeoHookeanForceField.cpp
    material/OgdenForceField.cpp
    material/PolynomialMaterialForceField.cpp
    material/ProjectiveForceField.cpp
    material/StabilizedHookeForceField.cpp
    material/StabilizedNeoHookeanForceField.cpp
    material/TendonMaterialForceField.cpp
    material/VolumePreservationForceField.cpp
    quadrature/BaseGaussPointSampler.cpp
    quadrature/GaussPointContainer.cpp
    quadrature/TopologyGaussPointSampler.cpp
    shapeFunction/BarycentricShapeFunction.cpp
    shapeFunction/BaseShapeFunction.cpp
    shapeFunction/BezierShapeFunction.cpp
    shapeFunction/HatShapeFunction.cpp
    shapeFunction/ImageShapeFunctionSelectNode.cpp
    shapeFunction/ShepardShapeFunction.cpp
    strainMapping/BaseStrainMapping.cpp
    strainMapping/CauchyStrainMapping.cpp
    strainMapping/CorotationalStrainMapping.cpp
    strainMapping/GreenStrainMapping.cpp
    strainMapping/InvariantMapping.cpp
    strainMapping/PlasticStrainMapping.cpp
    strainMapping/PrincipalStretchesMapping.cpp
    strainMapping/RelativeStrainMapping.cpp
    types/AffineComponents.cpp
    types/DeformationGradientTypes.cpp
    types/QuadraticComponents.cpp
    types/RigidConstraint.cpp
    types/StrainTypes.cpp
)

set(PYTHON_FILES

    python/Flexible/__init__.py
    python/Flexible/IO.py
    python/Flexible/StressVisu.py
    python/Flexible/API.py
    python/Flexible/sml.py
)

if(SOFA-PLUGIN_IMAGE)

    list(APPEND HEADER_FILES "deformationMapping/ImageDeformation.h")
    list(APPEND HEADER_FILES "mass/ImageDensityMass.h")
    list(APPEND HEADER_FILES "mass/ImageDensityMass.inl")
    list(APPEND HEADER_FILES "mass/MassFromDensity.cpp")
    list(APPEND HEADER_FILES "mass/MassFromDensity.h")
    list(APPEND HEADER_FILES "quadrature/ImageGaussPointSampler.h")
    list(APPEND HEADER_FILES "shapeFunction/BaseImageShapeFunction.h")
    list(APPEND HEADER_FILES "shapeFunction/DiffusionShapeFunction.h")
    list(APPEND HEADER_FILES "shapeFunction/ShapeFunctionDiscretizer.h")
    list(APPEND HEADER_FILES "shapeFunction/VoronoiShapeFunction.h")
    list(APPEND SOURCE_FILES "deformationMapping/ImageDeformation.cpp")
    list(APPEND SOURCE_FILES "mass/ImageDensityMass.cpp")
    list(APPEND SOURCE_FILES "quadrature/ImageGaussPointSampler.cpp")
    list(APPEND SOURCE_FILES "shapeFunction/DiffusionShapeFunction.cpp")
    list(APPEND SOURCE_FILES "shapeFunction/ImageShapeFunctionContainer.cpp")
    list(APPEND SOURCE_FILES "shapeFunction/ImageShapeFunctionContainer.h")
    list(APPEND SOURCE_FILES "shapeFunction/ShapeFunctionDiscretizer.cpp")
    list(APPEND SOURCE_FILES "shapeFunction/VoronoiShapeFunction.cpp")

    AddCompilerDefinitions("SOFA_HAVE_IMAGE" )


    AddLinkerDependencies(image)

endif()


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

if(APPLE)
    find_package(X11)
    if(NOT X11_FOUND)
        message(FATAL_ERROR "Failed to find X11 which is required to build image plugin")
    endif(NOT X11_FOUND)
    AddLinkerDependencies(${X11_X11_LIB})
    include_directories(${X11_X11_INCLUDE_PATH})
endif()



include(${SOFA_CMAKE_DIR}/postProject.cmake)
