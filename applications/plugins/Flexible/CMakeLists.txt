cmake_minimum_required(VERSION 2.8)

project("SofaFlexible")

include(${SOFA_CMAKE_DIR}/pre.cmake)

##plugin external dependencies
find_package( REQUIRED)
include_directories(${})

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
    quadrature/BaseGaussPointSampler.h 
    quadrature/TopologyGaussPointSampler.h 
    shapeFunction/BaseShapeFunction.h 
    shapeFunction/ShepardShapeFunction.h 
    shapeFunction/HatShapeFunction.h 
    shapeFunction/BarycentricShapeFunction.h 
    deformationMapping/BaseDeformationMapping.h 
    deformationMapping/BaseDeformationMapping.inl 
    deformationMapping/DistanceMapping.h 
    deformationMapping/DistanceMapping.inl 
    deformationMapping/ExtensionMapping.h 
    deformationMapping/ExtensionMapping.inl 
    deformationMapping/LinearMapping.h 
    deformationMapping/LinearJacobianBlock.h 
    deformationMapping/LinearJacobianBlock_point.inl 
    deformationMapping/LinearJacobianBlock_affine.inl 
    deformationMapping/LinearJacobianBlock_rigid.inl 
    deformationMapping/LinearJacobianBlock_quadratic.inl 
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
    strainMapping/BaseStrainMapping.h 
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
    #forceField/FlexibleTetrahedronFEMForceField.h 
    forceField/FlexibleForceField.h 

	)

set(SOURCE_FILES

    initFlexible.cpp 
    types/DeformationGradientTypes.cpp 
    types/StrainTypes.cpp 
    types/AffineComponents.cpp 
    types/QuadraticComponents.cpp 
    quadrature/BaseGaussPointSampler.cpp 
    quadrature/TopologyGaussPointSampler.cpp 
    shapeFunction/BaseShapeFunction.cpp 
    shapeFunction/ShepardShapeFunction.cpp 
    shapeFunction/HatShapeFunction.cpp 
    shapeFunction/BarycentricShapeFunction.cpp 
    deformationMapping/DistanceMapping.cpp 
    deformationMapping/ExtensionMapping.cpp 
    deformationMapping/LinearMapping_point.cpp 
    deformationMapping/LinearMapping_affine.cpp 
    deformationMapping/LinearMapping_rigid.cpp 
    deformationMapping/LinearMapping_quadratic.cpp 
    deformationMapping/MLSMapping_point.cpp 
    deformationMapping/MLSMapping_affine.cpp 
    deformationMapping/MLSMapping_rigid.cpp 
    deformationMapping/MLSMapping_quadratic.cpp 
    deformationMapping/TriangleDeformationMapping.cpp 
    deformationMapping/TriangleStrainAverageMapping.cpp 
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
    #forceField/FlexibleTetrahedronFEMForceField.cpp 
	forceField/FlexibleForceField.cpp

	)

if(SOFA_HAVE_IMAGE)

    if(SOFA_IMAGE_HAVE_OPENCV) #should be "SOFA_HAVE_OPENCV" -> use "SOFA_IMAGE_HAVE_OPENCV" until the opencv plugin is fixed..
        find_package( SofaOpenCVLibs REQUIRED)
        include_directories(${SOFAOPENCV_INCLUDE_DIR})
    endif()
    
    include_directories(${SOFAFLEXIBLE_INCLUDE_DIR})
        
   	list(APPEND HEADER_FILES "quadrature/ImageGaussPointSampler.h")
   	list(APPEND HEADER_FILES "shapeFunction/BaseImageShapeFunction.h")
   	list(APPEND HEADER_FILES "shapeFunction/VoronoiShapeFunction.h")
   	list(APPEND HEADER_FILES "shapeFunction/DiffusionShapeFunction.h")
   	list(APPEND HEADER_FILES "shapeFunction/ShapeFunctionDiscretizer.h")
   	list(APPEND HEADER_FILES "deformationMapping/ImageDeformation.h")
   	list(APPEND HEADER_FILES "mass/ImageDensityMass.h")
   	list(APPEND HEADER_FILES "mass/ImageDensityMass.inl")
    
   	list(APPEND SOURCE_FILES "quadrature/ImageGaussPointSampler.cpp")
   	list(APPEND SOURCE_FILES "shapeFunction/VoronoiShapeFunction.cpp")
   	list(APPEND SOURCE_FILES "shapeFunction/DiffusionShapeFunction.cpp")
   	list(APPEND SOURCE_FILES "shapeFunction/ShapeFunctionDiscretizer.cpp")
   	list(APPEND SOURCE_FILES "deformationMapping/ImageDeformation.cpp")
   	list(APPEND SOURCE_FILES "mass/ImageDensityMass.cpp")
        
endif()

set(README_FILE Flexible.txt )	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_Flexible")
set(LINKER_DEPENDENCIES SofaGUIQt )

include(${SOFA_CMAKE_DIR}/post.cmake)