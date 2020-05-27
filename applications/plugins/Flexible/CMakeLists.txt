cmake_minimum_required(VERSION 3.1)
project(Flexible VERSION 0.1)

find_package(SofaMisc REQUIRED)
find_package(SofaValidation REQUIRED)
sofa_find_package(SofaPython REQUIRED)
sofa_find_package(image QUIET)
sofa_find_package(SofaHighOrderTopology QUIET)
find_package(Eigen3 REQUIRED)

set(HEADER_FILES
    config.h.in
    BaseJacobian.h
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
    forceField/FlexibleCorotationalFEMForceField.h
    forceField/FlexibleCorotationalMeshFEMForceField.h
    helper.h
    mass/AffineMass.h
    material/BaseMaterial.h
    material/BaseMaterialForceField.h
    material/HEMLStVKForceField.h
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
    quadrature/GaussPointSmoother.h
    shapeFunction/BarycentricShapeFunction.h
    shapeFunction/BaseShapeFunction.h
    shapeFunction/HatShapeFunction.h
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
    strainMapping/LinearStrainJacobianBlock.h
    strainMapping/LinearStrainMapping.h
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
    engine/TransformEngine.cpp
    engine/ComputeDualQuatEngine.cpp
    engine/ComputeWeightEngine.cpp
    forceField/FlexibleCorotationalFEMForceField.cpp
    forceField/FlexibleCorotationalMeshFEMForceField.cpp
    initFlexible.cpp
    mass/AffineMass.cpp
    material/BaseMaterialForceField.cpp
    material/HEMLStVKForceField.cpp
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
    quadrature/GaussPointSmoother.cpp
    shapeFunction/BarycentricShapeFunction.cpp
    shapeFunction/BaseShapeFunction.cpp
    shapeFunction/HatShapeFunction.cpp
    shapeFunction/ShepardShapeFunction.cpp
    strainMapping/BaseStrainMapping.cpp
    strainMapping/CauchyStrainMapping.cpp
    strainMapping/CorotationalStrainMapping.cpp
    strainMapping/GreenStrainMapping.cpp
    strainMapping/InvariantMapping.cpp
    strainMapping/PlasticStrainMapping.cpp
    strainMapping/PrincipalStretchesMapping.cpp
    strainMapping/RelativeStrainMapping.cpp
    strainMapping/LinearStrainMapping.cpp
    types/AffineComponents.cpp
    types/DeformationGradientTypes.cpp
    types/QuadraticComponents.cpp
    types/RigidConstraint.cpp
    types/StrainTypes.cpp
    )
set(README_FILES
    README_Flexible.txt
    )
set(PYTHON_FILES
    python/Flexible/__init__.py
    python/Flexible/IO.py
    python/Flexible/StressVisu.py
    python/Flexible/API.py
    python/Flexible/sml.py
    )
# Config files and install rules for pythons scripts
sofa_install_pythonscripts(PLUGIN_NAME ${PROJECT_NAME} PYTHONSCRIPTS_SOURCE_DIR "python")

if(image_FOUND)
    list(APPEND HEADER_FILES
        deformationMapping/ImageDeformation.h
        mass/ImageDensityMass.h
        mass/ImageDensityMass.inl
        mass/MassFromDensity.cpp
        mass/MassFromDensity.h
        quadrature/ImageGaussPointSampler.h
        shapeFunction/BaseImageShapeFunction.h
        shapeFunction/DiffusionShapeFunction.h
        shapeFunction/ImageShapeFunctionSelectNode.h
        shapeFunction/ImageShapeFunctionContainer.h
        shapeFunction/ShapeFunctionDiscretizer.h
        shapeFunction/VoronoiShapeFunction.h
        )
    list(APPEND SOURCE_FILES
        deformationMapping/ImageDeformation.cpp
        mass/ImageDensityMass.cpp
        quadrature/ImageGaussPointSampler.cpp
        shapeFunction/DiffusionShapeFunction.cpp
        shapeFunction/ImageShapeFunctionContainer.cpp
        shapeFunction/ImageShapeFunctionSelectNode.cpp
        shapeFunction/ShapeFunctionDiscretizer.cpp
        shapeFunction/VoronoiShapeFunction.cpp
        )
else()
    message(STATUS "Flexible: could not find image")
endif()

# @warning SofaHighOrderTopology is an external plugin that cannot be defined previously to Flexible that is an interal plugin
# so it must require to run cmake two times
if(SofaHighOrderTopology_FOUND)
    list(APPEND HEADER_FILES shapeFunction/BezierShapeFunction.h)
    list(APPEND SOURCE_FILES shapeFunction/BezierShapeFunction.cpp)
else()
    message(STATUS "Flexible: could not find SofaHighOrderTopology")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${PYTHON_FILES} ${README_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC SofaMisc SofaValidation SofaBaseVisual)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

if(image_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC image)
endif()

# TODO: Clean this. We should find_package(MGDiffusion) somehow.
set(SOFA-EXTERNAL_MGDIFFUSI0N_PATH "" CACHE PATH "Path to the MGDIFFUSI0N library")
if(NOT SOFA-EXTERNAL_MGDIFFUSI0N_PATH STREQUAL "")
    target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_HAVE_MGDIFFUSI0N")
    target_include_directories(${PROJECT_NAME} "${SOFA-EXTERNAL_MGDIFFUSI0N_PATH}/mgdiffusion/DiffusionSolver/")

    # TODO fix this!
    # link_directories does not add a -Lpath to the link command so I found this workaround for now

    target_link_libraries(${PROJECT_NAME} PUBLIC
        ${SOFA-EXTERNAL_MGDIFFUSI0N_PATH}/mgdiffusion/DiffusionSolver/libDiffusionSolver.so
        ${SOFA-EXTERNAL_MGDIFFUSI0N_PATH}/PhysBAM/build/nocona/release/Public_Library/libPhysBAM_PhysBAM_Tools.so
        ${SOFA-EXTERNAL_MGDIFFUSI0N_PATH}/PhysBAM/build/nocona/release/Common_Libraries/libThreading_Tools.so)
endif()

if(SOFA_NINJA_BUILD_POOLS)
    set(NINJA_BUILD_POOL_FLEXIBLE "1" CACHE STRING "Cores used to build Flexible with ninja")
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS flexible_pool=${NINJA_BUILD_POOL_FLEXIBLE})
    set_property(TARGET ${PROJECT_NAME} PROPERTY JOB_POOL_COMPILE flexible_pool)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_FLEXIBLE")

if(SofaHighOrderTopology_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaHighOrderTopology)
    target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SofaHighOrderTopology_SOURCE_DIR}/..>")
endif()

if(SOFA_BUILD_TESTS)
    find_package(SofaTest QUIET)
    if(SofaTest_FOUND)
        add_subdirectory(Flexible_test)
    endif()
endif()

## Install rules for the library; CMake package configurations files
sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    RELOCATABLE "plugins"
    )
