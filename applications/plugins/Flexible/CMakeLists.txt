cmake_minimum_required(VERSION 2.8.12)
project(Flexible)

set(FLEXIBLE_MAJOR_VERSION 0)
set(FLEXIBLE_MINOR_VERSION 0)
set(FLEXIBLE_BUILD_VERSION 1)

set(FLEXIBLE_VERSION ${FLEXIBLE_MAJOR_VERSION}.${FLEXIBLE_MINOR_VERSION}.${FLEXIBLE_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")

set(FLEXIBLE_BUILD_LIB_DIR     ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(FLEXIBLE_BUILD_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(FLEXIBLE_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(FLEXIBLE_BUILD_CMAKE_DIR   ${CMAKE_BINARY_DIR}/Flexible)

set(FLEXIBLE_INSTALL_LIB_DIR     lib                      CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(FLEXIBLE_INSTALL_BIN_DIR     bin                      CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(FLEXIBLE_INSTALL_INCLUDE_DIR include                  CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")
set(FLEXIBLE_INSTALL_CMAKE_DIR   ${FLEXIBLE_INSTALL_LIB_DIR}/cmake/Flexible )

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
    deformationMapping/TetrahedronVolumeMapping.h
    deformationMapping/CorotationalMeshMapping.h
    engine/ComputeDualQuatEngine.h
    engine/ComputeDualQuatEngine.inl
    engine/ComputeWeightEngine.h
    mass/AffineMass.h
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
    deformationMapping/TetrahedronVolumeMapping.cpp
    deformationMapping/CorotationalMeshMapping.cpp
    engine/ComputeDualQuatEngine.cpp
    engine/ComputeWeightEngine.cpp
    mass/AffineMass.cpp
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


find_package(SofaMisc REQUIRED)

find_package(image)
if(image_FOUND)
   	list(APPEND HEADER_FILES "quadrature/ImageGaussPointSampler.h")
   	list(APPEND HEADER_FILES "shapeFunction/BaseImageShapeFunction.h")
   	list(APPEND HEADER_FILES "shapeFunction/VoronoiShapeFunction.h")
   	list(APPEND HEADER_FILES "shapeFunction/DiffusionShapeFunction.h")
   	list(APPEND HEADER_FILES "shapeFunction/ShapeFunctionDiscretizer.h")
   	list(APPEND HEADER_FILES "deformationMapping/ImageDeformation.h")
    list(APPEND HEADER_FILES "mass/MassFromDensity.h")
   	list(APPEND HEADER_FILES "mass/ImageDensityMass.h")
   	list(APPEND HEADER_FILES "mass/ImageDensityMass.inl")
    list(APPEND SOURCE_FILES "shapeFunction/ImageShapeFunctionContainer.h")
   	list(APPEND SOURCE_FILES "quadrature/ImageGaussPointSampler.cpp")
   	list(APPEND SOURCE_FILES "shapeFunction/VoronoiShapeFunction.cpp")
   	list(APPEND SOURCE_FILES "shapeFunction/DiffusionShapeFunction.cpp")
   	list(APPEND SOURCE_FILES "shapeFunction/ShapeFunctionDiscretizer.cpp")
   	list(APPEND SOURCE_FILES "deformationMapping/ImageDeformation.cpp")
    list(APPEND HEADER_FILES "mass/MassFromDensity.cpp")
   	list(APPEND SOURCE_FILES "mass/ImageDensityMass.cpp")
    list(APPEND SOURCE_FILES "shapeFunction/ImageShapeFunctionContainer.cpp")
endif()


# include python files
include(./python/python.cmake)

set(FLEXIBLE_DEPENDENCY_INCLUDE_DIR ${SofaMisc_INCLUDE_DIRS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Flexible.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/Flexible/Flexible.h)
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/Flexible/Flexible.h
         DESTINATION "${FLEXIBLE_INSTALL_INCLUDE_DIR}/Flexible")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} SofaComponentMisc)

set(FLEXIBLE_COMPILER_FLAGS "-DSOFA_BUILD_Flexible")

if(image_FOUND)
    target_link_libraries(${PROJECT_NAME} image)
    set(FLEXIBLE_COMPILER_FLAGS "${FLEXIBLE_COMPILER_FLAGS} -DSOFA_HAVE_IMAGE")
endif()

set(SOFA-EXTERNAL_MGDIFFUSI0N_PATH "" CACHE PATH "Path to the MGDIFFUSI0N library")
if(NOT SOFA-EXTERNAL_MGDIFFUSI0N_PATH STREQUAL "")
    set(FLEXIBLE_COMPILER_FLAGS "${FLEXIBLE_COMPILER_FLAGS} -DSOFA_HAVE_MGDIFFUSI0N")
    target_include_directories(${PROJECT_NAME} "${SOFA-EXTERNAL_MGDIFFUSI0N_PATH}/mgdiffusion/DiffusionSolver/")

    # link_directories( ${SOFA-EXTERNAL_MGDIFFUSI0N_PATH}/mgdiffusion/DiffusionSolver/ )
    # target_link_libraries(${PROJECT_NAME} "DiffusionSolver")
    # link_directories( ${SOFA-EXTERNAL_MGDIFFUSI0N_PATH}/PhysBAM/build/nocona/release/Public_Library/ )
    # link_directories( ${SOFA-EXTERNAL_MGDIFFUSI0N_PATH}/PhysBAM/build/nocona/release/Common_Libraries/ )
    # target_link_libraries(${PROJECT_NAME} "PhysBAM_PhysBAM_Tools" "Threading_Tools")

    # TODO fix this!
    # link_directories does not add a -Lpath to the link command so I found this workaround for now

    target_link_libraries(${PROJECT_NAME}
        ${SOFA-EXTERNAL_MGDIFFUSI0N_PATH}/mgdiffusion/DiffusionSolver/libDiffusionSolver.so
        ${SOFA-EXTERNAL_MGDIFFUSI0N_PATH}/PhysBAM/build/nocona/release/Public_Library/libPhysBAM_PhysBAM_Tools.so
        ${SOFA-EXTERNAL_MGDIFFUSI0N_PATH}/PhysBAM/build/nocona/release/Common_Libraries/libThreading_Tools.so)
endif()

if(WIN32)
    set(FLEXIBLE_COMPILER_FLAGS "${FLEXIBLE_COMPILER_FLAGS} -DEIGEN_DONT_ALIGN")
else()
	include_directories("/usr/include/suitesparse/")
endif()

if(UNIX)
    target_link_libraries(${PROJECT_NAME} pthread)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${FLEXIBLE_DEPENDENCY_INCLUDE_DIR}>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${FLEXIBLE_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_Flexible")


install(TARGETS ${PROJECT_NAME}
        COMPONENT Flexible_libraries
        EXPORT  FlexibleTargets
        RUNTIME DESTINATION "${FLEXIBLE_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${FLEXIBLE_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${FLEXIBLE_INSTALL_LIB_DIR}")

foreach(HEADER ${HEADER_FILES})
    string(REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
    install(FILES ${HEADER} DESTINATION "${FLEXIBLE_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}/${DIR}")
endforeach(HEADER HEADERS)

find_package(SofaTest QUIET)
if(SofaTest_FOUND)
    add_subdirectory(Flexible_test)
endif()

include(CMakePackageConfigHelpers)
# FlexibleConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${FLEXIBLE_BUILD_CMAKE_DIR}/FlexibleConfigVersion.cmake
                                 VERSION ${FLEXIBLE_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

install(FILES ${FLEXIBLE_BUILD_CMAKE_DIR}/FlexibleConfigVersion.cmake
        DESTINATION ${FLEXIBLE_INSTALL_CMAKE_DIR})


# FlexibleConfig.cmake (build tree)
set(FLEXIBLE_INCLUDE_DIR "${FLEXIBLE_BUILD_INCLUDE_DIR}")
configure_package_config_file(FlexibleConfig.cmake.in
                              ${FLEXIBLE_BUILD_CMAKE_DIR}/FlexibleConfig.cmake
                              INSTALL_DESTINATION ${FLEXIBLE_BUILD_CMAKE_DIR}
                              PATH_VARS FLEXIBLE_INCLUDE_DIR)

# FlexibleConfig.cmake (installed)
set(FLEXIBLE_INCLUDE_DIR "${FLEXIBLE_INSTALL_INCLUDE_DIR}")
configure_package_config_file(FlexibleConfig.cmake.in
                              ${FLEXIBLE_BUILD_CMAKE_DIR}/FlexibleConfigInstall.cmake
                              INSTALL_DESTINATION ${FLEXIBLE_INSTALL_CMAKE_DIR}
                              PATH_VARS FLEXIBLE_INCLUDE_DIR)

install(FILES ${FLEXIBLE_BUILD_CMAKE_DIR}/FlexibleConfigInstall.cmake
        DESTINATION ${FLEXIBLE_INSTALL_CMAKE_DIR}
        RENAME FlexibleConfig.cmake)

# FlexibleTargets.cmake (installed)
install(EXPORT FlexibleTargets
        DESTINATION ${FLEXIBLE_INSTALL_CMAKE_DIR})
