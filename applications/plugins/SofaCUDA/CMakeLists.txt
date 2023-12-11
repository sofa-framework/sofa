cmake_minimum_required(VERSION 3.12)
project(SofaCUDA LANGUAGES CUDA CXX)

set(SOFACUDA_MAJOR_VERSION 0)
set(SOFACUDA_MINOR_VERSION 1)
set(SOFACUDA_VERSION ${SOFACUDA_MAJOR_VERSION}.${SOFACUDA_MINOR_VERSION})

# quick and dirty fix for nvcc compatibility with -fno-partial-inlining flag
if(NOT WIN32)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif()

set(SOFACUDA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/SofaCUDA")

set(HEADER_FILES
    config.h.in
    
    ### Common
    sofa/gpu/cuda/CudaBaseVector.h
    sofa/gpu/cuda/CudaCommon.h
    sofa/gpu/cuda/CudaMath.h
    sofa/gpu/cuda/CudaMath.inl
    sofa/gpu/cuda/CudaMathRigid.h
    sofa/gpu/cuda/CudaMathRigid.inl
    sofa/gpu/cuda/CudaMatrix.h
    sofa/gpu/cuda/CudaMemoryManager.h
    sofa/gpu/cuda/CudaScan.h
    sofa/gpu/cuda/CudaSort.h
    sofa/gpu/cuda/CudaTexture.h
    sofa/gpu/cuda/CudaTypes.h
    sofa/gpu/cuda/mycuda.h

    ### Mechanical
    ${SOFACUDA_SOURCE_DIR}/component/statecontainer/CudaMechanicalObject.h
    ${SOFACUDA_SOURCE_DIR}/component/statecontainer/CudaMechanicalObject.inl

    ### Mappings
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaBarycentricMapping.h
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaBarycentricMapping.inl
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaBarycentricMappingRigid.h
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaIdentityMapping.h
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaIdentityMapping.inl
    ${SOFACUDA_SOURCE_DIR}/component/mapping/nonlinear/CudaRigidMapping.h
    ${SOFACUDA_SOURCE_DIR}/component/mapping/nonlinear/CudaRigidMapping.inl
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaSubsetMapping.h
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaSubsetMapping.inl


    ### Mass
    ${SOFACUDA_SOURCE_DIR}/component/mass/CudaDiagonalMass.h
    ${SOFACUDA_SOURCE_DIR}/component/mass/CudaDiagonalMass.inl
    ${SOFACUDA_SOURCE_DIR}/component/mass/CudaMeshMatrixMass.h
    ${SOFACUDA_SOURCE_DIR}/component/mass/CudaMeshMatrixMass.inl
    ${SOFACUDA_SOURCE_DIR}/component/mass/CudaUniformMass.h
    ${SOFACUDA_SOURCE_DIR}/component/mass/CudaUniformMass.inl
    

    ### FEM
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaHexahedronFEMForceField.h
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaHexahedronFEMForceField.inl
    sofa/gpu/cuda/CudaHexahedronTLEDForceField.h
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/hyperelastic/CudaStandardTetrahedralFEMForceField.h
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/hyperelastic/CudaStandardTetrahedralFEMForceField.inl
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/tensormass/CudaTetrahedralTensorMassForceField.h
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/tensormass/CudaTetrahedralTensorMassForceField.inl
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaTetrahedronFEMForceField.h
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaTetrahedronFEMForceField.inl
    sofa/gpu/cuda/CudaTetrahedronTLEDForceField.h
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaTriangularFEMForceFieldOptim.h
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaTriangularFEMForceFieldOptim.inl

    ### ForceFields
    ${SOFACUDA_SOURCE_DIR}/component/mechanicalload/CudaEllipsoidForceField.h
    ${SOFACUDA_SOURCE_DIR}/component/mechanicalload/CudaEllipsoidForceField.inl
    ${SOFACUDA_SOURCE_DIR}/component/mechanicalload/CudaLinearForceField.h
    ${SOFACUDA_SOURCE_DIR}/component/mechanicalload/CudaLinearForceField.inl
    ${SOFACUDA_SOURCE_DIR}/component/mechanicalload/CudaPlaneForceField.h
    ${SOFACUDA_SOURCE_DIR}/component/mechanicalload/CudaPlaneForceField.inl
    ${SOFACUDA_SOURCE_DIR}/component/mechanicalload/CudaSphereForceField.h
    ${SOFACUDA_SOURCE_DIR}/component/mechanicalload/CudaSphereForceField.inl
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/spring/CudaSpringForceField.h
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/spring/CudaSpringForceField.inl

    ### Collisions
    ${SOFACUDA_SOURCE_DIR}/component/collision/geometry/CudaLineModel.h
    ${SOFACUDA_SOURCE_DIR}/component/collision/geometry/CudaPointModel.h
    ${SOFACUDA_SOURCE_DIR}/component/collision/geometry/CudaSphereModel.h
    ${SOFACUDA_SOURCE_DIR}/component/collision/geometry/CudaTriangleModel.h

    ### Constraints
    ${SOFACUDA_SOURCE_DIR}/component/constraint/projective/CudaFixedProjectiveConstraint.h
    ${SOFACUDA_SOURCE_DIR}/component/constraint/projective/CudaFixedProjectiveConstraint.inl
    ${SOFACUDA_SOURCE_DIR}/component/constraint/projective/CudaLinearMovementProjectiveConstraint.h
    ${SOFACUDA_SOURCE_DIR}/component/constraint/projective/CudaLinearMovementProjectiveConstraint.inl
    ${SOFACUDA_SOURCE_DIR}/component/collision/response/contact/CudaPenalityContactForceField.h
    ${SOFACUDA_SOURCE_DIR}/component/collision/response/contact/CudaPenalityContactForceField.inl
  
)

set(SOURCE_FILES
    ### Common
    main.cpp
    sofa/gpu/cuda/CudaBaseVector.cpp
    sofa/gpu/cuda/mycuda.cpp

    ### Mechanical
    ${SOFACUDA_SOURCE_DIR}/component/statecontainer/CudaMechanicalObject.cpp
    sofa/gpu/cuda/CudaSetTopology.cpp

    ### Mappings
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaBarycentricMapping-3f.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaBarycentricMapping-3f1-3f.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaBarycentricMapping-3f1-d.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaBarycentricMapping-3f1-f.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaBarycentricMapping-3f1.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaBarycentricMapping-f.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaBarycentricMapping.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaBarycentricMappingRigid.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaIdentityMapping.cpp
    sofa/gpu/cuda/CudaMultiMapping.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mapping/nonlinear/CudaRigidMapping.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaSubsetMapping.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaSubsetMultiMapping.cpp

    ### Mass
    ${SOFACUDA_SOURCE_DIR}/component/mass/CudaDiagonalMass.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mass/CudaMeshMatrixMass.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mass/CudaUniformMass.cpp

    ### FEM
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaHexahedronFEMForceField.cpp
    sofa/gpu/cuda/CudaHexahedronTLEDForceField.cpp
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/hyperelastic/CudaStandardTetrahedralFEMForceField.cpp
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/tensormass/CudaTetrahedralTensorMassForceField.cpp
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaTetrahedronFEMForceField.cpp
    sofa/gpu/cuda/CudaTetrahedronTLEDForceField.cpp
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaTriangularFEMForceFieldOptim.cpp

    ### ForceFields
    ${SOFACUDA_SOURCE_DIR}/component/mechanicalload/CudaConstantForceField.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mechanicalload/CudaEllipsoidForceField.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mechanicalload/CudaLinearForceField.cpp
    ${SOFACUDA_SOURCE_DIR}/component/collision/response/contact/CudaPenalityContactForceField.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mechanicalload/CudaPlaneForceField.cpp
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/spring/CudaRestShapeSpringsForceField.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mechanicalload/CudaSphereForceField.cpp
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/spring/CudaSpringForceField.cpp
    

    ### Collisions
    sofa/gpu/cuda/CudaCollision.cpp
    ${SOFACUDA_SOURCE_DIR}/component/collision/geometry/CudaLineModel.cpp
    ${SOFACUDA_SOURCE_DIR}/component/collision/geometry/CudaPointModel.cpp
    ${SOFACUDA_SOURCE_DIR}/component/collision/geometry/CudaSphereModel.cpp
    ${SOFACUDA_SOURCE_DIR}/component/collision/geometry/CudaTriangleModel.cpp

    ### Constraints  
    ${SOFACUDA_SOURCE_DIR}/component/constraint/lagrangian/model/CudaBilateralLagrangianConstraint.cpp
    ${SOFACUDA_SOURCE_DIR}/component/constraint/projective/CudaFixedProjectiveConstraint.cpp
    ${SOFACUDA_SOURCE_DIR}/component/constraint/projective/CudaFixedTranslationProjectiveConstraint.cpp
    ${SOFACUDA_SOURCE_DIR}/component/constraint/projective/CudaLinearMovementProjectiveConstraint.cpp
    ${SOFACUDA_SOURCE_DIR}/component/constraint/projective/CudaLinearVelocityProjectiveConstraint.cpp

    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaBeamLinearMapping.cpp
    ${SOFACUDA_SOURCE_DIR}/component/engine/select/CudaBoxROI.cpp
    ${SOFACUDA_SOURCE_DIR}/component/engine/select/CudaNearestPointROI.cpp
    ${SOFACUDA_SOURCE_DIR}/component/engine/select/CudaSphereROI.cpp

    ${SOFACUDA_SOURCE_DIR}/component/engine/transform/CudaIndexValueMapper.cpp

    ### ConstraintCorrection
    ${SOFACUDA_SOURCE_DIR}/component/constraint/lagrangian/correction/CudaLinearSolverConstraintCorrection.cpp
    ${SOFACUDA_SOURCE_DIR}/component/constraint/lagrangian/correction/CudaPrecomputedConstraintCorrection.cpp
    ${SOFACUDA_SOURCE_DIR}/component/constraint/lagrangian/correction/CudaUncoupledConstraintCorrection.cpp
)

set(CUDA_SOURCES
    ### Common
    sofa/gpu/cuda/mycuda.cu
    sofa/gpu/cuda/CudaBaseVector.cu
    sofa/gpu/cuda/CudaScan.cu
    sofa/gpu/cuda/CudaSort.cu

    ### Mechanical
    ${SOFACUDA_SOURCE_DIR}/component/statecontainer/CudaMechanicalObject.cu

    ### Mappings
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaBarycentricMapping.cu
    ${SOFACUDA_SOURCE_DIR}/component/mapping/nonlinear/CudaRigidMapping.cu
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaSubsetMapping.cu

    ### Mass
    ${SOFACUDA_SOURCE_DIR}/component/mass/CudaDiagonalMass.cu
    ${SOFACUDA_SOURCE_DIR}/component/mass/CudaMeshMatrixMass.cu
    ${SOFACUDA_SOURCE_DIR}/component/mass/CudaUniformMass.cu

    ### FEM
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaHexahedronFEMForceField.cu
    sofa/gpu/cuda/CudaHexahedronTLEDForceField.cu
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/hyperelastic/CudaStandardTetrahedralFEMForceField.cu
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/tensormass/CudaTetrahedralTensorMassForceField.cu
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaTetrahedronFEMForceField.cu
    sofa/gpu/cuda/CudaTetrahedronTLEDForceField.cu
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaTriangularFEMForceFieldOptim.cu

    ### ForceFields
    ${SOFACUDA_SOURCE_DIR}/component/mechanicalload/CudaEllipsoidForceField.cu
    ${SOFACUDA_SOURCE_DIR}/component/mechanicalload/CudaLinearForceField.cu
    ${SOFACUDA_SOURCE_DIR}/component/collision/response/contact/CudaPenalityContactForceField.cu
    ${SOFACUDA_SOURCE_DIR}/component/mechanicalload/CudaPlaneForceField.cu
    ${SOFACUDA_SOURCE_DIR}/component/mechanicalload/CudaSphereForceField.cu
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/spring/CudaSpringForceField.cu

    ### Collisions
    
    ### Constraints
    ${SOFACUDA_SOURCE_DIR}/component/constraint/projective/CudaFixedProjectiveConstraint.cu
    ${SOFACUDA_SOURCE_DIR}/component/constraint/projective/CudaLinearMovementProjectiveConstraint.cu
    
    
)

sofa_find_package(Sofa.GL QUIET)
if(NOT Sofa.GL_FOUND)
    message(WARNING "Sofa.GL is not active.")
    message("OpenGL-related code wont be enabled (Sharing OpenGL buffers with CUDA)")
else()
    list(APPEND HEADER_FILES
        sofa/gpu/cuda/CudaVisualModel.h
        sofa/gpu/cuda/CudaVisualModel.inl
    )
    list(APPEND SOURCE_FILES
        sofa/gpu/cuda/CudaVisualModel.cpp
    )
    list(APPEND CUDA_SOURCES
        sofa/gpu/cuda/CudaVisualModel.cu
    )
endif()

#Compatibility
if(SOFA_ENABLE_LEGACY_HEADERS)
    list(APPEND HEADER_FILES
        sofa/gpu/cuda/CudaBarycentricMapping.h
        sofa/gpu/cuda/CudaBarycentricMapping.inl
        sofa/gpu/cuda/CudaBarycentricMappingRigid.h
        sofa/gpu/cuda/CudaRigidMapping.h
        sofa/gpu/cuda/CudaRigidMapping.inl
        sofa/gpu/cuda/CudaIdentityMapping.h
        sofa/gpu/cuda/CudaIdentityMapping.inl
        sofa/gpu/cuda/CudaSubsetMapping.h
        sofa/gpu/cuda/CudaSubsetMapping.inl

        sofa/gpu/cuda/CudaDiagonalMass.h
        sofa/gpu/cuda/CudaDiagonalMass.inl

        sofa/gpu/cuda/CudaMeshMatrixMass.h
        sofa/gpu/cuda/CudaMeshMatrixMass.inl

        sofa/gpu/cuda/CudaUniformMass.h
        sofa/gpu/cuda/CudaUniformMass.inl

        sofa/gpu/cuda/CudaHexahedronFEMForceField.h
        sofa/gpu/cuda/CudaHexahedronFEMForceField.inl
        sofa/gpu/cuda/CudaStandardTetrahedralFEMForceField.h
        sofa/gpu/cuda/CudaStandardTetrahedralFEMForceField.inl
        sofa/gpu/cuda/CudaTetrahedralTensorMassForceField.h
        sofa/gpu/cuda/CudaTetrahedralTensorMassForceField.inl
        sofa/gpu/cuda/CudaTetrahedronFEMForceField.h
        sofa/gpu/cuda/CudaTetrahedronFEMForceField.inl
        sofa/gpu/cuda/CudaTriangularFEMForceFieldOptim.h
        sofa/gpu/cuda/CudaTriangularFEMForceFieldOptim.inl

        sofa/gpu/cuda/CudaFixedConstraint.h
        sofa/gpu/cuda/CudaFixedConstraint.inl
        sofa/gpu/cuda/CudaLinearMovementConstraint.h
        sofa/gpu/cuda/CudaLinearMovementConstraint.inl

        sofa/gpu/cuda/CudaLineModel.h
        sofa/gpu/cuda/CudaPointModel.h
        sofa/gpu/cuda/CudaSphereModel.h
        sofa/gpu/cuda/CudaTriangleModel.h

        sofa/gpu/cuda/CudaPenalityContactForceField.h
        sofa/gpu/cuda/CudaPenalityContactForceField.inl

        sofa/gpu/cuda/CudaSpringForceField.h
        sofa/gpu/cuda/CudaSpringForceField.inl

        sofa/gpu/cuda/CudaSphereForceField.h
        sofa/gpu/cuda/CudaSphereForceField.inl

        sofa/gpu/cuda/CudaPlaneForceField.h
        sofa/gpu/cuda/CudaPlaneForceField.inl

        sofa/gpu/cuda/CudaLinearForceField.h
        sofa/gpu/cuda/CudaLinearForceField.inl

        sofa/gpu/cuda/CudaEllipsoidForceField.h
        sofa/gpu/cuda/CudaEllipsoidForceField.inl

        sofa/gpu/cuda/CudaMechanicalObject.h
        sofa/gpu/cuda/CudaMechanicalObject.inl
    )
endif()

set(README_FILES README.md)

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

sofa_find_package(VolumetricRendering QUIET)
if(VolumetricRendering_FOUND)
    list(APPEND HEADER_FILES sofa/gpu/cuda/CudaTetrahedralVisualModel.h)
    list(APPEND HEADER_FILES sofa/gpu/cuda/CudaTetrahedralVisualModel.inl)
    list(APPEND SOURCE_FILES sofa/gpu/cuda/CudaTetrahedralVisualModel.cpp)
    add_definitions("-DSOFACUDA_ENABLE_VOLUMETRICRENDERING")
else()
    message(STATUS "SofaCUDA: Plugin VolumetricRendering was not enabled/found, therefore CudaTetrahedralVisualModel will not be compiled.")
endif()

sofa_find_package(Sofa.GUI QUIET)
if(Sofa.GUI_FOUND)
    list(APPEND SOURCE_FILES sofa/gpu/gui/CudaMouseInteraction.cpp)
endif()

sofa_find_package(Sofa.GUI.Qt QUIET)
if(Sofa.GUI.Qt_FOUND)
    list(APPEND HEADER_FILES sofa/gpu/gui/CudaDataWidget.h)
    list(APPEND SOURCE_FILES sofa/gpu/gui/CudaDataWidget.cpp)
endif()


find_package(SofaDistanceGrid QUIET)
if(SofaDistanceGrid_FOUND)
    sofa_find_package(MiniFlowVR QUIET)
    if(MiniFlowVR_FOUND)
        message(STATUS "SofaCUDA: MiniFlowVR enabled.")
    else()
        message(STATUS "SofaCUDA: MiniFlowVR was not enabled, therefore some feature of CudaDistanceGridCollisionModel will not be compiled.")
    endif()

    list(APPEND HEADER_FILES
        sofa/gpu/cuda/CudaDistanceGridCollisionModel.h
        sofa/gpu/cuda/CudaContactMapper.h
        sofa/gpu/cuda/CudaCollisionDetection.h)

    list(APPEND SOURCE_FILES
        sofa/gpu/cuda/CudaDistanceGridCollisionModel.cpp
        sofa/gpu/cuda/CudaCollisionDistanceGrid.cpp
        sofa/gpu/cuda/CudaCollisionDetection.cpp)

    list(APPEND CUDA_SOURCES
        sofa/gpu/cuda/CudaContactMapper.cu
        sofa/gpu/cuda/CudaCollisionDetection.cu)
else()
    message(STATUS "SofaCUDA: SofaDistanceGrid was not enabled, therefore CudaDistanceGridCollisionModel will not be compiled.")
endif()

find_package(SofaSphFluid QUIET)
if(SofaSphFluid_FOUND)
    list(APPEND HEADER_FILES
        sofa/gpu/cuda/CudaParticleSource.h
        sofa/gpu/cuda/CudaParticleSource.inl
        sofa/gpu/cuda/CudaSPHFluidForceField.h
        sofa/gpu/cuda/CudaSPHFluidForceField.inl
        sofa/gpu/cuda/CudaParticlesRepulsionForceField.h
        sofa/gpu/cuda/CudaParticlesRepulsionForceField.inl
        sofa/gpu/cuda/CudaSpatialGridContainer.h
        sofa/gpu/cuda/CudaSpatialGridContainer.inl
        )
    list(APPEND SOURCE_FILES
    sofa/gpu/cuda/CudaParticleSource.cpp
    sofa/gpu/cuda/CudaSPHFluidForceField.cpp
    sofa/gpu/cuda/CudaParticlesRepulsionForceField.cpp
    sofa/gpu/cuda/CudaSpatialGridContainer.cpp
    )
    list(APPEND CUDA_SOURCES
    sofa/gpu/cuda/CudaParticleSource.cu
    sofa/gpu/cuda/CudaSPHFluidForceField.cu
    sofa/gpu/cuda/CudaParticlesRepulsionForceField.cu
    sofa/gpu/cuda/CudaSpatialGridContainer.cu
    )
    message(STATUS "SofaCUDA: optional dependency to SofaSphFluid found. ")
else()
    message(STATUS "SofaCUDA: optional dependency SofaSphFluid not found. ")
endif()

sofa_find_package(SofaValidation QUIET)
if(SofaValidation_FOUND)
    list(APPEND SOURCE_FILES
        sofa/gpu/cuda/CudaExtraMonitor.cpp
        )
    message(STATUS "SofaCUDA: found SofaValidation, enabling CudaExtraMonitor (Warning: this component is being deprecated). ")
endif()

option(SOFACUDA_VERBOSE_PTXAS "???" OFF)
if(SOFACUDA_VERBOSE_PTXAS)
    set(VERBOSE_PTXAS --ptxas-options=-v)
endif()

option(SOFACUDA_CUBLAS "Activate cublas support in CUDA (requires SOFACUDA_DOUBLE)." OFF)
if(SOFACUDA_CUBLAS)
    set(SOFA_GPU_CUBLAS 1)       # #cmakedefine
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
    find_package(CUDASparse REQUIRED)
endif()

option(SOFACUDA_CUDPP "Activate CUDPP (for RadixSort)." OFF)
if(SOFACUDA_CUDPP)
    set(SOFA_GPU_CUDPP 1)       # #cmakedefine
endif()

# Note: THRUST is included in CUDA SDK 4.0+, it is recommended to use it if available
option(SOFACUDA_THRUST "Activate THRUST (for RadixSort)." ON)
if(SOFACUDA_THRUST)
    set(SOFA_GPU_THRUST 1)       # #cmakedefine
endif()

option(SOFACUDA_DOUBLE "Activate double-precision support in CUDA (requires GT200+ GPU and -arch sm_13 flag." OFF)
if(SOFACUDA_DOUBLE)
    set(SOFA_GPU_CUDA_DOUBLE 1)       # #cmakedefine
endif()


option(SOFACUDA_DOUBLE_PRECISE "Enable double-precision for sqrt/div... (requires compute capability
>= 2 and CUDA_VERSION > 3.0)" OFF)
# Note: with SOFA_GPU_CUDA_PRECISE and SOFA_GPU_CUDA_DOUBLE you get IEEE
# 754-compliant floating point operations for addition and multiplication only.
if(SOFACUDA_DOUBLE_PRECISE)
    set(SOFA_GPU_CUDA_DOUBLE_PRECISE 1)       # #cmakedefine
endif()

option(SOFACUDA_PRECISE "Use IEEE 754-compliant floating point operations." OFF)

include(SofaCUDANvccFlags.cmake)

# nvcc uses a "host code compiler" to compile CPU code, specified by CUDA_HOST_COMPILER.
# With some versions of CMake, CUDA_HOST_COMPILER defaults to CMAKE_C_COMPILER,
# but few host compilers are actually supported. Workarounds should go here.
if (${CUDA_HOST_COMPILER} MATCHES "ccache$")
    message(STATUS "SofaCUDA: CUDA host compiler was set to ccache, changing to g++")
    set(CUDA_HOST_COMPILER "g++" CACHE STRING "Host side compiler used by NVCC" FORCE)
endif()

# quick and dirty fix for nvcc compatibility with -fno-partial-inlining flag
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

if(WIN32)
    add_definitions("-DWIN32")
    set(WINDOWSOPTIONS "/MD")
endif()

# Make the compiler definitions available to nvcc
add_definitions("-DSOFA_BUILD_SOFACUDA")
# Give include directories to nvcc compiler.
# Note: keep cuda_include_directories() before cuda_add_library().
cuda_include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
cuda_include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")
cuda_include_directories("${CMAKE_BINARY_DIR}/include")
cuda_add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${CUDA_SOURCES} ${README_FILES} OPTIONS ${VERBOSE_PTXAS} ${WINDOWSOPTIONS})


target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_SOURCE_DIR}/src>")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFACUDA_COMPILE_DEFINITIONS}")
target_link_libraries(${PROJECT_NAME}
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

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

if(Sofa.GL_FOUND)
    target_link_libraries(${PROJECT_NAME} Sofa.GL)
endif()

if(SofaDistanceGrid_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaDistanceGrid)
    if(MiniFlowVR_FOUND)
        target_link_libraries(${PROJECT_NAME} miniFlowVR)
    endif()
endif()


if(Sofa.GUI_FOUND)
    target_link_libraries(${PROJECT_NAME} Sofa.GUI)
endif()


if(Sofa.GUI.Qt_FOUND)
    target_link_libraries(${PROJECT_NAME} Sofa.GUI.Qt)
endif()

if(SofaValidation_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaValidation)
endif()

if(VolumetricRendering_FOUND)
    target_link_libraries(${PROJECT_NAME} VolumetricRendering)
endif()
if(SofaSphFluid_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaSphFluid)
endif()

if(SOFACUDA_CUBLAS)
    cuda_add_cublas_to_target(${PROJECT_NAME})
    target_link_libraries(${PROJECT_NAME} ${CUDA_SPARSE_LIBRARY})
endif()
if(SOFACUDA_CUDPP)
    target_link_libraries(${PROJECT_NAME} cudpp)
endif()



## SofaCUDANvccFlags.cmake
# Build tree
configure_file(SofaCUDANvccFlags.cmake ${CMAKE_BINARY_DIR}/lib/cmake/SofaCUDANvccFlags.cmake COPYONLY)
# Install tree
install(FILES SofaCUDANvccFlags.cmake DESTINATION lib/cmake/SofaCUDA)

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${SOFACUDA_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
