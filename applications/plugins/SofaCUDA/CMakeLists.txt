cmake_minimum_required(VERSION 3.22)
project(SofaCUDA LANGUAGES CUDA CXX)

set(SOFACUDA_MAJOR_VERSION 0)
set(SOFACUDA_MINOR_VERSION 1)
set(SOFACUDA_VERSION ${SOFACUDA_MAJOR_VERSION}.${SOFACUDA_MINOR_VERSION})

if(DEFINED CUDA_ARCH_LIST)
    message(WARNING "CUDA_ARCH_LIST variable is deprecated ; use cmake built-in variable CMAKE_CUDA_ARCHITECTURES instead (see https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html)")
    string(REPLACE "." "" CUDA_ARCH_LIST_REFORMATED "${CUDA_ARCH_LIST}")
    message(WARNING "${CUDA_ARCH_LIST_REFORMATED} will be set in CMAKE_CUDA_ARCHITECTURES.")
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST_REFORMATED})
endif()

if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0") 
    option(SOFACUDA_ENABLE_NATIVE_ARCHITECTURE "Set native as for CUDA_ARCHITECTURES (which will compile compatible architectures for the current system)")
    if(SOFACUDA_ENABLE_NATIVE_ARCHITECTURE)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
endif()

# set 75 as fallback value
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(NOTICE "CMAKE_CUDA_ARCHITECTURES is not set, it will be set by default to 75")
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(SOFACUDACORE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(SOFACUDA_SOURCE_DIR "${SOFACUDACORE_SOURCE_DIR}/SofaCUDA")

set(HEADER_FILES
    ${SOFACUDA_SOURCE_DIR}/config.h.in
    ${SOFACUDA_SOURCE_DIR}/init.h
    
    ### Common
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaBaseVector.h
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaCommon.h
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaContactMapper.h
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaMath.h
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaMath.inl
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaMathRigid.h
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaMathRigid.inl
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaMatrix.h
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaMemoryManager.h
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaScan.h
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaSort.h
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaTypes.h
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/mycuda.h

    ### Mechanical
    ${SOFACUDA_SOURCE_DIR}/component/statecontainer/CudaMechanicalObject.h
    ${SOFACUDA_SOURCE_DIR}/component/statecontainer/CudaMechanicalObject.inl
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaSingleStateAccessor.h

    ### Mappings
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaBarycentricMapping.h
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaBarycentricMapping.inl
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaBarycentricMappingRigid.h
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaBeamLinearMapping.h
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaIdentityMapping.h
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaIdentityMapping.inl
    ${SOFACUDA_SOURCE_DIR}/component/mapping/nonlinear/CudaRigidMapping.h
    ${SOFACUDA_SOURCE_DIR}/component/mapping/nonlinear/CudaRigidMapping.inl
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaSubsetMapping.h
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaSubsetMapping.inl
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaSubsetMultiMapping.h


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
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaHexahedronTLEDForceField.h
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/hyperelastic/CudaStandardTetrahedralFEMForceField.h
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/hyperelastic/CudaStandardTetrahedralFEMForceField.inl
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/tensormass/CudaTetrahedralTensorMassForceField.h
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/tensormass/CudaTetrahedralTensorMassForceField.inl
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaTetrahedronFEMForceField.h
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaTetrahedronFEMForceField.inl
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaTetrahedronTLEDForceField.h
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaTriangularFEMForceFieldOptim.h
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaTriangularFEMForceFieldOptim.inl

    ### ForceFields
    ${SOFACUDA_SOURCE_DIR}/component/mechanicalload/CudaConstantForceField.h
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
    ${SOFACUDA_SOURCE_DIR}/init.cpp
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaBaseVector.cpp
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaContactMapper.cpp
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/mycuda.cpp

    ### Mechanical
    ${SOFACUDA_SOURCE_DIR}/component/statecontainer/CudaMechanicalObject.cpp
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaSetTopology.cpp
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaSingleStateAccessor.cpp

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
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaMultiMapping.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mapping/nonlinear/CudaRigidMapping.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaSubsetMapping.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mapping/linear/CudaSubsetMultiMapping.cpp

    ### Mass
    ${SOFACUDA_SOURCE_DIR}/component/mass/CudaDiagonalMass.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mass/CudaMeshMatrixMass.cpp
    ${SOFACUDA_SOURCE_DIR}/component/mass/CudaUniformMass.cpp

    ### FEM
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaHexahedronFEMForceField.cpp
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaHexahedronTLEDForceField.cpp
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/hyperelastic/CudaStandardTetrahedralFEMForceField.cpp
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/tensormass/CudaTetrahedralTensorMassForceField.cpp
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaTetrahedronFEMForceField.cpp
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaTetrahedronTLEDForceField.cpp
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
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaCollision.cpp
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
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/mycuda.cu
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaBaseVector.cu
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaContactMapper.cu
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaScan.cu
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaSort.cu

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
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaHexahedronTLEDForceField.cu
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/hyperelastic/CudaStandardTetrahedralFEMForceField.cu
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/tensormass/CudaTetrahedralTensorMassForceField.cu
    ${SOFACUDA_SOURCE_DIR}/component/solidmechanics/fem/elastic/CudaTetrahedronFEMForceField.cu
    ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaTetrahedronTLEDForceField.cu
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
    message("OpenGL-related code won't be enabled (Sharing OpenGL buffers with CUDA)")
else()
    list(APPEND HEADER_FILES
        ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaVisualModel.h
        ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaVisualModel.inl
    )
    list(APPEND SOURCE_FILES
       ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaVisualModel.cpp
    )
    list(APPEND CUDA_SOURCES
        ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaVisualModel.cu
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

sofa_find_package(Sofa.GUI QUIET)
if(Sofa.GUI_FOUND)
    list(APPEND SOURCE_FILES ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/gui/CudaMouseInteraction.cpp)
endif()

sofa_find_package(Sofa.Qt QUIET)
if(Sofa.Qt_FOUND)
    list(APPEND HEADER_FILES ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/gui/CudaDataWidget.h)
    list(APPEND SOURCE_FILES ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/gui/CudaDataWidget.cpp)
endif()

sofa_find_package(SofaValidation QUIET)
if(SofaValidation_FOUND)
    list(APPEND SOURCE_FILES
        ${SOFACUDACORE_SOURCE_DIR}/sofa/gpu/cuda/CudaExtraMonitor.cpp
        )
    message(STATUS "SofaCUDA: found SofaValidation, enabling CudaExtraMonitor (Warning: this component is being deprecated). ")
endif()

option(SOFACUDA_VERBOSE_PTXAS "???" OFF)

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

sofa_find_package(CUDAToolkit 10.0 REQUIRED)

# nvcc uses a "host code compiler" to compile CPU code, specified by CUDA_HOST_COMPILER.
# With some versions of CMake, CUDA_HOST_COMPILER defaults to CMAKE_C_COMPILER,
# but few host compilers are actually supported. Workarounds should go here.
if (${CUDA_HOST_COMPILER} MATCHES "ccache$")
    message(STATUS "SofaCUDA: CUDA host compiler was set to ccache, changing to g++")
    set(CUDA_HOST_COMPILER "g++" CACHE STRING "Host side compiler used by NVCC" FORCE)
endif()

# quick and dirty fix for nvcc compatibility with -fno-partial-inlining flag
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${CUDA_SOURCES} ${README_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFACUDA_COMPILE_DEFINITIONS}")
if(SOFACUDA_VERBOSE_PTXAS)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
endif()
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /Zc:__cplusplus>)
endif()
if(WIN32)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-DWIN32>)
endif()

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

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17 cuda_std_17)
target_link_libraries(${PROJECT_NAME} CUDA::cudart)

if(Sofa.GL_FOUND)
    target_link_libraries(${PROJECT_NAME} Sofa.GL)
endif()

if(Sofa.GUI_FOUND)
    target_link_libraries(${PROJECT_NAME} Sofa.GUI)
endif()


if(Sofa.Qt_FOUND)
    target_link_libraries(${PROJECT_NAME} Sofa.Qt)
endif()

if(SofaValidation_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaValidation)
endif()

if(SOFACUDA_CUBLAS)
    target_link_libraries(${PROJECT_NAME} ${CUDA_cusparse_LIBRARY} CUDA::cublas)

endif()
if(SOFACUDA_CUDPP)
    target_link_libraries(${PROJECT_NAME} cudpp)
endif()

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${SOFACUDA_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    INCLUDE_SOURCE_DIR "src"
    RELOCATABLE "plugins"
)
