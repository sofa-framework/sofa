/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaCUDA/config.h>
#include <SofaCUDA/init.h>
#include <sofa/gpu/cuda/mycuda.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::gpu::cuda
{

// component::collision::geometry
extern void registerLineCollisionModel(sofa::core::ObjectFactory* factory);
extern void registerPointCollisionModel(sofa::core::ObjectFactory* factory);
extern void registerSphereCollisionModel(sofa::core::ObjectFactory* factory);
extern void registerTriangleCollisionModel(sofa::core::ObjectFactory* factory);

// component::collision::response::contact
extern void registerPenalityContactForceField(sofa::core::ObjectFactory* factory);

// component::constraint::lagrangian::correction
extern void registerLinearSolverConstraintCorrection(sofa::core::ObjectFactory* factory);
extern void registerPrecomputedConstraintCorrection(sofa::core::ObjectFactory* factory);
extern void registerUncoupledConstraintCorrection(sofa::core::ObjectFactory* factory);

// component::constraint::lagrangian::model
extern void registerBilateralLagrangianConstraint(sofa::core::ObjectFactory* factory);

// component::constraint::projective
extern void registerFixedProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerFixedTranslationProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerLinearMovementProjectiveConstraint(sofa::core::ObjectFactory* factory);
extern void registerLinearVelocityProjectiveConstraint(sofa::core::ObjectFactory* factory);

// component::engine::select
extern void registerBoxROI(sofa::core::ObjectFactory* factory);
extern void registerNearestPointROI(sofa::core::ObjectFactory* factory);
extern void registerSphereROI(sofa::core::ObjectFactory* factory);

// component::engine::transform
extern void registerIndexValueMapper(sofa::core::ObjectFactory* factory);

// component::mapping::linear
extern void registerBarycentricMapping(sofa::core::ObjectFactory* factory);
extern void registerBarycentricMapping_f(sofa::core::ObjectFactory* factory);
extern void registerBarycentricMapping_3fRigid(sofa::core::ObjectFactory* factory);
extern void registerBarycentricMapping_3f(sofa::core::ObjectFactory* factory);
extern void registerBarycentricMapping_3f1(sofa::core::ObjectFactory* factory);
extern void registerBarycentricMapping_3f1_f(sofa::core::ObjectFactory* factory);
extern void registerBarycentricMapping_3f1_3f(sofa::core::ObjectFactory* factory);
extern void registerBarycentricMapping_3f1_d(sofa::core::ObjectFactory* factory);
extern void registerBeamLinearMapping(sofa::core::ObjectFactory* factory);
extern void registerIdentityMapping(sofa::core::ObjectFactory* factory);
extern void registerSubsetMapping(sofa::core::ObjectFactory* factory);
extern void registerSubsetMultiMapping(sofa::core::ObjectFactory* factory);

// component::mapping::nonlinear
extern void registerRigidMapping(sofa::core::ObjectFactory* factory);

// component::mass
extern void registerDiagonalMass(sofa::core::ObjectFactory* factory);
extern void registerMeshMatrixMass(sofa::core::ObjectFactory* factory);
extern void registerUniformMass(sofa::core::ObjectFactory* factory);

// component::mechanicalload
extern void registerConstantForceField(sofa::core::ObjectFactory* factory);
extern void registerEllipsoidForceField(sofa::core::ObjectFactory* factory);
extern void registerLinearForceField(sofa::core::ObjectFactory* factory);
extern void registerPlaneForceField(sofa::core::ObjectFactory* factory);
extern void registerSphereForceField(sofa::core::ObjectFactory* factory);

// component::solidmechanics::fem::elastic
extern void registerHexahedronFEMForceField(sofa::core::ObjectFactory* factory);
extern void registerTetrahedronFEMForceField(sofa::core::ObjectFactory* factory);
extern void registerTriangularFEMForceFieldOptim(sofa::core::ObjectFactory* factory);

// component::solidmechanics::fem::hyperelastic
extern void registerStandardTetrahedralFEMForceField(sofa::core::ObjectFactory* factory);

// component::solidmechanics::spring
extern void registerMeshSpringForceField(sofa::core::ObjectFactory* factory);
extern void registerQuadBendingSprings(sofa::core::ObjectFactory* factory);
extern void registerRestShapeSpringsForceField(sofa::core::ObjectFactory* factory);
extern void registerSpringForceField(sofa::core::ObjectFactory* factory);
extern void registerTriangleBendingSprings(sofa::core::ObjectFactory* factory);

// component::solidmechanics::tensormass
extern void registerTetrahedralTensorMassForceField(sofa::core::ObjectFactory* factory);

// component::statecontainer
extern void registerMechanicalObject(sofa::core::ObjectFactory* factory);

// component::visualmodel
extern void registerVisualModel(sofa::core::ObjectFactory* factory);

// component::collision::detection::intersection
extern void registerProximityIntersection(sofa::core::ObjectFactory* factory);

// component::topology::container::dynamic
extern void registerPointSetGeometryAlgorithms(sofa::core::ObjectFactory* factory);
extern void registerEdgeSetGeometryAlgorithms(sofa::core::ObjectFactory* factory);
extern void registerTriangleSetGeometryAlgorithms(sofa::core::ObjectFactory* factory);
extern void registerQuadSetGeometryAlgorithms(sofa::core::ObjectFactory* factory);
extern void registerTetrahedronSetGeometryAlgorithms(sofa::core::ObjectFactory* factory);
extern void registerHexahedronSetGeometryAlgorithms(sofa::core::ObjectFactory* factory);

// gui::component::performer;
extern void registerMouseInteractor(sofa::core::ObjectFactory* factory);

// SofaCUDA
extern void registerTetrahedronTLEDForceField(sofa::core::ObjectFactory* factory);
extern void registerHexahedronTLEDForceField(sofa::core::ObjectFactory* factory);

//Here are just several convenient functions to help user to know what contains the plugin
extern "C" {
SOFA_GPU_CUDA_API void initExternalModule();
SOFA_GPU_CUDA_API const char* getModuleName();
SOFA_GPU_CUDA_API const char* getModuleVersion();
SOFA_GPU_CUDA_API const char* getModuleLicense();
SOFA_GPU_CUDA_API const char* getModuleDescription();
SOFA_GPU_CUDA_API bool moduleIsInitialized();
SOFA_GPU_CUDA_API void registerObjects(sofa::core::ObjectFactory* factory);
}

bool isModuleInitialized = false;

void init()
{
    static bool first = true;
    if (first)
    {
        isModuleInitialized = sofa::gpu::cuda::mycudaInit();
        first = false;
    }
}

void initExternalModule()
{
    init();
}

const char* getModuleName()
{
    return MODULE_NAME;
}

const char* getModuleVersion()
{
    return MODULE_VERSION;
}

const char* getModuleLicense()
{
    return "LGPL";
}

const char* getModuleDescription()
{
    return "GPU-based computing using NVIDIA CUDA";
}

bool moduleIsInitialized()
{
    return isModuleInitialized;
}

void registerObjects(sofa::core::ObjectFactory* factory)
{
    registerLineCollisionModel(factory);
    registerPointCollisionModel(factory);
    registerSphereCollisionModel(factory);
    registerTriangleCollisionModel(factory);
    registerPenalityContactForceField(factory);
    registerLinearSolverConstraintCorrection(factory);
    registerPrecomputedConstraintCorrection(factory);
    registerUncoupledConstraintCorrection(factory);
    registerBilateralLagrangianConstraint(factory);
    registerFixedProjectiveConstraint(factory);
    registerFixedTranslationProjectiveConstraint(factory);
    registerLinearMovementProjectiveConstraint(factory);
    registerLinearVelocityProjectiveConstraint(factory);
    registerBoxROI(factory);
    registerNearestPointROI(factory);
    registerSphereROI(factory);
    registerIndexValueMapper(factory);
    registerBarycentricMapping(factory);
    registerBarycentricMapping_f(factory);
    registerBarycentricMapping_3fRigid(factory);
    registerBarycentricMapping_3f(factory);
    registerBarycentricMapping_3f1(factory);
    registerBarycentricMapping_3f1_f(factory);
    registerBarycentricMapping_3f1_3f(factory);
    registerBarycentricMapping_3f1_d(factory);
    registerBeamLinearMapping(factory);
    registerIdentityMapping(factory);
    registerSubsetMapping(factory);
    registerSubsetMultiMapping(factory);
    registerRigidMapping(factory);
    registerDiagonalMass(factory);
    registerMeshMatrixMass(factory);
    registerUniformMass(factory);
    registerConstantForceField(factory);
    registerEllipsoidForceField(factory);
    registerLinearForceField(factory);
    registerPlaneForceField(factory);
    registerSphereForceField(factory);
    registerHexahedronFEMForceField(factory);
    registerTetrahedronFEMForceField(factory);
    registerTriangularFEMForceFieldOptim(factory);
    registerStandardTetrahedralFEMForceField(factory);
    registerMeshSpringForceField(factory);
    registerQuadBendingSprings(factory);
    registerRestShapeSpringsForceField(factory);
    registerSpringForceField(factory);
    registerTriangleBendingSprings(factory);
    registerTetrahedralTensorMassForceField(factory);
    registerMechanicalObject(factory);
    registerVisualModel(factory);
    registerProximityIntersection(factory);
    registerPointSetGeometryAlgorithms(factory);
    registerEdgeSetGeometryAlgorithms(factory);
    registerTriangleSetGeometryAlgorithms(factory);
    registerQuadSetGeometryAlgorithms(factory);
    registerTetrahedronSetGeometryAlgorithms(factory);
    registerHexahedronSetGeometryAlgorithms(factory);
    registerMouseInteractor(factory);
}

}
