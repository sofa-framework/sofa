/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_Elasticity_test_INL
#define SOFA_Elasticity_test_INL

#include "Elasticity_test.h"

// Solvers
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/odesolver/StaticSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>

// Box roi
#include <sofa/component/engine/PairBoxRoi.h>
#include <sofa/component/engine/BoxROI.h>
#include <sofa/component/engine/GenerateCylinder.h>

// Constraint
#include <sofa/component/projectiveconstraintset/ProjectToLineConstraint.h>
#include <sofa/component/projectiveconstraintset/FixedConstraint.h>
#include <sofa/component/projectiveconstraintset/AffineMovementConstraint.h>
#include <sofa/component/projectiveconstraintset/FixedPlaneConstraint.h>

// ForceField
#include <sofa/component/forcefield/TrianglePressureForceField.h>

#include <sofa/component/visualmodel/VisualStyle.h>
#include <sofa/component/topology/RegularGridTopology.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>

namespace sofa
{
using namespace simulation;
using namespace component::odesolver;
using namespace component::topology;
typedef component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;

/// Create a scene with a regular grid and an affine constraint for patch test

template<class DataTypes> PatchTestStruct<DataTypes>
Elasticity_test<DataTypes>::createRegularGridScene(
        Node::SPtr root,
        Coord startPoint,
        Coord endPoint,
        int numX,
        int numY,
        int numZ,
        Vec<6,SReal> entireBoxRoi,
        Vec<6,SReal> inclusiveBox,
        Vec<6,SReal> includedBox)
{
    // Definitions
    PatchTestStruct<DataTypes> patchStruct;
    typedef typename DataTypes::Real Real;
    typedef typename component::container::MechanicalObject<DataTypes> MechanicalObject;
    typedef typename sofa::component::mass::UniformMass <DataTypes, Real> UniformMass;
    typedef component::topology::RegularGridTopology RegularGridTopology;
    typedef typename component::engine::BoxROI<DataTypes> BoxRoi;
    typedef typename sofa::component::engine::PairBoxROI<DataTypes> PairBoxRoi;
    typedef typename component::projectiveconstraintset::AffineMovementConstraint<DataTypes> AffineMovementConstraint;
    typedef component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;

    // Root node
    root->setGravity( Coord3(0,0,0) );
    root->setAnimate(false);
    root->setDt(0.05);

    // Node square
    simulation::Node::SPtr SquareNode = root->createChild("Square");

    // Euler implicit solver and cglinear solver
    component::odesolver::EulerImplicitSolver::SPtr solver = addNew<component::odesolver::EulerImplicitSolver>(SquareNode,"EulerImplicitSolver");
    solver->f_rayleighStiffness.setValue(0.5);
    solver->f_rayleighMass.setValue(0.5);
    CGLinearSolver::SPtr cgLinearSolver = addNew< CGLinearSolver >(SquareNode,"linearSolver");

    // Mass
    typename UniformMass::SPtr mass = addNew<UniformMass>(SquareNode,"mass");

    // Regular grid topology
    RegularGridTopology::SPtr gridMesh = addNew<RegularGridTopology>(SquareNode,"loader");
    gridMesh->setNumVertices(numX,numY,numZ);
    gridMesh->setPos(startPoint[0],endPoint[0],startPoint[1],endPoint[1],startPoint[2],endPoint[2]);

    //Mechanical object
    patchStruct.dofs = addNew<MechanicalObject>(SquareNode,"mechanicalObject");
    patchStruct.dofs->setName("mechanicalObject");
    patchStruct.dofs->setSrc("@"+gridMesh->getName(), gridMesh.get());

    //BoxRoi to find all mesh points
    helper::vector < defaulttype::Vec<6,Real> > vecBox;
    vecBox.push_back(entireBoxRoi);
    typename BoxRoi::SPtr boxRoi = addNew<BoxRoi>(SquareNode,"boxRoi");
    boxRoi->boxes.setValue(vecBox);

    //PairBoxRoi to define the constrained points = points of the border
    typename PairBoxRoi::SPtr pairBoxRoi = addNew<PairBoxRoi>(SquareNode,"pairBoxRoi");
    pairBoxRoi->inclusiveBox.setValue(inclusiveBox);
    pairBoxRoi->includedBox.setValue(includedBox);

    //Affine constraint
    patchStruct.affineConstraint  = addNew<AffineMovementConstraint>(SquareNode,"affineConstraint");
    setDataLink(&boxRoi->f_indices,&patchStruct.affineConstraint->m_meshIndices);
    setDataLink(&pairBoxRoi->f_indices,& patchStruct.affineConstraint->m_indices);

    patchStruct.SquareNode = SquareNode;
    return patchStruct;
}

template<class DataTypes>
CylinderTractionStruct<DataTypes>  Elasticity_test<DataTypes>::createCylinderTractionScene(
        int resolutionCircumferential,
        int resolutionRadial,
        int resolutionHeight,
        int maxIter)
{
    // Definitions
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef typename component::container::MechanicalObject<DataTypes> MechanicalObject;
    typedef typename component::engine::BoxROI<DataTypes> BoxRoi;
    typedef component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;
    typename simulation::Node::SPtr root;
    CylinderTractionStruct<DataTypes> tractionStruct;

    // Root node
    root = sofa::simulation::getSimulation()->createNewGraph("root");
    tractionStruct.root=root;

    root->setGravity( Coord3(0,0,0) );
    root->setAnimate(false);
    root->setDt(0.05);


    // GenerateCylinder object
    typename sofa::component::engine::GenerateCylinder<DataTypes>::SPtr eng= sofa::modeling::addNew<sofa::component::engine::GenerateCylinder<DataTypes> >(root,"cylinder");
    eng->f_radius=0.2;
    eng->f_height=1.0;
    eng->f_resolutionCircumferential=resolutionCircumferential;
    eng->f_resolutionRadial=resolutionRadial;
    eng->f_resolutionHeight=resolutionHeight;
    // TetrahedronSetTopologyContainer object
    typename sofa::component::topology::TetrahedronSetTopologyContainer::SPtr container1= sofa::modeling::addNew<sofa::component::topology::TetrahedronSetTopologyContainer>(root,"Container1");
    sofa::modeling::setDataLink(&eng->f_tetrahedron,&container1->d_tetrahedron);
    sofa::modeling::setDataLink(&eng->f_outputX,&container1->d_initPoints);
    container1->d_createTriangleArray=true;
    // TetrahedronSetGeometryAlgorithms object
    typename sofa::component::topology::TetrahedronSetGeometryAlgorithms<DataTypes>::SPtr geo1= sofa::modeling::addNew<sofa::component::topology::TetrahedronSetGeometryAlgorithms<DataTypes> >(root);

    // CGLinearSolver
    typename CGLinearSolver::SPtr cgLinearSolver = addNew< CGLinearSolver >(root,"linearSolver");
    cgLinearSolver->f_maxIter=maxIter;
    cgLinearSolver->f_tolerance =1e-9;
    cgLinearSolver->f_smallDenominatorThreshold=1e-9;
    // StaticSolver
    typename component::odesolver::StaticSolver::SPtr solver = addNew<component::odesolver::StaticSolver>(root,"StaticSolver");
    // mechanicalObject object
    typename MechanicalObject::SPtr meca1= sofa::modeling::addNew<MechanicalObject>(root);
    sofa::modeling::setDataLink(&eng->f_outputX,&meca1->x);
    tractionStruct.dofs=meca1;
    // MeshMatrixMass
    typename sofa::component::mass::MeshMatrixMass<DataTypes,Real>::SPtr mass= sofa::modeling::addNew<sofa::component::mass::MeshMatrixMass<DataTypes,Real> >(root,"BezierMass");
    mass->m_massDensity=1.0;
    mass->lumping=false;
    /// box fixed
    helper::vector < defaulttype::Vec<6,Real> > vecBox;
    defaulttype::Vec<6,Real> box;
    box[0]= -0.01;box[1]= -0.01;box[2]= -0.01;box[3]= 0.01;box[4]= 0.01;box[5]= 0.01;
    vecBox.push_back(box);
    typename BoxRoi::SPtr boxRoi1 = addNew<BoxRoi>(root,"boxRoiFix");
    boxRoi1->boxes.setValue(vecBox);
    // FixedConstraint
    typename component::projectiveconstraintset::FixedConstraint<DataTypes>::SPtr fc=
        addNew<typename component::projectiveconstraintset::FixedConstraint<DataTypes> >(root);
    sofa::modeling::setDataLink(&boxRoi1->f_indices,&fc->f_indices);
    // FixedPlaneConstraint
    typename component::projectiveconstraintset::FixedPlaneConstraint<DataTypes>::SPtr fpc=
        addNew<typename component::projectiveconstraintset::FixedPlaneConstraint<DataTypes> >(root);
    fpc->dmin= -0.01;
    fpc->dmax= 0.01;
    fpc->direction=Coord(0,0,1);
    /// box pressure
    box[0]= -0.2;box[1]= -0.2;box[2]= 0.99;box[3]= 0.2;box[4]= 0.2;box[5]= 1.01;
    vecBox[0]=box;
    typename BoxRoi::SPtr boxRoi2 = addNew<BoxRoi>(root,"boxRoiPressure");
    boxRoi2->boxes.setValue(vecBox);
    boxRoi2->f_computeTriangles=true;
    /// TrianglePressureForceField
    typename component::forcefield::TrianglePressureForceField<DataTypes>::SPtr tpff=
        addNew<typename component::forcefield::TrianglePressureForceField<DataTypes> >(root);
    tractionStruct.forceField=tpff;
    sofa::modeling::setDataLink(&boxRoi2->f_triangleIndices,&tpff->triangleList);
    // ProjectToLineConstraint
    typename component::projectiveconstraintset::ProjectToLineConstraint<DataTypes>::SPtr ptlc=
        addNew<typename component::projectiveconstraintset::ProjectToLineConstraint<DataTypes> >(root);
    ptlc->f_direction=Coord(1,0,0);
    ptlc->f_origin=Coord(0,0,0);
    sofa::helper::vector<unsigned> vArray;
    vArray.push_back(resolutionCircumferential*(resolutionRadial-1)+1);
    ptlc->f_indices.setValue(vArray);

    return tractionStruct;
}


/// Create an assembly of a siff hexahedral grid with other objects
template<typename DT>
simulation::Node::SPtr Elasticity_test<DT>::createGridScene(
        Coord  startPoint,
        Coord endPoint,
        int numX,
        int numY,
        int numZ,
        SReal totalMass,
        SReal stiffnessValue,
        SReal dampingRatio )
{
    using helper::vector;

    // The graph root node
    Node::SPtr  root = simulation::getSimulation()->createNewGraph("root");
    root->setGravity( Coord3(0,-10,0) );
    root->setAnimate(false);
    root->setDt(0.01);
    component::visualmodel::addVisualStyle(root)->setShowVisual(false).setShowCollision(false).setShowMapping(true).setShowBehavior(true);

    Node::SPtr simulatedScene = root->createChild("simulatedScene");

    EulerImplicitSolver::SPtr eulerImplicitSolver = New<EulerImplicitSolver>();
    simulatedScene->addObject( eulerImplicitSolver );
    CGLinearSolver::SPtr cgLinearSolver = New<CGLinearSolver>();
    simulatedScene->addObject(cgLinearSolver);

    // The rigid object
    Node::SPtr rigidNode = simulatedScene->createChild("rigidNode");
    MechanicalObjectRigid3::SPtr rigid_dof = addNew<MechanicalObjectRigid3>(rigidNode, "dof");
    UniformMassRigid3::SPtr rigid_mass = addNew<UniformMassRigid3>(rigidNode,"mass");
    FixedConstraintRigid3::SPtr rigid_fixedConstraint = addNew<FixedConstraintRigid3>(rigidNode,"fixedConstraint");

    // Particles mapped to the rigid object
    Node::SPtr mappedParticles = rigidNode->createChild("mappedParticles");
    MechanicalObject3::SPtr mappedParticles_dof = addNew< MechanicalObject3>(mappedParticles,"dof");
    RigidMappingRigid3_to_3::SPtr mappedParticles_mapping = addNew<RigidMappingRigid3_to_3>(mappedParticles,"mapping");
    mappedParticles_mapping->setModels( rigid_dof.get(), mappedParticles_dof.get() );

    // The independent particles
    Node::SPtr independentParticles = simulatedScene->createChild("independentParticles");
    MechanicalObject3::SPtr independentParticles_dof = addNew< MechanicalObject3>(independentParticles,"dof");

    // The deformable grid, connected to its 2 parents using a MultiMapping
    Node::SPtr deformableGrid = independentParticles->createChild("deformableGrid"); // first parent
    mappedParticles->addChild(deformableGrid);                                       // second parent

    RegularGridTopology::SPtr deformableGrid_grid = addNew<RegularGridTopology>( deformableGrid, "grid" );
    deformableGrid_grid->setNumVertices(numX,numY,numZ);
    deformableGrid_grid->setPos(startPoint[0],endPoint[0],startPoint[1],endPoint[1],startPoint[2],endPoint[2]);

    MechanicalObject3::SPtr deformableGrid_dof = addNew< MechanicalObject3>(deformableGrid,"dof");

    SubsetMultiMapping3_to_3::SPtr deformableGrid_mapping = addNew<SubsetMultiMapping3_to_3>(deformableGrid,"mapping");
    deformableGrid_mapping->addInputModel(independentParticles_dof.get()); // first parent
    deformableGrid_mapping->addInputModel(mappedParticles_dof.get());      // second parent
    deformableGrid_mapping->addOutputModel(deformableGrid_dof.get());

    UniformMass3::SPtr mass = addNew<UniformMass3>(deformableGrid,"mass" );
    mass->mass.setValue( totalMass/(numX*numY*numZ) );

    RegularGridSpringForceField3::SPtr spring = addNew<RegularGridSpringForceField3>(deformableGrid, "spring");
    spring->setLinesStiffness(stiffnessValue);
    spring->setQuadsStiffness(stiffnessValue);
    spring->setCubesStiffness(stiffnessValue);
    spring->setLinesDamping(dampingRatio);


    // ======  Set up the multimapping and its parents, based on its child
    // initialize the grid, so that the particles are located in space
    deformableGrid_grid->init();
    deformableGrid_dof->init();
    //    cerr<<"SimpleSceneCreator::createGridScene size = "<< deformableGrid_dof->getSize() << endl;
    MechanicalObject3::ReadVecCoord  xgrid = deformableGrid_dof->readPositions();
    //    cerr<<"SimpleSceneCreator::createGridScene xgrid = " << xgrid << endl;


    // create the rigid frames and their bounding boxes
    size_t numRigid = 2;
    vector<BoundingBox> boxes(numRigid);
    vector< vector<size_t> > indices(numRigid); // indices of the particles in each box
    double eps = (endPoint[0]-startPoint[0])/(numX*2);

    // first box, x=xmin
    boxes[0] = BoundingBox(Vec3d(startPoint[0]-eps, startPoint[1]-eps, startPoint[2]-eps),
                           Vec3d(startPoint[0]+eps,   endPoint[1]+eps,   endPoint[2]+eps));

    // second box, x=xmax
    boxes[1] = BoundingBox(Vec3d(endPoint[0]-eps, startPoint[1]-eps, startPoint[2]-eps),
                           Vec3d(endPoint[0]+eps,   endPoint[1]+eps,   endPoint[2]+eps));
    rigid_dof->resize(numRigid);
    MechanicalObjectRigid3::WriteVecCoord xrigid = rigid_dof->writePositions();
    xrigid[0].getCenter()=Coord(startPoint[0], 0.5*(startPoint[1]+endPoint[1]), 0.5*(startPoint[2]+endPoint[2]));
    xrigid[1].getCenter()=Coord(  endPoint[0], 0.5*(startPoint[1]+endPoint[1]), 0.5*(startPoint[2]+endPoint[2]));

    // find the particles in each box
    vector<bool> isFree(xgrid.size(),true);
    size_t numMapped = 0;
    for(size_t i=0; i<xgrid.size(); i++){
        for(size_t b=0; b<numRigid; b++ )
        {
            if( isFree[i] && boxes[b].contains(xgrid[i]) )
            {
                indices[b].push_back(i); // associate the particle with the box
                isFree[i] = false;
                numMapped++;
            }
        }
    }

    // distribute the particles to the different solids. One solid for each box.
    mappedParticles_dof->resize(numMapped);
    independentParticles_dof->resize( numX*numY*numZ - numMapped );
    MechanicalObject3::WriteVecCoord xmapped = mappedParticles_dof->writePositions();
    mappedParticles_mapping->globalToLocalCoords.setValue(true); // to define the mapped positions in world coordinates
    MechanicalObject3::WriteVecCoord xindependent = independentParticles_dof->writePositions();
    vector< pair<MechanicalObject3*,size_t> > parentParticles(xgrid.size());

    // independent particles
    size_t independentIndex=0;
    for( size_t i=0; i<xgrid.size(); i++ ){
        if( isFree[i] ){
            parentParticles[i]=make_pair(independentParticles_dof.get(),independentIndex);
            xindependent[independentIndex] = xgrid[i];
            independentIndex++;
        }
    }

    // mapped particles
    size_t mappedIndex=0;
    vector<unsigned>* pointsPerFrame = mappedParticles_mapping->pointsPerFrame.beginEdit();
    for( size_t b=0; b<numRigid; b++ )
    {
        const vector<size_t>& ind = indices[b];
        pointsPerFrame->push_back(ind.size()); // tell the mapping the number of points associated with this frame
        for(size_t i=0; i<ind.size(); i++)
        {
            parentParticles[ind[i]]=make_pair(mappedParticles_dof.get(),mappedIndex);
            xmapped[mappedIndex] = xgrid[ ind[i] ];
            mappedIndex++;

        }
    }
    mappedParticles_mapping->pointsPerFrame.endEdit();

    // now add all the particles to the multimapping
    for( size_t i=0; i<xgrid.size(); i++ )
    {
        deformableGrid_mapping->addPoint( parentParticles[i].first, parentParticles[i].second );
    }


    return root;

}

template<class DataTypes>
simulation::Node::SPtr Elasticity_test<DataTypes>::createMassSpringSystem(
        simulation::Node::SPtr root,
        double stiffness,
        double mass,
        double restLength,
        VecCoord xFixedPoint,
        VecDeriv vFixedPoint,
        VecCoord xMass,
        VecDeriv vMass)
{

// Fixed point
simulation::Node::SPtr fixedPointNode = root->createChild("FixedPointNode");
MechanicalObject3::SPtr FixedPoint = addNew<MechanicalObject3>(fixedPointNode,"fixedPoint");

// Set position and velocity
FixedPoint->resize(1);
MechanicalObject3::WriteVecCoord xdof = FixedPoint->writePositions();
copyToData( xdof, xFixedPoint );
MechanicalObject3::WriteVecDeriv vdof = FixedPoint->writeVelocities();
copyToData( vdof, vFixedPoint );

FixedConstraint3::SPtr fixed = modeling::addNew<FixedConstraint3>(fixedPointNode,"FixedPointNode");
fixed->addConstraint(0);      // attach particle


// Mass
simulation::Node::SPtr massNode = root->createChild("MassNode");
MechanicalObject3::SPtr massDof = addNew<MechanicalObject3>(massNode,"massNode");

// Set position and velocity
FixedPoint->resize(1);
MechanicalObject3::WriteVecCoord xMassDof = massDof->writePositions();
copyToData( xMassDof, xMass );
MechanicalObject3::WriteVecDeriv vMassDof = massDof->writeVelocities();
copyToData( vMassDof, vMass );

UniformMass3::SPtr massPtr = addNew<UniformMass3>(massNode,"mass");
massPtr->totalMass.setValue( mass );

// attach a spring
StiffSpringForceField3::SPtr spring = New<StiffSpringForceField3>(FixedPoint.get(), massDof.get());
root->addObject(spring);
spring->addSpring(0,0,stiffness ,0, restLength);

return root;

}


}// sofa

#endif

