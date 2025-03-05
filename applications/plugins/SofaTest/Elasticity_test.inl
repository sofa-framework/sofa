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
#ifndef SOFA_Elasticity_test_INL
#define SOFA_Elasticity_test_INL

#include "Elasticity_test.h"

// Solvers
#include <sofa/component/odesolver/backward/EulerImplicitSolver.h>
#include <sofa/component/odesolver/backward/StaticSolver.h>
#include <sofa/component/linearsolver/iterative/CGLinearSolver.h>

// Box roi
#include <sofa/component/engine/select/BoxROI.h>
#include <sofa/component/engine/select/PairBoxRoi.h>
#include <sofa/component/engine/generate/GenerateCylinder.h>


// Constraint
#include <sofa/component/constraint/projective/LineProjectiveConstraint.h>
#include <sofa/component/constraint/projective/FixedConstraint.h>
#include <sofa/component/constraint/projective/AffineMovementProjectiveConstraint.h>
#include <sofa/component/constraint/projective/FixedPlaneProjectiveConstraint.h>

// ForceField
#include <sofa/component/mechanicalload/TrianglePressureForceField.h>

#include <sofa/component/statecontainer/MechanicalObject.h>

#include <sofa/component/mass/UniformMass.h>
#include <sofa/component/mass/MeshMatrixMass.h>

#include <sofa/component/topology/container/grid/RegularGridTopology.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetGeometryAlgorithms.h>

#include <sofa/component/visual/VisualStyle.h>

#include <sofa/component/solidmechanics/spring/RegularGridSpringForceField.h>
#include <sofa/component/solidmechanics/spring/SpringForceField.h>

#include <sofa/component/mapping/linear/SubsetMultiMapping.h>
#include <sofa/component/mapping/nonlinear/RigidMapping.h>

namespace sofa
{

typedef component::statecontainer::MechanicalObject<defaulttype::Rigid3Types> MechanicalObjectRigid3;
typedef component::statecontainer::MechanicalObject<defaulttype::Vec3Types> MechanicalObject3;
typedef component::solidmechanics::spring::RegularGridSpringForceField<defaulttype::Vec3Types> RegularGridSpringForceField3;
typedef component::solidmechanics::spring::SpringForceField<defaulttype::Vec3Types > SpringForceField3;
typedef component::linearsolver::iterative::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;
typedef component::mapping::nonlinear::RigidMapping<defaulttype::Rigid3Types, defaulttype::Vec3Types> RigidMappingRigid3_to_3;
typedef component::mapping::linear::SubsetMultiMapping<defaulttype::Vec3Types, defaulttype::Vec3Types> SubsetMultiMapping3_to_3;
typedef component::mass::UniformMass<defaulttype::Rigid3Types> UniformMassRigid3;
typedef component::mass::UniformMass<defaulttype::Vec3Types> UniformMass3;
typedef component::constraint::projective::FixedConstraint<defaulttype::Rigid3Types> FixedConstraintRigid3;
typedef component::constraint::projective::FixedConstraint<defaulttype::Vec3Types> FixedConstraint3;

/// Create a scene with a regular grid and an affine constraint for patch test

template<class DataTypes> PatchTestStruct<DataTypes>
Elasticity_test<DataTypes>::createRegularGridScene(
        simulation::Node::SPtr root,
        Coord startPoint,
        Coord endPoint,
        int numX,
        int numY,
        int numZ,
        sofa::type::Vec<6,SReal> entireBoxRoi,
        sofa::type::Vec<6,SReal> inclusiveBox,
        sofa::type::Vec<6,SReal> includedBox)
{
    // Definitions
    PatchTestStruct<DataTypes> patchStruct;
    typedef typename DataTypes::Real Real;
    typedef typename component::statecontainer::MechanicalObject<DataTypes> MechanicalObject;
    typedef typename sofa::component::mass::UniformMass <DataTypes> UniformMass;
    typedef component::topology::container::grid::RegularGridTopology RegularGridTopology;
    typedef typename component::engine::select::BoxROI<DataTypes> BoxRoi;
    typedef typename sofa::component::engine::select::PairBoxROI<DataTypes> PairBoxRoi;
    typedef typename sofa::component::constraint::projective::AffineMovementProjectiveConstraint<DataTypes> AffineMovementProjectiveConstraint;
    typedef component::linearsolver::iterative::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;

    // Root node
    root->setGravity( Coord(0,0,0) );
    root->setAnimate(false);
    root->setDt(0.05);

    // Node square
    simulation::Node::SPtr SquareNode = root->createChild("Square");

    // Euler implicit solver and cglinear solver
    component::odesolver::backward::EulerImplicitSolver::SPtr solver = modeling::addNew<component::odesolver::backward::EulerImplicitSolver>(SquareNode,"EulerImplicitSolver");
    solver->f_rayleighStiffness.setValue(0.5);
    solver->f_rayleighMass.setValue(0.5);
    CGLinearSolver::SPtr cgLinearSolver = modeling::addNew< CGLinearSolver >(SquareNode,"linearSolver");
    cgLinearSolver->d_maxIter.setValue(25);
    cgLinearSolver->d_tolerance.setValue(1e-5);
    cgLinearSolver->d_smallDenominatorThreshold.setValue(1e-5);

    // Mass
    typename UniformMass::SPtr mass = modeling::addNew<UniformMass>(SquareNode,"mass");

    // Regular grid topology
    RegularGridTopology::SPtr gridMesh = modeling::addNew<RegularGridTopology>(SquareNode,"loader");
    gridMesh->setSize(numX,numY,numZ);
    gridMesh->setPos(startPoint[0],endPoint[0],startPoint[1],endPoint[1],startPoint[2],endPoint[2]);

    //Mechanical object
    patchStruct.dofs = modeling::addNew<MechanicalObject>(SquareNode,"mechanicalObject");
    patchStruct.dofs->setName("mechanicalObject");
    patchStruct.dofs->setSrc("@"+gridMesh->getName(), gridMesh.get());

    //BoxRoi to find all mesh points
    type::vector< type::Vec<6,Real> > vecBox;
    vecBox.push_back(entireBoxRoi);
    typename BoxRoi::SPtr boxRoi = modeling::addNew<BoxRoi>(SquareNode,"boxRoi");
    boxRoi->d_alignedBoxes.setValue(vecBox);
    boxRoi->d_strict.setValue(false);

    //PairBoxRoi to define the constrained points = points of the border
    typename PairBoxRoi::SPtr pairBoxRoi = modeling::addNew<PairBoxRoi>(SquareNode,"pairBoxRoi");
    pairBoxRoi->inclusiveBox.setValue(inclusiveBox);
    pairBoxRoi->includedBox.setValue(includedBox);

    //Affine constraint
    patchStruct.affineConstraint  = modeling::addNew<AffineMovementProjectiveConstraint>(SquareNode,"affineConstraint");
    modeling::setDataLink(&boxRoi->d_indices,&patchStruct.affineConstraint->m_meshIndices);
    modeling::setDataLink(&pairBoxRoi->f_indices,& patchStruct.affineConstraint->m_indices);

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
    typedef typename component::statecontainer::MechanicalObject<DataTypes> MechanicalObject;
    typedef typename component::engine::select::BoxROI<DataTypes> BoxRoi;
    typedef component::linearsolver::iterative::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;
    typename simulation::Node::SPtr root;
    CylinderTractionStruct<DataTypes> tractionStruct;

    // Root node
    root = sofa::simulation::getSimulation()->createNewGraph("root");
    tractionStruct.root=root;

    root->setGravity( Coord(0,0,0) );
    root->setAnimate(false);
    root->setDt(0.05);


    // GenerateCylinder object
    typename sofa::component::engine::generate::GenerateCylinder<DataTypes>::SPtr eng= sofa::modeling::addNew<sofa::component::engine::generate::GenerateCylinder<DataTypes> >(root,"cylinder");
    eng->f_radius=0.2;
    eng->f_height=1.0;
    eng->f_resolutionCircumferential=resolutionCircumferential;
    eng->f_resolutionRadial=resolutionRadial;
    eng->f_resolutionHeight=resolutionHeight;
    // TetrahedronSetTopologyContainer object
    typename sofa::component::topology::container::dynamic::TetrahedronSetTopologyContainer::SPtr container1= sofa::modeling::addNew<sofa::component::topology::container::dynamic::TetrahedronSetTopologyContainer>(root,"Container1");
    sofa::modeling::setDataLink(&eng->f_tetrahedra,&container1->d_tetrahedron);
    sofa::modeling::setDataLink(&eng->f_outputTetrahedraPositions,&container1->d_initPoints);
    container1->d_createTriangleArray=true;
    // TetrahedronSetGeometryAlgorithms object
    typename sofa::component::topology::container::dynamic::TetrahedronSetGeometryAlgorithms<DataTypes>::SPtr geo1= sofa::modeling::addNew<sofa::component::topology::container::dynamic::TetrahedronSetGeometryAlgorithms<DataTypes> >(root);

    // CGLinearSolver
    typename CGLinearSolver::SPtr cgLinearSolver = modeling::addNew< CGLinearSolver >(root,"linearSolver");
    cgLinearSolver->d_maxIter.setValue(maxIter);
    cgLinearSolver->d_tolerance.setValue(1e-9);
    cgLinearSolver->d_smallDenominatorThreshold.setValue(1e-9);
    // StaticSolver
    typename component::odesolver::backward::StaticSolver::SPtr solver = modeling::addNew<component::odesolver::backward::StaticSolver>(root,"StaticSolver");
    // mechanicalObject object
    typename MechanicalObject::SPtr meca1= sofa::modeling::addNew<MechanicalObject>(root);
    sofa::modeling::setDataLink(&eng->f_outputTetrahedraPositions,&meca1->x);
    tractionStruct.dofs=meca1;
    // MeshMatrixMass
    typename sofa::component::mass::MeshMatrixMass<DataTypes>::SPtr mass= sofa::modeling::addNew<sofa::component::mass::MeshMatrixMass<DataTypes> >(root,"BezierMass");
    sofa::type::vector< Real > massDensity;
    massDensity.clear();
    massDensity.resize(1);
    massDensity[0] = 1.0;
    mass->d_massDensity.setValue(massDensity);
    mass->d_lumping=false;
    /// box fixed
    type::vector< type::Vec<6,Real> > vecBox;
    type::Vec<6,Real> box;
    box[0]= -0.01;box[1]= -0.01;box[2]= -0.01;box[3]= 0.01;box[4]= 0.01;box[5]= 0.01;
    vecBox.push_back(box);
    typename BoxRoi::SPtr boxRoi1 = modeling::addNew<BoxRoi>(root,"boxRoiFix");
    boxRoi1->d_alignedBoxes.setValue(vecBox);
    boxRoi1->d_strict.setValue(false);
    // FixedConstraint
    typename component::constraint::projective::FixedConstraint<DataTypes>::SPtr fc=
        modeling::addNew<typename component::constraint::projective::FixedConstraint<DataTypes> >(root);
    sofa::modeling::setDataLink(&boxRoi1->d_indices,&fc->d_indices);
    // FixedPlaneProjectiveConstraint
    typename component::constraint::projective::FixedPlaneProjectiveConstraint<DataTypes>::SPtr fpc=
            modeling::addNew<typename component::constraint::projective::FixedPlaneProjectiveConstraint<DataTypes> >(root);
    fpc->d_dmin= -0.01;
    fpc->d_dmax= 0.01;
    fpc->d_direction=Coord(0,0,1);
    /// box pressure
    box[0]= -0.2;box[1]= -0.2;box[2]= 0.99;box[3]= 0.2;box[4]= 0.2;box[5]= 1.01;
    vecBox[0]=box;
    typename BoxRoi::SPtr boxRoi2 = modeling::addNew<BoxRoi>(root,"boxRoiPressure");
    boxRoi2->d_alignedBoxes.setValue(vecBox);
    boxRoi2->d_computeTriangles=true;
    boxRoi2->d_strict.setValue(false);
    /// TrianglePressureForceField
    typename sofa::component::mechanicalload::TrianglePressureForceField<DataTypes>::SPtr tpff=
            modeling::addNew<typename sofa::component::mechanicalload::TrianglePressureForceField<DataTypes> >(root);
    tractionStruct.forceField=tpff;
    sofa::modeling::setDataLink(&boxRoi2->d_triangleIndices,&tpff->triangleList);
    // LineProjectiveConstraint
    typename component::constraint::projective::LineProjectiveConstraint<DataTypes>::SPtr ptlc=
            modeling::addNew<typename component::constraint::projective::LineProjectiveConstraint<DataTypes> >(root);
    ptlc->f_direction=Coord(1,0,0);
    ptlc->f_origin=Coord(0,0,0);
    sofa::type::vector<sofa::Index> vArray;
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
    using type::vector;
    using core::objectmodel::New;

    // The graph root node
    simulation::Node::SPtr  root = simulation::getSimulation()->createNewGraph("root");
    root->setGravity( Coord(0,-10,0) );
    root->setAnimate(false);
    root->setDt(0.01);
    component::visual::addVisualStyle(root)->setShowVisual(false).setShowCollision(false).setShowMapping(true).setShowBehavior(true);

    simulation::Node::SPtr simulatedScene = root->createChild("simulatedScene");

    component::odesolver::backward::EulerImplicitSolver::SPtr eulerImplicitSolver = New<component::odesolver::backward::EulerImplicitSolver>();
    simulatedScene->addObject( eulerImplicitSolver );
    CGLinearSolver::SPtr cgLinearSolver = New<CGLinearSolver>();
    simulatedScene->addObject(cgLinearSolver);

    // The rigid object
    simulation::Node::SPtr rigidNode = simulatedScene->createChild("rigidNode");
    MechanicalObjectRigid3::SPtr rigid_dof = modeling::addNew<MechanicalObjectRigid3>(rigidNode, "dof");
    UniformMassRigid3::SPtr rigid_mass = modeling::addNew<UniformMassRigid3>(rigidNode,"mass");
    FixedConstraintRigid3::SPtr rigid_fixedConstraint = modeling::addNew<FixedConstraintRigid3>(rigidNode,"fixedConstraint");

    // Particles mapped to the rigid object
    simulation::Node::SPtr mappedParticles = rigidNode->createChild("mappedParticles");
    MechanicalObject3::SPtr mappedParticles_dof = modeling::addNew< MechanicalObject3>(mappedParticles,"dof");
    RigidMappingRigid3_to_3::SPtr mappedParticles_mapping = modeling::addNew<RigidMappingRigid3_to_3>(mappedParticles,"mapping");
    mappedParticles_mapping->setModels( rigid_dof.get(), mappedParticles_dof.get() );

    // The independent particles
    simulation::Node::SPtr independentParticles = simulatedScene->createChild("independentParticles");
    MechanicalObject3::SPtr independentParticles_dof = modeling::addNew< MechanicalObject3>(independentParticles,"dof");

    // The deformable grid, connected to its 2 parents using a MultiMapping
    simulation::Node::SPtr deformableGrid = independentParticles->createChild("deformableGrid"); // first parent
    mappedParticles->addChild(deformableGrid);                                       // second parent

    component::topology::container::grid::RegularGridTopology::SPtr deformableGrid_grid = modeling::addNew<component::topology::container::grid::RegularGridTopology>( deformableGrid, "grid" );
    deformableGrid_grid->setSize(numX,numY,numZ);
    deformableGrid_grid->setPos(startPoint[0],endPoint[0],startPoint[1],endPoint[1],startPoint[2],endPoint[2]);

    MechanicalObject3::SPtr deformableGrid_dof = modeling::addNew< MechanicalObject3>(deformableGrid,"dof");

    SubsetMultiMapping3_to_3::SPtr deformableGrid_mapping = modeling::addNew<SubsetMultiMapping3_to_3>(deformableGrid,"mapping");
    deformableGrid_mapping->addInputModel(independentParticles_dof.get()); // first parent
    deformableGrid_mapping->addInputModel(mappedParticles_dof.get());      // second parent
    deformableGrid_mapping->addOutputModel(deformableGrid_dof.get());

    UniformMass3::SPtr mass = modeling::addNew<UniformMass3>(deformableGrid,"mass" );
    mass->d_vertexMass.setValue( totalMass/(numX*numY*numZ) );

    RegularGridSpringForceField3::SPtr spring = modeling::addNew<RegularGridSpringForceField3>(deformableGrid, "spring");
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
    vector<sofa::type::BoundingBox> boxes(numRigid);
    vector< vector<size_t> > indices(numRigid); // indices of the particles in each box
    double eps = (endPoint[0]-startPoint[0])/(numX*2);

    // first box, x=xmin
    boxes[0] = sofa::type::BoundingBox(sofa::type::Vec3d(startPoint[0]-eps, startPoint[1]-eps, startPoint[2]-eps),
            sofa::type::Vec3d(startPoint[0]+eps,   endPoint[1]+eps,   endPoint[2]+eps));

    // second box, x=xmax
    boxes[1] = sofa::type::BoundingBox(sofa::type::Vec3d(endPoint[0]-eps, startPoint[1]-eps, startPoint[2]-eps),
            sofa::type::Vec3d(endPoint[0]+eps,   endPoint[1]+eps,   endPoint[2]+eps));
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
    mappedParticles_mapping->d_globalToLocalCoords.setValue(true); // to define the mapped positions in world coordinates
    MechanicalObject3::WriteVecCoord xindependent = independentParticles_dof->writePositions();
    vector< std::pair<MechanicalObject3*,size_t> > parentParticles(xgrid.size());

    // independent particles
    size_t independentIndex=0;
    for( size_t i=0; i<xgrid.size(); i++ ){
        if( isFree[i] ){
            parentParticles[i]=std::make_pair(independentParticles_dof.get(),independentIndex);
            xindependent[independentIndex] = xgrid[i];
            independentIndex++;
        }
    }

    // mapped particles
    size_t mappedIndex=0;
    vector<unsigned>* rigidIndexPerPoint = mappedParticles_mapping->d_rigidIndexPerPoint.beginEdit();
    for( size_t b=0; b<numRigid; b++ )
    {
        const vector<size_t>& ind = indices[b];
        for(size_t i=0; i<ind.size(); i++)
        {
            rigidIndexPerPoint->push_back( b ); // tell the mapping the number of points associated with this frame
            parentParticles[ind[i]]=std::make_pair(mappedParticles_dof.get(),mappedIndex);
            xmapped[mappedIndex] = xgrid[ ind[i] ];
            mappedIndex++;

        }
    }
    mappedParticles_mapping->d_rigidIndexPerPoint.endEdit();

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
    MechanicalObject3::SPtr FixedPoint = modeling::addNew<MechanicalObject3>(fixedPointNode,"fixedPoint");

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
MechanicalObject3::SPtr massDof = modeling::addNew<MechanicalObject3>(massNode,"massNode");

// Set position and velocity
FixedPoint->resize(1);
MechanicalObject3::WriteVecCoord xMassDof = massDof->writePositions();
copyToData( xMassDof, xMass );
MechanicalObject3::WriteVecDeriv vMassDof = massDof->writeVelocities();
copyToData( vMassDof, vMass );

UniformMass3::SPtr massPtr = modeling::addNew<UniformMass3>(massNode,"mass");
massPtr->d_totalMass.setValue( mass );

// attach a spring
SpringForceField3::SPtr spring = core::objectmodel::New<SpringForceField3>(FixedPoint.get(), massDof.get());
root->addObject(spring);
spring->addSpring(0,0,stiffness ,0, restLength);

return root;

}

}// sofa

#endif

