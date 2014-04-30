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

#ifndef SOFA_SIMPLEOBJECTCREATOR_INL
#define SOFA_SIMPLEOBJECTCREATOR_INL

#include "SceneCreator.h"

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

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>

namespace sofa
{

namespace modeling {

using namespace simulation;

/// Create a scene with a regular grid and an affine constraint for patch test

template<class DataTypes> PatchTestStruct<DataTypes> createRegularGridScene(simulation::Node::SPtr root, Vec<3,SReal> startPoint, Vec<3,SReal> endPoint, unsigned numX, unsigned numY, unsigned numZ, Vec<6,SReal> entireBoxRoi, Vec<6,SReal> inclusiveBox, Vec<6,SReal> includedBox)
{
    // Definitions
    PatchTestStruct<DataTypes> patchStruct;
    typedef typename component::container::MechanicalObject<DataTypes> MechanicalObject;
    typedef typename sofa::component::mass::UniformMass <DataTypes, SReal> UniformMass;
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
    helper::vector < defaulttype::Vec<6,SReal> > vecBox;
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

template<class DataTypes> CylinderTractionStruct<DataTypes>  createCylinderTractionScene(size_t resolutionCircumferential,size_t  resolutionRadial,  size_t  resolutionHeight,size_t maxIter)
{
    // Definitions
	 typedef typename DataTypes::Coord Coord;
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
	typename sofa::component::mass::MeshMatrixMass<DataTypes,SReal>::SPtr mass= sofa::modeling::addNew<sofa::component::mass::MeshMatrixMass<DataTypes,SReal> >(root,"BezierMass");
	mass->m_massDensity=1.0;
	mass->lumping=false;
	/// box fixed
	helper::vector < defaulttype::Vec<6,SReal> > vecBox;
	defaulttype::Vec<6,SReal> box;
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
	sofa::helper::vector<unsigned int> vArray;
	vArray.push_back(resolutionCircumferential*(resolutionRadial-1)+1);
	ptlc->f_indices.setValue(vArray);
    
    return tractionStruct;
}


}// modeling

}// sofa

#endif
