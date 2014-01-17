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


#include "Sofa_test.h"
#include<sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/component/init.h>
#include <sofa/core/ExecParams.h>

//Including Simulation
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>

// Including constraint, force and mass
#include <sofa/component/topology/RegularGridTopology.h>
#include <sofa/component/projectiveconstraintset/BilinearMovementConstraint.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/component/interactionforcefield/MeshSpringForceField.h>
#include <sofa/component/forcefield/TetrahedronFEMForceField.h>
#include <sofa/core/MechanicalParams.h>

#include <sofa/defaulttype/VecTypes.h>
#include <plugins/SceneCreator/SceneCreator.h>

namespace sofa {

using std::cout;
using std::cerr;
using std::endl;
using namespace component;
using namespace defaulttype;
using namespace modeling;

/**  Test suite for ProjectToLineConstraint.
The test cases are defined in the #Test_Cases member group.
  */
template <typename _DataTypes>
struct Patch_test : public Sofa_test<typename _DataTypes::Real>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Deriv Deriv;
    typedef projectiveconstraintset::BilinearMovementConstraint<DataTypes> BilinearMovementConstraint;
    typedef container::MechanicalObject<DataTypes> MechanicalObject;
    typedef typename component::interactionforcefield::MeshSpringForceField<DataTypes> MeshSpringForceField;
    typedef typename component::forcefield::TetrahedronFEMForceField<DataTypes> TetraForceField;
    
    /// Root of the scene graph
    simulation::Node::SPtr root;      
    /// Simulation
    simulation::Simulation* simulation;  
    // Structure which contains current node and pointers to the mechanical object and the bilinear constraint
    PatchTestStruct<DataTypes> patchStruct;
    
    // Create the context for the scene
    void SetUp()
    { 
        // Init simulation
        sofa::component::init();
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

         root = simulation::getSimulation()->createNewGraph("root");
    }

    // Create a scene with a 2D regular grid and a bilinear constraint
    void createScene2DRegularGrid()
    { 
        // Initialization
        Vec<6,SReal> box (-0.1,-0.1,0,1.1,1.1,0);
        helper::vector< Vec<6,SReal> > vecBox;
        vecBox.push_back(box);

        // Create a scene with a regular grid
        patchStruct = createRegularGridScene<DataTypes>(
                        root,  // attached to the root node
                        Vec<3,SReal>(0,0,0), // Start point of regular grid
                        Vec<3,SReal>(1,1,0), // End point of regular grid
                        5,5,1,  // Resolution of the regular grid
                        vecBox, // BoxRoi to find all mesh points
                        Vec<6,SReal>(-0.1,-0.1,0,1.1,1.1,0), // inclusive box of pair box roi
                        Vec<6,SReal>(0.1,0.1,0,0.9,0.9,0)); // included box of pair box roi

        simulation::Node::SPtr SquareNode = patchStruct.SquareNode;
     
        //Force field for 2D Grid
        typename MeshSpringForceField::SPtr meshSpringForceField = addNew<MeshSpringForceField> (SquareNode,"forceField");
        meshSpringForceField->setStiffness(10);

        // Set the corner movements of the bilinear constraint
        VecDeriv cornerMovementsData = patchStruct.bilinearConstraint->m_cornerMovements.getValue();
        cornerMovementsData.push_back(Deriv(0,0,0));
        cornerMovementsData.push_back(Deriv(0.01,-0.01,0));
        cornerMovementsData.push_back(Deriv(0,0,0));
        cornerMovementsData.push_back(Deriv(-0.01,0.01,0));
        patchStruct.bilinearConstraint->m_cornerMovements.setValue(cornerMovementsData);

    }

     // Create a scene with a 3D regular grid and a bilinear constraint
    void createScene3DRegularGrid()
    {
        // Initialization
        Vec<6,SReal> box (-0.1,-0.1,-0.1,1.1,1.1,1.1);
        helper::vector< Vec<6,SReal> > vecBox;
        vecBox.push_back(box);

        // Create a scene with a regular grid
        patchStruct = createRegularGridScene<DataTypes>(
                        root,  // attached to the root node
                        Vec<3,SReal>(0,0,0), // Start point of regular grid
                        Vec<3,SReal>(1,1,1), // End point of regular grid
                        5,5,5,  // Resolution of the regular grid
                        vecBox, // BoxRoi to find all mesh points
                        Vec<6,SReal>(-0.1,-0.1,-0.1,1.1,1.1,1.1), // inclusive box of pair box roi
                        Vec<6,SReal>(0.1,0.1,0.1,0.9,0.9,0.9)); // included box of pair box roi
       
        simulation::Node::SPtr SquareNode = patchStruct.SquareNode;
  
        // Force field for 3D Grid
        typename TetraForceField::SPtr tetraFEM = addNew<TetraForceField>(SquareNode,"forceField");
        tetraFEM->setMethod("polar");
        tetraFEM->setYoungModulus(20);
        tetraFEM->setPoissonRatio(0.4);

        // Set the corner movements of the bilinear constraint
        VecDeriv cornerMovementsData = patchStruct.bilinearConstraint->m_cornerMovements.getValue();
        cornerMovementsData.push_back(Deriv(0,0,0));
        cornerMovementsData.push_back(Deriv(0.01,-0.01,-0.01));
        cornerMovementsData.push_back(Deriv(0.01,0.01,-0.01));
        cornerMovementsData.push_back(Deriv(-0.01,0.01,-0.01));
        cornerMovementsData.push_back(Deriv(-0.01,-0.01,0.01));
        cornerMovementsData.push_back(Deriv(0.01,-0.01,0.01));
        cornerMovementsData.push_back(Deriv(0,0,0));
        cornerMovementsData.push_back(Deriv(-0.01,0.01,0.01));
        patchStruct.bilinearConstraint->m_cornerMovements.setValue(cornerMovementsData);
 
    }

    bool test_projectPosition()
    {
        // Init simulation
        sofa::simulation::getSimulation()->init(root.get());

        // Animate
        do
        {sofa::simulation::getSimulation()->animate(root.get(),0.5);}
        while(root->getAnimationLoop()->getTime() < 22); 
   
        // Get the simulated final positions
        typename MechanicalObject::WriteVecCoord x = patchStruct.dofs->writePositions();

        // Compute the theoretical final positions    
        VecCoord finalPos;
        patchStruct.bilinearConstraint->getFinalPositions( finalPos,*patchStruct.dofs->write(core::VecCoordId::position()) );

        // Compare the theoretical positions and the simulated positions   
        bool succeed=true;
        for(size_t i=0; i<finalPos.size(); i++ )
        {
            if((finalPos[i]-x[i]).norm()>7e-4)
            {   
                succeed = false;
                ADD_FAILURE() << "final Position of point " << i << " is wrong: " << x[i] << std::endl <<"the expected Position is " << finalPos[i] << "difference = " <<(finalPos[i]-x[i]).norm() << std::endl;
            }
        }
        return succeed;
    }


    void TearDown()
    {
        if (root!=NULL)
            sofa::simulation::getSimulation()->unload(root);
//        cerr<<"tearing down"<<endl;
    }

};

// Define the list of DataTypes to instanciate
using testing::Types;
typedef Types<
    Vec3Types,
    Vec3Types
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(Patch_test, DataTypes);

// first test case
TYPED_TEST( Patch_test , patchTest2D )
{
   this->createScene2DRegularGrid();
   ASSERT_TRUE( this->test_projectPosition());
}

// second test case
TYPED_TEST( Patch_test , patchTest3D )
{
    this->createScene3DRegularGrid();
    ASSERT_TRUE( this->test_projectPosition());
}

} // namespace sofa
