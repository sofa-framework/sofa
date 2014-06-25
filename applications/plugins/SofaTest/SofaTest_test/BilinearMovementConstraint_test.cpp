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
#include <SofaComponentMain/init.h>
#include <sofa/core/ExecParams.h>

//Including Simulation
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>

//Including Solvers
#include <SofaExplicitOdeSolver/EulerSolver.h>
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>

// Including topology
#include <sofa/component/topology//RegularGridTopology.h>
#include <SofaBaseTopology/PointSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaBaseTopology/TriangleSetTopologyModifier.h>
#include <SofaBaseTopology/TriangleSetTopologyAlgorithms.h>
#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>

// Including constraint, force and mass
#include <SofaLoader/GridMeshCreator.h>
#include <SofaBoundaryCondition/BilinearMovementConstraint.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaSimpleFem/TriangularFEMForceField.h>
#include <SofaBaseMechanics/UniformMass.h>
#include <sofa/core/MechanicalParams.h>

// Including engine
#include <SofaEngine/BoxROI.h>
#include <SofaEngine/PairBoxRoi.h>

#include <sofa/defaulttype/VecTypes.h>


namespace sofa {

using std::cout;
using std::cerr;
using std::endl;
using namespace component;
using namespace defaulttype;

typedef component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;


/**  Test suite for ProjectToLineConstraint.
The test cases are defined in the #Test_Cases member group.
  */
template <typename _DataTypes>
struct BilinearMovementConstraint_test : public Sofa_test<typename _DataTypes::Real>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef projectiveconstraintset::BilinearMovementConstraint<DataTypes> BilinearMovementConstraint;
    typedef container::MechanicalObject<DataTypes> MechanicalObject; 
    
    /// Root of the scene graph
    simulation::Node::SPtr root;      
    /// Simulation
    simulation::Simulation* simulation;         

    typename MechanicalObject::SPtr dofs;
    typename BilinearMovementConstraint::SPtr bilinearConstraint;
    

    void SetUp()
    {      
        // Init
        sofa::component::init();
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

        // Create the scene from the xml file
        std::string fileName = "examples/Components/constraint/BilinearConstraint.scn";
        fileName = sofa::helper::system::DataRepository.getFile(fileName);
        root = sofa::core::objectmodel::SPtr_dynamic_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(fileName.c_str()));
        
        // Init simulation
        sofa::simulation::getSimulation()->init(root.get());
 
        // Animate
        do
        {sofa::simulation::getSimulation()->animate(root.get(),1.0);}
        while(root->getAnimationLoop()->getTime() < 350); 

    }



    bool test_projectPosition()
    {
        // Search in the scene the mechanical object
        dofs = root->get<MechanicalObject>(root->SearchDown);
        
        // Search in the scene the bilinear constraint
        bilinearConstraint = root->get<BilinearMovementConstraint>(root->SearchDown);
       
        // Get the simulated final positions
        typename MechanicalObject::WriteVecCoord x = dofs->writePositions();

        // Compute the theoretical final positions    
        VecCoord finalPos;
        bilinearConstraint->getFinalPositions( finalPos,*dofs->write(core::VecCoordId::position()) );
            
         // Compare the theoretical positions and the simulated positions   
         bool succeed=true;
         for(size_t i=0; i<finalPos.size(); i++ )
         {
             if((finalPos[i]-x[i]).norm()>1e-3)
             {   
                 succeed = false;
                 ADD_FAILURE() << "final Position of point " << i << " is wrong: " << x[i] << std::endl <<"the expected Position is " << finalPos[i] << std::endl;
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
TYPED_TEST_CASE(BilinearMovementConstraint_test, DataTypes);
// first test case
TYPED_TEST( BilinearMovementConstraint_test , patchTest )
{
   ASSERT_TRUE( this->test_projectPosition());
}


} // namespace sofa
