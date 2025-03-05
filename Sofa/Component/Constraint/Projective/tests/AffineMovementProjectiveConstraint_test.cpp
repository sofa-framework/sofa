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
#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;
#include <sofa/testing/NumericTest.h>
using sofa::testing::NumericTest;

#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/solidmechanics/spring/MeshSpringForceField.h>
#include <sofa/component/constraint/projective/AffineMovementProjectiveConstraint.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/RandomGenerator.h>
#include <SceneCreator/SceneCreator.h>

#include <sofa/component/topology/testing/RegularGridNodeCreation.h>


using namespace sofa::type;
using namespace sofa::defaulttype;

namespace sofa {
namespace {

template <typename _DataTypes>
struct AffineMovementProjectiveConstraint_test : public BaseSimulationTest, NumericTest<typename _DataTypes::Coord::value_type>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::CPos CPos;
    typedef typename Coord::value_type Real;
    typedef component::statecontainer::MechanicalObject<DataTypes> MechanicalObject;
    typedef typename component::solidmechanics::spring::MeshSpringForceField<DataTypes> MeshSpringForceField;

    /// Root of the scene graph
    simulation::Node::SPtr root;
    /// Tested simulation
    simulation::Simulation* simulation;
    /// Structure which contains current node and pointers to the mechanical object and the affine constraint
    PatchTestStruct<DataTypes> patchStruct;
    /// Tested Rotation: random rotation matrix
    type::Mat<3,3,Real> testedRotation;
    /// Tested Translation: random translation
    Coord testedTranslation;
    /// Seed for random value
    long seed;
    // Random generator
    sofa::helper::RandomGenerator randomGenerator;

     // Create the context for the scene
     void doSetUp() override
     {
         // Init simulation
         simulation = sofa::simulation::getSimulation();

         root = simulation::getSimulation()->createNewGraph("root");

         // Init seed with a random value between 0 and 100
         randomGenerator.initSeed( (long)time(0) );
         seed = randomGenerator.random<long>(0,100);

         // Create a scene with a regular grid
         patchStruct = sofa::createRegularGridScene<DataTypes>(
                        root,  // attached to the root node
                        Vec<3,SReal>(0,0,0), // Start point of regular grid
                        Vec<3,SReal>(1,1,0), // End point of regular grid
                        5,5,1,  // Resolution of the regular grid
                        Vec<6,SReal>(-0.1,-0.1,0,1.1,1.1,0), // BoxRoi to find all mesh points
                        Vec<6,SReal>(-0.1,-0.1,0,1.1,1.1,0), // inclusive box of pair box roi
                        Vec<6,SReal>(0.1,0.1,0,0.9,0.9,0)); // included box of pair box roi

        simulation::Node::SPtr SquareNode = patchStruct.SquareNode;

        //Force field for 2D Grid
        typename MeshSpringForceField::SPtr meshSpringForceField = modeling::addNew<MeshSpringForceField> (SquareNode,"forceField");
        meshSpringForceField->setStiffness(10);

        // Init seed
        randomGenerator.initSeed(seed);

        // Random Rotation
        SReal x,y,z,w;
        x = 0;
        y = 0;
        z = 1;
        w = randomGenerator.random<SReal>(0.0,360.0);
        Quat quat(x,y,z,w);
        quat.normalize();
        quat.toMatrix(testedRotation);

        patchStruct.affineConstraint->d_rotation.setValue(testedRotation);

        // Random Translation
        for(size_t i=0;i<Coord::total_size;++i)
        {
            testedTranslation[i]=randomGenerator.random<SReal>(-2.0,2.0);
            if(i==2)
                testedTranslation[i]=0;
        }

        patchStruct.affineConstraint->d_translation.setValue(testedTranslation);
        patchStruct.affineConstraint->d_endConstraintTime.setValue(0.1);
    }

     // After simulation compare the positions of points to the theoretical positions.
     bool projectPosition(double convergenceAccuracy, double diffMaxBetweenSimulatedAndTheoreticalPosition)
     {
        // Init simulation
        sofa::simulation::node::initRoot(root.get());

        // Compute the theoretical final positions
        VecCoord finalPos;
        patchStruct.affineConstraint->getFinalPositions( finalPos,*patchStruct.dofs->write(core::vec_id::write_access::position) );


        // Initialize
        size_t numNodes = finalPos.size();
        VecCoord xprev(numNodes);
        VecDeriv dx(numNodes);
        bool hasConverged = true;

        for (size_t i=0; i<numNodes; i++)
        {
            xprev[i] = CPos(0,0,0);
        }

        // Animate
        do
        {
            hasConverged = true;
            sofa::simulation::node::animate(root.get(), 0.5);
            typename MechanicalObject::ReadVecCoord x = patchStruct.dofs->readPositions();

            // Compute dx
            for (size_t i=0; i<x.size(); i++)
            {
                dx[i] = x[i]-xprev[i];
                // Test convergence
                if(dx[i].norm()>convergenceAccuracy) hasConverged = false;
            }

            // xprev = x
            for (size_t i=0; i<numNodes; i++)
            {
                xprev[i]=x[i];
            }
        }
        while(!hasConverged); // not converged

        // Get simulated positions
        typename MechanicalObject::WriteVecCoord x = patchStruct.dofs->writePositions();

        // Compare the theoretical positions and the simulated positions
        bool succeed=true;
        for(size_t i=0; i<finalPos.size(); i++ )
        {
            if((finalPos[i]-x[i]).norm()>diffMaxBetweenSimulatedAndTheoreticalPosition)
            {
                succeed = false;
                ADD_FAILURE() << "final Position of point " << i << " is wrong: " << x[i] << std::endl <<"the expected Position is " << finalPos[i] << std::endl
                    << "difference = " <<(finalPos[i]-x[i]).norm() << std::endl <<"rotation = " << testedRotation << std::endl << " translation = " << testedTranslation << std::endl
                    << "seed =" << seed;

            }
        }
        return succeed;
    }

};

// Define the list of DataTypes to instantiate
using ::testing::Types;
typedef Types<
    defaulttype::Vec3Types
> DataTypes; // the types to instantiate.

// Test suite for all the instantiations
TYPED_TEST_SUITE(AffineMovementProjectiveConstraint_test, DataTypes);
// first test case
TYPED_TEST( AffineMovementProjectiveConstraint_test , testValue )
{
   EXPECT_MSG_NOEMIT(Error) ;
   ASSERT_TRUE( this->projectPosition(5e-6,5e-5));
}




}// namespace
}// namespace sofa







