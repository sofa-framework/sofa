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
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/topology/container/dynamic/PointSetTopologyContainer.h>
#include <sofa/component/constraint/projective/SkeletalMotionProjectiveConstraint.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/defaulttype/VecTypes.h>


namespace sofa {

using namespace component;
using namespace defaulttype;



/**  Test suite for ProjectToLineConstraint.
The test cases are defined in the #Test_Cases member group.
  */
template <typename _DataTypes>
struct SkeletalMotionProjectiveConstraint_test : public BaseSimulationTest, NumericTest<typename _DataTypes::Coord::value_type>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::CPos CPos;
    typedef typename DataTypes::CRot CRot;
    typedef typename Coord::value_type Real;

    typedef constraint::projective::SkeletalMotionProjectiveConstraint<DataTypes> SkeletalMotionProjectiveConstraint;
    typedef constraint::projective::SkeletonJoint<DataTypes> SkeletonJoint;
    typedef statecontainer::MechanicalObject<DataTypes> MechanicalObject;

    simulation::Node::SPtr root;                 ///< Root of the scene graph, created by the constructor an re-used in the tests
    simulation::Simulation* simulation;          ///< created by the constructor an re-used in the tests

    type::SVector<SkeletonJoint> joints;        ///< skeletal joint
    typename SkeletalMotionProjectiveConstraint::SPtr projection;
    typename MechanicalObject::SPtr dofs;

    /// Create the context for the tests.
    void SetUp() override
    {
        simulation = sofa::simulation::getSimulation();

        /// Create the scene
        root = simulation->createNewGraph("root");

        dofs = core::objectmodel::New<MechanicalObject>();
        root->addObject(dofs);

        projection = core::objectmodel::New<SkeletalMotionProjectiveConstraint>();
        root->addObject(projection);




    }

    /** @name Test_Cases
      For each of these cases, we can test if the projections work
      */
    ///@{
    /** Constrain one particle, and not the last one.
    Detects bugs like not setting the projection matrix entries beyond the last constrained particle
    */
    void init_2bones()
    {
        joints.clear();

        typename MechanicalObject::WriteVecCoord x = dofs->writePositions();
        x.resize(2);
        VecCoord rigids(2);
        DataTypes::setCPos(x[1], CPos(0,1,0));
        DataTypes::setCRot(x[1], CRot(0.707107,0, 0, 0.707107)); // rotation x: 90 degree
        Coord target(CPos(1,1,1), CRot(0,0.382683,0,0.92388)); //rotation y : 45 degrees


        joints.resize(2);

        joints[0].addChannel(x[0], 0);
        joints[0].addChannel(target, 1);
        joints[0].setRestPosition(x[0]);

        joints[1].addChannel(x[1], 0);
        joints[1].setRestPosition(x[1]);
        joints[1].mParentIndex = 0;

        type::vector<int> bones(2,0); bones[1] = 1;
        projection->setSkeletalMotion(joints, bones);

        /// Init
        sofa::simulation::node::initRoot(root.get());
        sofa::simulation::node::animate(root.get(), 0.25_sreal);
        sofa::simulation::node::animate(root.get(), 0.25_sreal);

    }



    bool test_projectPosition()
    {
        projection->projectPosition(core::mechanicalparams::defaultInstance(), *dofs->write(core::VecCoordId::position()));
        typename MechanicalObject::ReadVecCoord x = dofs->readPositions();
        Coord target0(CPos(0.5,0.5,0.5), CRot(0, 0.19509, 0, 0.980785));
        Coord target1(CPos(0.5,1.5,0.5), CRot(0.69352, 0.13795, -0.13795, 0.69352));

        bool succeed = true;
         if( !this->isSmall((x[0].getCenter() - target0.getCenter()).norm(),100) ||
            !this->isSmall((x[1].getCenter() - target1.getCenter()).norm(),100) )
        {
             succeed = false;
             ADD_FAILURE() << "Position of constrained bones is wrong: "<<x[0].getCenter()<<", "<<x[1].getCenter();
        }

        if( !(x[0].getOrientation() == target0.getOrientation()) ||
            !(x[1].getOrientation() == target1.getOrientation()) )
        {
            succeed = false;
            ADD_FAILURE() << "Rotation of constrained bones is wrong: "<<x[0].getOrientation()<<", "<<x[1].getOrientation();
        }

        return succeed;
    }

    bool test_projectVelocity()
    {
        projection->projectVelocity(core::mechanicalparams::defaultInstance(), *dofs->write(core::VecDerivId::velocity()));
        typename MechanicalObject::ReadVecDeriv x = dofs->readVelocities();
        bool succeed = true;
        Deriv target(CPos(1,1,1), typename Deriv::Rot(0,0.785397,0));

        if(!(x[0]==target) || !(x[1]==target))
        {
            succeed = false;
             ADD_FAILURE() << "velocities of constrained bones is wrong: "<<x[0]<<", "<<x[1]<<", expected: "<<target;
        }

        return succeed;
    }

    void TearDown() override
    {
        if (root!=nullptr)
            sofa::simulation::node::unload(root);
    }


 };


// Define the list of DataTypes to instanciate
using ::testing::Types;
typedef Types<
    Rigid3Types
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_SUITE(SkeletalMotionProjectiveConstraint_test, DataTypes);
// first test case
TYPED_TEST( SkeletalMotionProjectiveConstraint_test , twoConstrainedBones )
{
    EXPECT_MSG_NOEMIT(Error) ;
    this->init_2bones();
    ASSERT_TRUE(  this->test_projectPosition() );
    ASSERT_TRUE(  this->test_projectVelocity() );
}


} // namespace sofa

