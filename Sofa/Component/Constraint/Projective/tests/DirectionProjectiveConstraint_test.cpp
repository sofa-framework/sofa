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
#include <sofa/component/constraint/projective/DirectionProjectiveConstraint.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/defaulttype/VecTypes.h>


namespace sofa {

namespace {

using namespace component;
using namespace defaulttype;
using namespace core::objectmodel;

/**  Test suite for DirectionProjectiveConstraint.
The test cases are defined in the #Test_Cases member group.
  */
template <typename _DataTypes>
struct DirectionProjectiveConstraint_test : public BaseSimulationTest, NumericTest<typename _DataTypes::Coord::value_type>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::CPos CPos;
    typedef typename Coord::value_type Real;
    typedef constraint::projective::DirectionProjectiveConstraint<DataTypes> DirectionProjectiveConstraint;
    typedef typename DirectionProjectiveConstraint::Indices Indices;
    typedef component::topology::container::dynamic::PointSetTopologyContainer PointSetTopologyContainer;
    typedef statecontainer::MechanicalObject<DataTypes> MechanicalObject;

    simulation::Node::SPtr root;                 ///< Root of the scene graph, created by the constructor an re-used in the tests
    simulation::Simulation* simulation;          ///< created by the constructor an re-used in the tests

    unsigned numNodes;                         ///< number of particles used for the test
    Indices indices;                           ///< indices of the nodes to project
    CPos direction;                            ///< direction to project to
    typename DirectionProjectiveConstraint::SPtr projection;
    typename MechanicalObject::SPtr dofs;

     /// Create the context for the tests.
    void doSetUp() override
    {
        //Init
        simulation = sofa::simulation::getSimulation();
        ASSERT_NE(simulation, nullptr);

        /// Create the scene
        root = simulation->createNewGraph("root");

        const PointSetTopologyContainer::SPtr topology = New<PointSetTopologyContainer>();
        root->addObject(topology);

        dofs = New<MechanicalObject>();
        root->addObject(dofs);

        // Project direction constraint
        projection = New<DirectionProjectiveConstraint>();
        root->addObject(projection);

        /// Set the values
        numNodes = 3;
        dofs->resize(numNodes);

        direction = CPos(0,1,1);
        projection->d_direction.setValue(direction);

    }

    /** @name Test_Cases
      For each of these cases, we can test if the projections work
      */
    ///@{
    /** Constraint one particle, and not the last one.
    */
    void init_oneConstrainedParticle()
    {
        indices.clear();
        indices.push_back(0);
//        std::sort(indices.begin(),indices.end()); // checking vectors in linear time requires sorted indices
        projection->d_indices.setValue(indices);

        /// Init
        sofa::simulation::node::initRoot(root.get());
    }

    /** Constraint all the particles.
    */
    void init_allParticlesConstrained()
    {
        indices.clear();
        for(unsigned i = 0; i<numNodes; i++)
            indices.push_back(i);
         projection->d_indices.setValue(indices);

         /// Init
         sofa::simulation::node::initRoot(root.get());
    }
    ///@}

    // Test project position
    bool test_projectPosition()
    {
       VecCoord xprev(numNodes);
       typename MechanicalObject::WriteVecCoord x = dofs->writePositions();
       for (unsigned i=0; i<numNodes; i++){
           xprev[i] = x[i] = CPos(i,0,0);
       }

       projection->projectPosition(core::mechanicalparams::defaultInstance(), *dofs->write(core::vec_id::write_access::position) );

       type::vector<CPos> m_origin;

       // particle original position
       const VecCoord& xOrigin = projection->getMState()->read(core::vec_id::read_access::position)->getValue();
        for( typename Indices::const_iterator it = indices.begin() ; it != indices.end() ; ++it )
        {
             m_origin.push_back( DataTypes::getCPos(xOrigin[*it]) );
        }


       bool succeed=true;
       typename Indices::const_iterator it = indices.begin(); // must be sorted
       for(unsigned i=0; i<numNodes; i++ )
       {
           if ((it!=indices.end()) && ( i==*it ))  // constrained particle
           {
              CPos crossprod = (x[i]-m_origin[i]).cross(direction); // should be parallel
              Real scal = crossprod*crossprod; // null if x is on the line

              if( !this->isSmall(scal,100) ){
                  succeed = false;
                  ADD_FAILURE() << "Position of constrained particle " << i << " is wrong: " << x[i] ;
              }
               it++;
           }
           else           // unconstrained particle: check that it has not changed
           {
              CPos dx = x[i]-xprev[i];
              Real scal = dx*dx;

              if( !this->isSmall(scal,100) ){
                  succeed = false;
                  ADD_FAILURE() << "Position of unconstrained particle " << i << " is wrong: " << x[i] ;
              }
           }

       }
       return succeed;
    }

    // Test project velocity
    bool test_projectVelocity()
    {
       VecDeriv vprev(numNodes);
       typename MechanicalObject::WriteVecDeriv v = dofs->writeVelocities();
       for (unsigned i=0; i<numNodes; i++){
           vprev[i] = v[i] = CPos(i,0,0);
       }

       projection->projectVelocity(core::mechanicalparams::defaultInstance(), *dofs->write(core::vec_id::write_access::velocity) );

       bool succeed=true;
       typename Indices::const_iterator it = indices.begin(); // must be sorted
       for(unsigned i=0; i<numNodes; i++ )
       {
          if ((it!=indices.end()) && ( i==*it ))  // constrained particle
           {
              CPos crossprod = v[i].cross(direction); // should be parallel
              Real scal = crossprod.norm(); // null if v is ok

              if( !this->isSmall(scal,100) ){
                  succeed = false;
                  ADD_FAILURE() << "Velocity of constrained particle " << i << " is wrong: " << v[i] ;
              }
               it++;
           }
           else           // unconstrained particle: check that it has not changed
           {
              CPos dv = v[i]-vprev[i];
              Real scal = dv*dv;
              if( !this->isSmall(scal,100) ){
                  succeed = false;
                  ADD_FAILURE() << "Velocity of unconstrained particle " << i << " is wrong: " << v[i] ;
              }
           }

       }
       return succeed;
    }

    void doTearDown() override
    {
        if (root!=nullptr)
            sofa::simulation::node::unload(root);
    }


 };


// Define the list of DataTypes to instantiate
using ::testing::Types;
typedef Types<
    Vec3Types
> DataTypes; // the types to instantiate.

// Test suite for all the instantiations
TYPED_TEST_SUITE(DirectionProjectiveConstraint_test, DataTypes);

// first test case
TYPED_TEST( DirectionProjectiveConstraint_test , oneConstrainedParticle )
{
    EXPECT_MSG_NOEMIT(Error) ;
    this->init_oneConstrainedParticle();
    ASSERT_TRUE(  this->test_projectPosition() );
    ASSERT_TRUE(  this->test_projectVelocity() );
}
// second test case
TYPED_TEST( DirectionProjectiveConstraint_test , allParticlesConstrained )
{
    EXPECT_MSG_NOEMIT(Error) ;
    this->init_allParticlesConstrained();
    ASSERT_TRUE(  this->test_projectPosition() );
    ASSERT_TRUE(  this->test_projectVelocity() );
}

} // anonymous namespace

} // namespace sofa

