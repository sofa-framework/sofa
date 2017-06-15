/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaTest/Sofa_test.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <SofaBaseTopology/PointSetTopologyContainer.h>
#include <SofaBoundaryCondition/ProjectToLineConstraint.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/defaulttype/VecTypes.h>

#include <SofaTest/TestMessageHandler.h>


namespace sofa {
using namespace component;
using namespace defaulttype;



/**  Test suite for ProjectToLineConstraint.
The test cases are defined in the #Test_Cases member group.
  */
template <typename _DataTypes>
struct ProjectToLineConstraint_test : public Sofa_test<typename _DataTypes::Real>
{
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::CPos CPos;
    typedef typename Coord::value_type Real;
    typedef projectiveconstraintset::ProjectToLineConstraint<DataTypes> ProjectToLineConstraint;
    typedef typename ProjectToLineConstraint::Indices Indices;
    typedef component::topology::PointSetTopologyContainer PointSetTopologyContainer;
    typedef container::MechanicalObject<DataTypes> MechanicalObject;

    simulation::Node::SPtr root;                 ///< Root of the scene graph, created by the constructor an re-used in the tests
    simulation::Simulation* simulation;          ///< created by the constructor an re-used in the tests

    unsigned numNodes;                         ///< number of particles used for the test
    Indices indices;                           ///< indices of the nodes to project
    CPos origin;                               ///< origin of the plane to project to
    CPos direction;                            ///< direction of the line to project to
    typename ProjectToLineConstraint::SPtr projection;
    typename MechanicalObject::SPtr dofs;

    /// Create the context for the tests.
    void SetUp()
    {
//        if( sofa::simulation::getSimulation()==NULL )
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

        /// Create the scene
        root = simulation->createNewGraph("root");

        PointSetTopologyContainer::SPtr topology = core::objectmodel::New<PointSetTopologyContainer>();
        root->addObject(topology);

        dofs = core::objectmodel::New<MechanicalObject>();
        root->addObject(dofs);

        projection = core::objectmodel::New<ProjectToLineConstraint>();
        root->addObject(projection);

        /// Set the values
        numNodes = 3;
        dofs->resize(numNodes);


        origin = CPos(0,0,0);
        projection->f_origin.setValue(origin);
        direction = CPos(1,1,1);
        projection->f_direction.setValue(direction);

    }

    /** @name Test_Cases
      For each of these cases, we can test if the projections work
      */
    ///@{
    /** Constrain one particle, and not the last one.
    Detects bugs like not setting the projection matrix entries beyond the last constrained particle
    */
    void init_oneConstrainedParticle()
    {
        indices.clear();
        indices.push_back(1);
        std::sort(indices.begin(),indices.end()); // checking vectors in linear time requires sorted indices
        projection->f_indices.setValue(indices);

        /// Init
        sofa::simulation::getSimulation()->init(root.get());
    }

    /** Constrain all the particles.
    */
    void init_allParticlesConstrained()
    {
        indices.clear();
        for(unsigned i = 0; i<numNodes; i++)
            indices.push_back(i);
         projection->f_indices.setValue(indices);

         /// Init
         sofa::simulation::getSimulation()->init(root.get());
    }
    ///@}


    bool test_projectPosition()
    {
       VecCoord xprev(numNodes);
       typename MechanicalObject::WriteVecCoord x = dofs->writePositions();
       for (unsigned i=0; i<numNodes; i++){
           xprev[i] = x[i] = CPos(i,0,0);
       }
//       cerr<<"test_projectPosition, x before = " << x << endl;
       projection->projectPosition(core::MechanicalParams::defaultInstance(), *dofs->write(core::VecCoordId::position()) );
//       cerr<<"test_projectPosition, x after = " << x << endl;

       bool succeed=true;
       typename Indices::const_iterator it = indices.begin(); // must be sorted
       for(unsigned i=0; i<numNodes; i++ )
       {
           if ((it!=indices.end()) && ( i==*it ))  // constrained particle
           {
              CPos crossprod = (x[i]-origin).cross(direction); // should be parallel
              Real scal = crossprod*crossprod; // null if x is on the line
//              cerr<<"scal = "<< scal << endl;
              if( !Sofa_test<typename _DataTypes::Real>::isSmall(scal,100) ){
                  succeed = false;
                  ADD_FAILURE() << "Position of constrained particle " << i << " is wrong: " << x[i] ;
              }
               it++;
           }
           else           // unconstrained particle: check that it has not changed
           {
              CPos dx = x[i]-xprev[i];
              Real scal = dx*dx;
//              cerr<<"scal gap = "<< scal << endl;
              if( !Sofa_test<typename _DataTypes::Real>::isSmall(scal,100) ){
                  succeed = false;
                  ADD_FAILURE() << "Position of unconstrained particle " << i << " is wrong: " << x[i] ;
              }
           }

       }
       return succeed;
    }

    bool test_projectVelocity()
    {
       VecDeriv vprev(numNodes);
       typename MechanicalObject::WriteVecDeriv v = dofs->writeVelocities();
       for (unsigned i=0; i<numNodes; i++){
           vprev[i] = v[i] = CPos(i,0,0);
       }
//       cerr<<"test_projectVelocity, v before = " << v << endl;
       projection->projectVelocity(core::MechanicalParams::defaultInstance(), *dofs->write(core::VecDerivId::velocity()) );
//       cerr<<"test_projectVelocity, v after = " << v << endl;

       bool succeed=true;
       typename Indices::const_iterator it = indices.begin(); // must be sorted
       for(unsigned i=0; i<numNodes; i++ )
       {
          if ((it!=indices.end()) && ( i==*it ))  // constrained particle
           {
              CPos crossprod = v[i].cross(direction); // should be parallel
              Real scal = crossprod.norm(); // null if v is ok
//              cerr<<"scal = "<< scal << endl;
              if( !Sofa_test<typename _DataTypes::Real>::isSmall(scal,100) ){
                  succeed = false;
                  ADD_FAILURE() << "Velocity of constrained particle " << i << " is wrong: " << v[i] ;
              }
               it++;
           }
           else           // unconstrained particle: check that it has not changed
           {
              CPos dv = v[i]-vprev[i];
              Real scal = dv*dv;
//              cerr<<"scal gap = "<< scal << endl;
              if( !Sofa_test<typename _DataTypes::Real>::isSmall(scal,100) ){
                  succeed = false;
                  ADD_FAILURE() << "Velocity of unconstrained particle " << i << " is wrong: " << v[i] ;
              }
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
    Vec3Types
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(ProjectToLineConstraint_test, DataTypes);
// first test case
TYPED_TEST( ProjectToLineConstraint_test , oneConstrainedParticle )
{
    EXPECT_MSG_NOEMIT(Error) ;
    this->init_oneConstrainedParticle();
    ASSERT_TRUE(  this->test_projectPosition() );
    ASSERT_TRUE(  this->test_projectVelocity() );
}
// next test case
TYPED_TEST( ProjectToLineConstraint_test , allParticlesConstrained )
{
    EXPECT_MSG_NOEMIT(Error) ;
    this->init_allParticlesConstrained();
    ASSERT_TRUE(  this->test_projectPosition() );
    ASSERT_TRUE(  this->test_projectVelocity() );
}


} // namespace sofa

