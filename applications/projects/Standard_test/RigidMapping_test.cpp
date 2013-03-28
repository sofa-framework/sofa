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

/* Francois Faure, 2013 */

#include "Mapping_test.h"
#include <sofa/component/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/component/mapping/RigidMapping.h>
#include <sofa/component/container/MechanicalObject.h>


namespace sofa {

using std::cout;
using std::cerr;
using std::endl;
using namespace core;
using namespace component;
using defaulttype::Vec;
using defaulttype::Mat;


/**  Test suite for RigidMapping.
The test cases are defined in the #Test_Cases member group.
  */
template <typename _RigidMapping>
struct RigidMapping_test : public Mapping_test<typename _RigidMapping::In, typename _RigidMapping::Out>
{

    typedef _RigidMapping RigidMapping;

    typedef typename RigidMapping::In InDataTypes;
    typedef typename InDataTypes::VecCoord InVecCoord;
    typedef typename InDataTypes::VecDeriv InVecDeriv;
    typedef typename InDataTypes::Coord InCoord;
    typedef typename InDataTypes::Deriv InDeriv;
    typedef container::MechanicalObject<InDataTypes> InMechanicalObject;
    typedef typename InMechanicalObject::ReadVecCoord  ReadInVecCoord;
    typedef typename InMechanicalObject::WriteVecCoord WriteInVecCoord;
    typedef typename InMechanicalObject::WriteVecDeriv WriteInVecDeriv;
    typedef typename InCoord::Pos Translation;
    typedef typename InCoord::Rot Rotation;
    typedef typename InDataTypes::Real InReal;
    typedef Mat<InDataTypes::spatial_dimensions,InDataTypes::spatial_dimensions,InReal> RotationMatrix;


    typedef typename RigidMapping::Out OutDataTypes;
    typedef typename OutDataTypes::VecCoord OutVecCoord;
    typedef typename OutDataTypes::VecDeriv OutVecDeriv;
    typedef typename OutDataTypes::Coord OutCoord;
    typedef typename OutDataTypes::Deriv OutDeriv;
    typedef container::MechanicalObject<OutDataTypes> OutMechanicalObject;
    typedef typename OutMechanicalObject::WriteVecCoord WriteOutVecCoord;
    typedef typename OutMechanicalObject::WriteVecDeriv WriteOutVecDeriv;
    typedef typename OutMechanicalObject::ReadVecCoord ReadOutVecCoord;
    typedef typename OutMechanicalObject::ReadVecDeriv ReadOutVecDeriv;

    simulation::Node::SPtr root;                 ///< Root of the scene graph, created by the constructor an re-used in the tests
    simulation::Simulation* simulation;          ///< created by the constructor an re-used in the tests

    typename RigidMapping::SPtr rigidMapping;
    typename InMechanicalObject::SPtr inDofs;
    typename OutMechanicalObject::SPtr outDofs;
//    OutVecCoord expectedChildCoords;   ///< expected child positions after apply
//    OutVecDeriv expectedChildVels;     ///< expected child velocities after apply

    /// Create the context for the matrix tests.
    void SetUp()
    {
        sofa::component::init();
        sofa::simulation::setSimulation(simulation = new sofa::simulation::graph::DAGSimulation());

        /// Parent node
        root = simulation->createNewGraph("root");

        inDofs = addNew<InMechanicalObject>(root);

        /// Child node
        simulation::Node::SPtr childNode = root->createChild("childNode");
        outDofs = addNew<OutMechanicalObject>(childNode);
        rigidMapping = addNew<RigidMapping>(root);
        rigidMapping->setModels(inDofs.get(),outDofs.get());
        this->setMapping(rigidMapping);


    }

    /** @name Test_Cases
      For each of these cases, we can test if the mapping work
      */
    ///@{
    /** Map particles in local coords to a single frame.
    */
    void init_oneRigid_fourParticles_localCoords()
    {
        inDofs->resize(1);
        outDofs->resize(4);

        // child positions
        rigidMapping->globalToLocalCoords.setValue(false); // initial child positions are given in local coordinates
        WriteOutVecCoord xout = outDofs->writePositions();
        WriteOutVecDeriv vout = outDofs->writeVelocities();
        // vertices of the unit tetrahedron
        OutDataTypes::set( xout[0] ,0.,0.,0.);
//        OutDataTypes::set( vout[0] ,0.,0.,0.);
        OutDataTypes::set( xout[1] ,1.,0.,0.);
//        OutDataTypes::set( vout[1] ,1.,0.,0.);
        OutDataTypes::set( xout[2] ,0.,1.,0.);
//        OutDataTypes::set( vout[2] ,0.,1.,0.);
        OutDataTypes::set( xout[3] ,0.,0.,1.);
//        OutDataTypes::set( vout[3] ,0.,0.,1.);

        // parent position
        WriteInVecCoord xin = inDofs->writePositions();
        InDataTypes::set( xin[0], 1.,-2.,3. );
        Rotation rot = InDataTypes::rotationEuler(-1.,2.,-3.);
        InDataTypes::setCRot( xin[0], rot );

//        // parent velocity
//        WriteInVecDeriv vin = inDofs->writeVelocities();

        // expected mapped values
        this->expectedChildCoords.resize(xout.size());
//        expectedChildVels.  resize(xout.size());
        RotationMatrix m;
        xin[0].writeRotationMatrix(m);
        for(unsigned i=0; i<xout.size(); i++ )
        {
            // note that before init, xout is still in relative coordinates
            this->expectedChildCoords[i] = xin[0].getCenter() + m * xout[i];
//            expectedChildVels  [i] = vin[0].velocityAtRotatedPoint( m*xout[i] );
        }


        /// Init
        sofa::simulation::getSimulation()->init(root.get());
    }

    // no other test case up to now

    ///@}



    void TearDown()
    {
        if (root!=NULL)
            sofa::simulation::getSimulation()->unload(root);
        //        cerr<<"tearing down"<<endl;
    }


};


// Define the list of types to instanciate. We do not necessarily need to test all combinations.
using testing::Types;
typedef Types<
mapping::RigidMapping<defaulttype::Rigid2fTypes,defaulttype::Vec2fTypes>,
mapping::RigidMapping<defaulttype::Rigid3dTypes,defaulttype::Vec3dTypes>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(RigidMapping_test, DataTypes);
// first test case
TYPED_TEST( RigidMapping_test , oneRigid_fourParticles_localCoords )
{
    this->init_oneRigid_fourParticles_localCoords();
    ASSERT_TRUE(  this->test_apply() );
    ASSERT_TRUE(  this->test_applyJ() );
}
//// next test case
//TYPED_TEST( RigidMapping_test , allParticlesConstrained )
//{
//    this->init_allParticlesConstrained();
//    ASSERT_TRUE(  this->test_projectPosition() );
//    ASSERT_TRUE(  this->test_projectVelocity() );
//}

} // namespace sofa
