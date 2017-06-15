/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <Compliant/forcefield/DiagonalStiffness.h>
#include <SofaTest/ForceField_test.h>

#include <SofaTest/TestMessageHandler.h>
using sofa::helper::logging::Message ;

#include <SofaSimulationGraph/DAGSimulation.h>
using sofa::core::objectmodel::ComponentState ;
using sofa::core::objectmodel::BaseObject ;
using sofa::simulation::Simulation ;
using sofa::simulation::Node ;



namespace sofa {

using namespace modeling;

/**  Test suite for DiagonalStiffness
  */
template <typename _DiagonalStiffness>
struct DiagonalStiffness_test : public ForceField_test<_DiagonalStiffness>
{

    typedef _DiagonalStiffness ForceType;
    typedef ForceField_test<_DiagonalStiffness> Inherited;
    typedef typename ForceType::DataTypes DataTypes;

    typedef typename ForceType::VecCoord VecCoord;
    typedef typename ForceType::VecDeriv VecDeriv;

    VecCoord x;
    VecDeriv v,f;

    /** @name Test_Cases
      For each of these cases, we check if the accurate forces are computed
    */
    DiagonalStiffness_test():Inherited::ForceField_test()
    {
        this->errorFactorPotentialEnergy = 3; // increading tolerance for potential energy test due to non-linearities

        //Position
        x.resize(3);
        DataTypes::set( x[0], 0,0,0);
        DataTypes::set( x[1], -1,0,0);
        DataTypes::set( x[2], 7,0,0);

        //Velocity
        v.resize(3);
        DataTypes::set( v[0], 0,0,0);
        DataTypes::set( v[1], 0,0,0);
        DataTypes::set( v[2], 0,0,0);

        //Force
        f.resize(3);
        DataTypes::set( f[0], 0,0,0 );
        DataTypes::set( f[1], 100,0,0 );
        DataTypes::set( f[2], -49,0,0 );

        // Set force parameters
        VecDeriv& stiffnesses = *Inherited::force->diagonal.beginWriteOnly();
        stiffnesses.resize(3);
        stiffnesses[0] = 10;
        stiffnesses[1] = 100;
        stiffnesses[2] = 7;
        Inherited::force->diagonal.endEdit();
    }

    //Test the value of the force it should be equal for each vertex to Pressure*area/4
    void test_valueForce()
    {
        // run the forcefield_test
        Inherited::run_test( x, v, f );
    }

};

// ========= Define the list of types to instanciate.
//using testing::Types;
typedef testing::Types<
component::forcefield::DiagonalStiffness<defaulttype::Vec1Types>
> TestTypes; // the types to instanciate.





// ========= Tests to run for each instanciated type
TYPED_TEST_CASE(DiagonalStiffness_test, TestTypes);




// test case
TYPED_TEST( DiagonalStiffness_test , extension )
{
    this->debug = false;

    // run test
    this->test_valueForce();
}


} // namespace sofa
