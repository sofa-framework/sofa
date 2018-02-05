/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
// Base class
#include <SofaTest/ForceField_test.h>
//Force field
#include <SofaBoundaryCondition/TrianglePressureForceField.h>
#include <SofaBaseTopology/TopologySparseData.inl>

#include <SofaTest/TestMessageHandler.h>


namespace sofa {

/**  Test TrianglePressureForceField.
  */
template <typename _TrianglePressureForceField>
struct TrianglePressureForceField_test : public ForceField_test<_TrianglePressureForceField>
{
    typedef ForceField_test<_TrianglePressureForceField> Inherited;
    typedef _TrianglePressureForceField ForceType;
    typedef typename ForceType::DataTypes DataTypes;

    typedef typename ForceType::VecCoord VecCoord;
    typedef typename ForceType::VecDeriv VecDeriv;
    typedef typename ForceType::Coord Coord;
    typedef typename ForceType::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef defaulttype::Vec<3,Real> Vec3;

    VecCoord x;
    VecDeriv v,f;

    TrianglePressureForceField_test(): Inherited::ForceField_test(std::string(SOFABOUNDARYCONDITION_TEST_SCENES_DIR) + "/" + "TrianglePressureForceField.scn")
    {
        // potential energy is not implemented and won't be tested
        this->flags &= ~Inherited::TEST_POTENTIAL_ENERGY;

        // Set vectors, using DataTypes::set to cope with tests in dimension 2
        //Position
        x.resize(3);
        DataTypes::set( x[0], 0,0,0);
        DataTypes::set( x[1], 1,0,0);
        DataTypes::set( x[2], 1,1,0);

        //Velocity
        v.resize(3);
        DataTypes::set( v[0], 0,0,0);
        DataTypes::set( v[1], 0,0,0);
        DataTypes::set( v[2], 0,0,0);

        //Force
         f.resize(3);
        Vec3 f0(0,0,0.1);
        DataTypes::set( f[0],  f0[0], f0[1], f0[2]);
        DataTypes::set( f[1],  f0[0], f0[1], f0[2]);
        DataTypes::set( f[2],  f0[0], f0[1], f0[2]);

        // Set the properties of the force field
        Inherited::force->normal.setValue(Deriv(0,0,1));
        Inherited::force->dmin.setValue(-0.01);
        Inherited::force->dmax.setValue(0.01);
        Inherited::force->pressure=Coord(0,0,0.6);
    }

    //Test the value of the force it should be equal for each vertex to Pressure*area/4
    void test_valueForce()
    {
        // run the forcefield_test
        Inherited::run_test( x, v, f );
    }

    // Test that the force value is constant
    void test_constantForce()
    {
        sofa::simulation::getSimulation()->init(Inherited::node.get());

        // Do a few animation steps
        for(int k=0;k<10;k++)
        {
            sofa::simulation::getSimulation()->animate(Inherited::node.get(),0.5);
        }

        // run the forcefield_test
        Inherited::run_test( x, v, f );
    }

};

// Types to instantiate.
typedef testing::Types<
    component::forcefield::TrianglePressureForceField<defaulttype::Vec3Types>
> TestTypes;



// Tests to run for each instantiated type
TYPED_TEST_CASE(TrianglePressureForceField_test, TestTypes);

// first test case: test force value
TYPED_TEST( TrianglePressureForceField_test , trianglePressureForceFieldTest)
{
    this->errorMax *= 10;
    this->debug = false;

    this->test_valueForce();
}

// second test case: test that force is constant
TYPED_TEST( TrianglePressureForceField_test , constantTrianglePressureForceFieldTest)
{
    this->errorMax *= 10;
    this->debug = false;

    this->test_constantForce();
}
}// namespace sofa
