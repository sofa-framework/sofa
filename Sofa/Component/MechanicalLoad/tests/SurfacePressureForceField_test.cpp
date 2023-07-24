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

//Force field
#include <sofa/component/mechanicalload/SurfacePressureForceField.h>

#include <sofa/component/solidmechanics/testing/ForceFieldTestCreation.h>


namespace sofa
{

/**  Test SurfacePressureForceField.
  */
template <typename SurfacePressureForceFieldType>
struct SurfacePressureForceField_test : public ForceField_test<SurfacePressureForceFieldType>
{
    typedef ForceField_test<SurfacePressureForceFieldType> Inherited;
    typedef SurfacePressureForceFieldType ForceType;
    typedef typename ForceType::DataTypes DataTypes;

    typedef typename ForceType::VecCoord VecCoord;
    typedef typename ForceType::VecDeriv VecDeriv;
    typedef typename ForceType::Coord Coord;
    typedef typename ForceType::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef type::Vec<3, Real> Vec3;

    VecCoord x;
    VecDeriv v, f;

    SurfacePressureForceField_test()
        : Inherited::ForceField_test(std::string(SOFA_COMPONENT_MECHANICALLOAD_TEST_SCENES_DIR) + "/" + "SurfacePressureForceField.scn")
    {
        // potential energy is not implemented and won't be tested
        this->flags &= ~Inherited::TEST_POTENTIAL_ENERGY;

        // Set vectors, using DataTypes::set to cope with tests in dimension 2
        //Position
        x.resize(3);
        DataTypes::set(x[0], 0, 0, 0);
        DataTypes::set(x[1], 1, 0, 0);
        DataTypes::set(x[2], 1, 1, 0);

        //Velocity
        v.resize(3);
        DataTypes::set(v[0], 0, 0, 0);
        DataTypes::set(v[1], 0, 0, 0);
        DataTypes::set(v[2], 0, 0, 0);

        //Force
        f.resize(3);
        Vec3 f0(0, 0, 0.1);
        DataTypes::set(f[0], f0[0], f0[1], f0[2]);
        DataTypes::set(f[1], f0[0], f0[1], f0[2]);
        DataTypes::set(f[2], f0[0], f0[1], f0[2]);

        // Set the properties of the force field
        Inherited::force->m_pressure.setValue(0.6);
    }

    //Test the value of the force it should be equal for each vertex to Pressure*area/4
    void test_valueForce()
    {
        // run the forcefield_test
        Inherited::run_test(x, v, f);
    }

    // Test that the force value is constant
    void test_constantForce()
    {
        sofa::simulation::node::initRoot(Inherited::node.get());

        // Do a few animation steps
        for (int k = 0; k < 10; k++)
        {
            sofa::simulation::node::animate(Inherited::node.get(), 0.5);
        }

        // run the forcefield_test
        Inherited::run_test(x, v, f, false);
    }

};


// Types to instantiate.
typedef ::testing::Types<
    component::mechanicalload::SurfacePressureForceField<defaulttype::Vec3Types>
> TestTypes;


// Tests to run for each instantiated type
TYPED_TEST_SUITE(SurfacePressureForceField_test, TestTypes);

// first test case: test force value
TYPED_TEST(SurfacePressureForceField_test, surfacePressureForceFieldTest)
{
    this->errorMax = 1;
    this->deltaRange.first = 1e7;
    this->deltaRange.second = 1e8;

    this->test_valueForce();
}

// second test case: test that force is constant
TYPED_TEST(SurfacePressureForceField_test, constantSurfacePressureForceFieldTest)
{
    this->errorMax = 1;
    this->deltaRange.first = 1e7;
    this->deltaRange.second = 1e8;

    this->test_constantForce();
}
} // namespace sofa
