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
#include <sofa/component/solidmechanics/testing/ForceFieldTestCreation.h>
#include <sofa/component/solidmechanics/fem/elastic/TetrahedronFEMForceField.h>
#include <sofa/simulation/common/SceneLoaderXML.h>


namespace sofa
{

/**
 *  Test suite for TetrahedronFEMForceField.
 */
template <typename _TetrahedronFEMForceField>
struct TetrahedronFEMForceField_stepTest : public ForceField_test<_TetrahedronFEMForceField>
{

    typedef _TetrahedronFEMForceField ForceType;
    typedef ForceField_test<_TetrahedronFEMForceField> Inherited;
    typedef typename ForceType::DataTypes DataTypes;

    typedef typename ForceType::VecCoord VecCoord;
    typedef typename ForceType::VecDeriv VecDeriv;
    typedef typename ForceType::Coord Coord;
    typedef typename ForceType::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef type::Vec<3,Real> Vec3;

    VecCoord x;
    VecDeriv v,f;

    /** @name Test_Cases
      For each of these cases, we check if the accurate forces are computed
    */
    TetrahedronFEMForceField_stepTest():Inherited::ForceField_test(std::string(SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_TEST_SCENES_DIR) + "/" + "TetrahedronFEMForceFieldRegularTetra.scn")
    {
        //Position
        x.resize(4);
        DataTypes::set( x[0], 0,0,0);
        DataTypes::set( x[1], 1,0,0);
        DataTypes::set( x[2], (Real)0.5, (Real)0.8660254037, (Real)0);
        //DataTypes::set( x[3], (SReal)0.5, (SReal)0.28867,(SReal)1.632993);
        DataTypes::set( x[3], (Real)0.5, (Real)0.28867513,(Real)2);
        //Velocity
        v.resize(4);
        DataTypes::set( v[0], 0,0,0);
        DataTypes::set( v[1], 0,0,0);
        DataTypes::set( v[2], 0,0,0);
        DataTypes::set( v[3], 0,0,0);

        //Force e*E*S*1/3  = 1*40*sqrt(3)/4*1/3
        f.resize(4);
        Vec3 fup(0,0,std::sqrt(3.)*10.0/3.0);
        Vec3 fdown(0,0,std::sqrt(3.)*10.0/9.0);
        DataTypes::set( f[0],  fdown[0], fdown[1], (Real)fdown[2]);
        DataTypes::set( f[1],  fdown[0], fdown[1], (Real)fdown[2]);
        DataTypes::set( f[2],  fdown[0], fdown[1], (Real)fdown[2]);
        DataTypes::set( f[3],  -fup[0], -fup[1], -(Real)fup[2]);

        // Set force parameters
        this->force->setPoissonRatio(0);
        this->force->d_youngModulus.setValue({40});
        this->force->d_method.setValue("small");

        // Init simulation
        sofa::simulation::node::initRoot(Inherited::node.get());
    }

    //Test the value of the force it should be equal for each vertex to Pressure*area/4
    void test_valueForce()
    {
        // run the forcefield_test
        Inherited::run_test( x, v, f );
    }


};


// ========= Define the list of types to instanciate.
//using ::testing::Types;
typedef ::testing::Types<
sofa::component::solidmechanics::fem::elastic::TetrahedronFEMForceField<defaulttype::Vec3Types>
> TestTypes; // the types to instanciate.



// ========= Tests to run for each instanciated type
TYPED_TEST_SUITE(TetrahedronFEMForceField_stepTest, TestTypes);

// test case
TYPED_TEST(TetrahedronFEMForceField_stepTest, extension )
{
    this->errorMax *= 1e6;
    this->deltaRange = std::make_pair( 1, this->errorMax * 10 );
    this->debug = false;

    // Young modulus, poisson ratio method

    // run test
    this->test_valueForce();
}

}
