#include "stdafx.h"

// Base class
#include "ForceField_test.h"
//Force field
#include <SofaBoundaryCondition/TrianglePressureForceField.h>
#include <SofaBaseTopology/TopologySparseData.inl>

namespace sofa {

/**  Test QuadPressureForceField.
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
    typedef helper::Vec<3,Real> Vec3;

    VecCoord x;
    VecDeriv v,f;

    TrianglePressureForceField_test(): Inherited::ForceField_test(std::string(SOFATEST_SCENES_DIR) + "/" + "TrianglePressureForceField.scn")
    {
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
    component::forcefield::TrianglePressureForceField<defaulttype::Vec3dTypes>
> TestTypes; 



// Tests to run for each instantiated type
TYPED_TEST_CASE(TrianglePressureForceField_test, TestTypes);

// first test case: test force value
TYPED_TEST( TrianglePressureForceField_test , trianglePressureForceFieldTest)
{
    this->errorMax = 1000;
    this->deltaMax = 1000;
    this->debug = false;

    this->test_valueForce();
}

// second test case: test that force is constant
TYPED_TEST( TrianglePressureForceField_test , constantTrianglePressureForceFieldTest)
{
    this->errorMax = 1000;
    this->deltaMax = 1000;
    this->debug = false;

    this->test_constantForce();
}
}// namespace sofa
