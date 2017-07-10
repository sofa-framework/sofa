
#include <SofaTest/Mapping_test.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <Compliant/mapping/DifferenceFromTargetMapping.h>


/**  Test suite for DifferenceFromTargetMapping
  */
template <typename Mapping>
struct DifferenceFromTargetMappingTest : public sofa::Mapping_test<Mapping>
{

    typedef DifferenceFromTargetMappingTest self;
    typedef sofa::Mapping_test<Mapping> base;

    typedef sofa::defaulttype::Vec<3,SReal> Vec3;

    Mapping* mapping;

    DifferenceFromTargetMappingTest() {
        mapping = static_cast<Mapping*>(this->base::mapping);
    }

    bool test()
    {
        // parents
        typename self::InVecCoord xin(2);
        self::In::set( xin[0], 0,0,0 );
        self::In::set( xin[1], -5,6,7 );

        // child
        typename self::OutVecCoord expected(2);
        self::Out::set( expected[0], 1,1,1 );
        self::Out::set( expected[1], 1,1,1 );

        // mapping parameters
        typename self::OutVecCoord targets(2);
        self::Out::set( targets[0], -1,-1,-1 );
        self::Out::set( targets[1], -6,5,6 );
        mapping->targets.setValue(targets);

        return this->runTest(xin, expected);
    }

};


// Define the list of types to instanciate.
using testing::Types;
typedef Types<
    sofa::component::mapping::DifferenceFromTargetMapping<sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types>,
    sofa::component::mapping::DifferenceFromTargetMapping<sofa::defaulttype::Rigid3Types, sofa::defaulttype::Vec3Types>,
    sofa::component::mapping::DifferenceFromTargetMapping<sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec1Types>
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(DifferenceFromTargetMappingTest, DataTypes);

TYPED_TEST( DifferenceFromTargetMappingTest, test )
{
    ASSERT_TRUE( this->test() );
}

