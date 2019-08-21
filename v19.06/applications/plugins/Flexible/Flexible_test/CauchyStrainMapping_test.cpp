#include "../strainMapping/CauchyStrainMapping.h"

#include "StrainMapping_test.h"


namespace sofa {


    template <typename _Mapping>
    struct CauchyStrainMappingTest : public Mapping_test<_Mapping>
    {
        /* Test the cauchy strain mapping:
        * Create a deformation gradient F. Then the strain E is mapped from the deformation gradient as:
        - \f$ E = [ F + F^T ]/2 - I  \f$
        * The expected mapped values should be equal to the strain. 
        * Note that the strain is actually stored into vectors using Voigt notation. 
        */
        
        typedef Mapping_test<_Mapping> Inherited;
        typedef typename Inherited::In In;
        typedef typename Inherited::Real Real;
        typedef typename Inherited::OutVecCoord OutVecCoord;
        typedef typename Inherited::InVecCoord InVecCoord;
        typedef typename In::Frame InFrame;


        using Inherited::runTest;
        bool runTest()
        {
            defaulttype::Mat<In::material_dimensions,In::material_dimensions,Real> strain; 

            // create a deformation gradient
            for( unsigned int i=0 ; i<In::material_dimensions ; ++i )
                for( unsigned int j=i ; j<In::material_dimensions ; ++j )
                {
                    strain[i][j] = (i+1)*2+j*0.3; 
                }

                defaulttype::Mat<In::material_dimensions,In::material_dimensions,Real> defo( strain );

                //Green Lagrange Tensor E = (strain.transpose()+strain)/2 - Identity
                defo = (strain + strain.transposed())*0.5 - strain.s_identity;

                // expected mapped values
                OutVecCoord expectedChildCoords(1);
                expectedChildCoords[0].getVec() = defaulttype::StrainMatToVoigt( defo );

                InVecCoord xin(1);
                OutVecCoord xout(1);

                // parent position
                InFrame &f = xin[0].getF();

                // stretch + shear
                for( unsigned int i=0 ; i<In::material_dimensions ; ++i )
                {
                    for( unsigned int j=0 ; j<In::material_dimensions ; ++j )
                    {
                        f[i][j] = strain[i][j];
                    }
                }

//                cerr<<"CauchyStrainMappingTest::runTest, f="<< f << endl;
//                cerr<<"CauchyStrainMappingTest::runTest, expected="<< expectedChildCoords << endl;

                return Inherited::runTest(xin,xout,xin,expectedChildCoords);

        }

    };

    // Define the list of types to instanciate.
    typedef Types<
        CauchyStrainMapping<defaulttype::F331Types,defaulttype::E331Types>,
        CauchyStrainMapping<defaulttype::F321Types,defaulttype::E321Types>,
        CauchyStrainMapping<defaulttype::F311Types,defaulttype::E311Types>,
        CauchyStrainMapping<defaulttype::F332Types,defaulttype::E332Types>
    > CauchyDataTypes; // the types to instanciate.

    // Test suite for all the instanciations
    TYPED_TEST_CASE(CauchyStrainMappingTest, CauchyDataTypes);
    // first test case
    TYPED_TEST( CauchyStrainMappingTest , test_auto )
    {
        ASSERT_TRUE( this->runTest() );
    }


} // namespace sofa
