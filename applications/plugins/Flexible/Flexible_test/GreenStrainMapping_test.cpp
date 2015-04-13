#include "../strainMapping/GreenStrainMapping.h"

#include "StrainMapping_test.h"

namespace sofa {



    template <typename _Mapping>
    struct GreenStrainMappingTest : public Mapping_test<_Mapping>
    {
        /* Test the green strain mapping:
        * Create a deformation gradient F. Then the strain E is mapped from the deformation gradient as:
        - \f$ E = [ F^T.F - I ]/2  \f$*
        * The expected mapped values should be equal to the strain. 
        * Note that the strain is actually stored into vectors using Voigt notation. 
        */
        
        typedef Mapping_test<_Mapping> Inherited;
        typedef typename Inherited::In In;
        typedef typename Inherited::Real Real;
        typedef typename Inherited::OutVecCoord OutVecCoord;
        typedef typename Inherited::InVecCoord InVecCoord;
        typedef typename In::Frame InFrame;


        bool runTest()
        {
            this->deltaRange = std::make_pair( 100, 10000 );
            this->errorMax = this->deltaRange.second;
            this->errorFactorDJ = 500;

            defaulttype::Mat<In::material_dimensions,In::material_dimensions,Real> strain; 

            // create a deformation gradient
            for( unsigned int i=0 ; i<In::material_dimensions ; ++i )
                for( unsigned int j=i ; j<In::material_dimensions ; ++j )
                {
                    strain[i][j] = (i+1)*2+j*0.3; 
                }

                defaulttype::Mat<In::material_dimensions,In::material_dimensions,Real> defo( strain );

                //Green Lagrange Tensor E = 0.5*(strain.transpose()*strain - Identity)
                defo = ((strain.transposed())*strain - strain.Identity())*0.5;

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

//                cerr<<"StrainMappingTest::runTest, f="<< f << endl;
//                cerr<<"StrainMappingTest::runTest, expected="<< expectedChildCoords << endl;

                return Inherited::runTest(xin,xout,xin,expectedChildCoords);

        }

    };

    // Define the list of types to instanciate.
    typedef Types<
        GreenStrainMapping<defaulttype::F331Types,defaulttype::E331Types>,
        GreenStrainMapping<defaulttype::F321Types,defaulttype::E321Types>,
        GreenStrainMapping<defaulttype::F311Types,defaulttype::E311Types>,
        GreenStrainMapping<defaulttype::F332Types,defaulttype::E332Types>,
        GreenStrainMapping<defaulttype::F332Types,defaulttype::E333Types>
    > GreenDataTypes; // the types to instanciate.

    // Test suite for all the instanciations
    TYPED_TEST_CASE(GreenStrainMappingTest, GreenDataTypes);
    // first test case
    TYPED_TEST( GreenStrainMappingTest , test_auto )
    {
        ASSERT_TRUE( this->runTest() );
    }



} // namespace sofa
