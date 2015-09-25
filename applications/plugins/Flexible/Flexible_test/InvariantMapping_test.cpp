#include "../strainMapping/InvariantMapping.h"

#include "StrainMapping_test.h"


namespace sofa {


    template <typename _Mapping>
    struct InvariantMappingTest : public Mapping_test<_Mapping>
    {
        /* Test the invariant strain mapping:
        * Create a deformation gradient F. Then the strain E is mapped from the deformation gradient as:
        - \f$ E -> [ I1 , I2, J ] \f$
        where:
        - \f$ I1 = trace(C) \f$ ,                          
        - \f$ I2 = [ ( trace(C)^2-trace(C^2) )/2 ]  \f$ ,   
        - \f$ J = det(F) \f$ ,                             
        - \f$ C=F^TF \f$ is the right Cauchy deformation tensor
        */
        
        typedef Mapping_test<_Mapping> Inherited;
        typedef typename Inherited::In In;
        typedef typename Inherited::Real Real;
        typedef typename Inherited::InVecCoord InVecCoord;
        typedef typename Inherited::OutVecCoord OutVecCoord;
        typedef typename In::Frame InFrame;

        using Inherited::runTest;
        bool runTest()
        {
            this->deltaRange = std::make_pair( 100, 10000 );
            this->errorMax = this->deltaRange.second;
            this->errorFactorDJ = 10;

            defaulttype::Mat<In::material_dimensions,In::material_dimensions,Real> strain; 

            // create a deformation gradient
            for( unsigned int i=0 ; i<In::material_dimensions ; ++i )
            {   for( unsigned int j=i ; j<In::material_dimensions ; ++j )
                {
                    strain[i][j] = (i+1)*2+j*0.3; 
                }
            }
//                defaulttype::Mat<In::material_dimensions,In::material_dimensions,Real> defo( strain );
                defaulttype::Mat<In::material_dimensions,In::material_dimensions,Real> C;
                C = (strain.transposed())*strain;
                
                // Invariants

                Real I1 = trace(C);
                Real I2 = (I1*I1-trace(C*C))/2;
                Real J = determinant(strain);

                // expected mapped values
                OutVecCoord expectedChildCoords(1);
                expectedChildCoords[0].getVec()[0] = I1;
                expectedChildCoords[0].getVec()[1] = I2;
                expectedChildCoords[0].getVec()[2] = J;


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

//                cerr<<"InvariantMappingTest::runTest, f="<< f << endl;
//                cerr<<"InvaraintMappingTest::runTest, expected="<< expectedChildCoords << endl;

                return Inherited::runTest(xin,xout,xin,expectedChildCoords);

        }

    };

    // Define the list of types to instanciate.
    typedef Types<
        InvariantMapping<defaulttype::F331Types,defaulttype::I331Types>
    > InvariantDataTypes; // the types to instanciate.

    // Test suite for all the instanciations
    TYPED_TEST_CASE(InvariantMappingTest, InvariantDataTypes);
    // first test case
    TYPED_TEST( InvariantMappingTest , test_auto )
    {
        ASSERT_TRUE( this->runTest() );
    }

} // namespace sofa
