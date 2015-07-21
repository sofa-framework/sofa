#include "../strainMapping/CorotationalStrainMapping.h"

#include "StrainMapping_test.h"


namespace sofa {


    template <typename _Mapping>
    struct CorotationalStrainMappingTest : public StrainMappingTest<_Mapping>
    {
         /* Test the corotational strain mapping:
         * Create a symmetric deformation gradient encoding a pure deformation D.
         * Then the strain E is mapped from the deformation gradient as:
         - \f$ E = D - I  \f$
        */
        typedef StrainMappingTest<_Mapping> Inherited;

        typedef typename Inherited::In In;
        typedef typename Inherited::Real Real;
        typedef typename Inherited::OutVecCoord OutVecCoord;



        bool runTest( unsigned method )
        {
            this->deltaRange = std::make_pair( 100, 10000 );
            this->errorMax = this->deltaRange.second*2;
            this->errorFactorDJ = 10;

            static_cast<_Mapping*>(this->mapping)->f_geometricStiffness.setValue(1);
            static_cast<_Mapping*>(this->mapping)->f_method.beginEdit()->setSelectedItem( method );

            defaulttype::Mat<3,3,Real> rotation;
            defaulttype::Mat<In::material_dimensions,In::material_dimensions,Real> symGradDef; // local frame with only stretch and shear and no rotation

            // create a symmetric deformation gradient, encoding a pure deformation.
            for( unsigned int i=0 ; i<In::material_dimensions ; ++i )
                for( unsigned int j=i ; j<In::material_dimensions ; ++j )
                {
                    symGradDef[i][j] = (i+1)*2+j*0.3; // todo randomize it being careful not to create a rotation
                    if( i!=j ) symGradDef[j][i] = symGradDef[i][j];
                }
//          /*cerr<<*/"symGradDef = " << symGradDef << endl;

                // expected mapped values
                OutVecCoord expectedChildCoords(1);
                defaulttype::Mat<In::material_dimensions,In::material_dimensions,Real> defo( symGradDef );
                for( unsigned int i=0 ; i<In::material_dimensions ; ++i )
                    defo[i][i] -= 1.0;
                expectedChildCoords[0].getVec() = defaulttype::StrainMatToVoigt( defo );
//              cerr<<"voigt strain = " << defo << endl;

                helper::Quater<Real>::fromEuler( 0.1, -.2, .3 ).toMatrix(rotation); // random rotation to combine to strain
//                helper::Quater<Real>::fromEuler( 0,0,0 ).toMatrix(rotation); // debug with no rotation

                return Inherited::runTest( rotation, symGradDef, expectedChildCoords );

        }

    };


    // Define the list of types to instanciate.
    typedef Types<
        CorotationalStrainMapping<defaulttype::F331Types,defaulttype::E331Types>,
        CorotationalStrainMapping<defaulttype::F321Types,defaulttype::E321Types>,
        CorotationalStrainMapping<defaulttype::F311Types,defaulttype::E311Types>
    > CorotationalDataTypes; // the types to instanciate.

    // Test suite for all the instanciations
    TYPED_TEST_CASE(CorotationalStrainMappingTest, CorotationalDataTypes);
    // test cases
    TYPED_TEST( CorotationalStrainMappingTest , polar )
    {
        ASSERT_TRUE( this->runTest( 0 ) ); // polar
    }
    TYPED_TEST( CorotationalStrainMappingTest , svd )
    {
        ASSERT_TRUE( this->runTest( 3 ) ); // svd
    }



} // namespace sofa
