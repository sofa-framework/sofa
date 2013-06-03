
// I do not know why extern template needs to be desactivated, but for now it does the job
#define SOFA_NO_EXTERN_TEMPLATE
#ifdef SOFA_EXTERN_TEMPLATE
#undef SOFA_EXTERN_TEMPLATE
#endif

#include <sofa/helper/Quater.h>

#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

#include "../strainMapping/CorotationalStrainMapping.h"
#include "../strainMapping/PrincipalStretchesMapping.h"

#include <Mapping_test.h>


namespace sofa {

    using std::cout;
    using std::cerr;
    using std::endl;
    using namespace core;
    using namespace component;
    using defaulttype::Vec;
    using defaulttype::Mat;
    using testing::Types;

    using namespace component::mapping;


    template <typename _Mapping>
    struct StrainMappingTest : public Mapping_test<_Mapping>
    {
        typedef Mapping_test<_Mapping> Inherited;

        typedef typename Inherited::In In;
        typedef typename Inherited::WriteInVecCoord WriteInVecCoord;
        typedef typename Inherited::Real Real;
        typedef typename Inherited::OutVecCoord OutVecCoord;
        typedef typename Inherited::InVecCoord InVecCoord;
        typedef typename In::Frame InFrame;

        bool runTest( defaulttype::Mat<3,3,Real>& rotation, defaulttype::Mat<In::material_dimensions,In::material_dimensions,Real>& strain, const OutVecCoord& expectedChildCoords)
        {
            this->deltaMax = 100;
            this->errorMax = 1000;

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

            // rotation
            f = rotation * f;


            return Inherited::runTest(xin,xout,xin,expectedChildCoords);
        }

    };




//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////



  template <typename _Mapping>
  struct CorotationalStrainMappingTest : public StrainMappingTest<_Mapping>
  {
      typedef StrainMappingTest<_Mapping> Inherited;

      typedef typename Inherited::In In;
      typedef typename Inherited::Real Real;
      typedef typename Inherited::OutVecCoord OutVecCoord;



      bool runTest( unsigned method )
      {
          static_cast<_Mapping*>(this->mapping)->f_geometricStiffness.setValue(1);
          static_cast<_Mapping*>(this->mapping)->f_method.beginEdit()->setSelectedItem( method );

          defaulttype::Mat<3,3,Real> rotation;
          defaulttype::Mat<In::material_dimensions,In::material_dimensions,Real> strain; // stretch + shear

          for( unsigned int i=0 ; i<In::material_dimensions ; ++i )
          for( unsigned int j=i ; j<In::material_dimensions ; ++j )
          {
              strain[i][j] = (i+1)*2+j*0.3; // todo randomize it being careful not to create a rotation
              if( i!=j ) strain[j][i] = strain[i][j];
          }

          helper::Quater<Real>::fromEuler( 0.1, -.2, .3 ).toMatrix(rotation);


          // expected mapped values
          OutVecCoord expectedChildCoords(1);
          defaulttype::Mat<In::material_dimensions,In::material_dimensions,Real> defo( strain );
          for( unsigned int i=0 ; i<In::material_dimensions ; ++i )
              defo[i][i] -= 1.0;
          expectedChildCoords[0].getVec() = defaulttype::StrainMatToVoigt( defo );

          return Inherited::runTest( rotation, strain, expectedChildCoords );

      }

  };


  // Define the list of types to instanciate. We do not necessarily need to test all combinations.
  typedef Types<
  CorotationalStrainMapping<defaulttype::F331Types,defaulttype::E331Types>,
  CorotationalStrainMapping<defaulttype::F321Types,defaulttype::E321Types>
  > CorotationalDataTypes; // the types to instanciate.

  // Test suite for all the instanciations
  TYPED_TEST_CASE(CorotationalStrainMappingTest, CorotationalDataTypes);
  // first test case
  TYPED_TEST( CorotationalStrainMappingTest , test_auto )
  {
      ASSERT_TRUE( this->runTest( 0 ) ); // polar
      ASSERT_TRUE( this->runTest( 3 ) ); // svd
  }


  //////////////////////////////////////////////////////
  //////////////////////////////////////////////////////
  //////////////////////////////////////////////////////



    template <typename _Mapping>
    struct PrincipalStretchesMappingTest : public StrainMappingTest<_Mapping>
    {
        typedef StrainMappingTest<_Mapping> Inherited;

        typedef typename Inherited::In In;
        typedef typename Inherited::Real Real;
        typedef typename Inherited::OutVecCoord OutVecCoord;

        bool runTest()
        {
            static_cast<_Mapping*>(this->mapping)->asStrain.setValue(1);

            defaulttype::Mat<3,3,Real> rotation;
            defaulttype::Mat<In::material_dimensions,In::material_dimensions,Real> strain; // stretch + shear

            for( unsigned int i=0 ; i<In::material_dimensions ; ++i )
            {
                strain[i][i] = (i+1)*2; // todo randomize it
            }

            helper::Quater<Real>::fromEuler( 0.1, -.2, .3 ).toMatrix(rotation);


            // TODO: principal stretches are not sorted... hard to compare
            OutVecCoord expectedChildCoords(1);
            for( unsigned int i=0 ; i<In::material_dimensions ; ++i )
                expectedChildCoords[0].getVec()[i] = strain[i][i] - 1;

            return Inherited::runTest( rotation, strain, expectedChildCoords );
        }

    };


    // Define the list of types to instanciate. We do not necessarily need to test all combinations.
//    typedef Types<
//    component::mapping::PrincipalStretchesMapping<defaulttype::F331Types,defaulttype::U331Types>,
//    component::mapping::PrincipalStretchesMapping<defaulttype::F321Types,defaulttype::U321Types>
//    > PrincipalStretchesDataTypes; // the types to instanciate.

//    // Test suite for all the instanciations
//    TYPED_TEST_CASE(PrincipalStretchesMappingTest, PrincipalStretchesDataTypes);
//    // first test case
//    TYPED_TEST( PrincipalStretchesMappingTest , test_auto )
//    {
//        ASSERT_TRUE(  this->runTest() );
//    }







} // namespace sofa
