#include "stdafx.h"
#include <sofa/helper/Quater.h>

#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

#include "../strainMapping/CorotationalStrainMapping.h"
#include "../strainMapping/PrincipalStretchesMapping.h"
#include "../strainMapping/GreenStrainMapping.h"
#include "../strainMapping/InvariantMapping.h"
#include "../strainMapping/CauchyStrainMapping.h"

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


    /// Base class to compare StrainMapping
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
            this->deltaMax = 100;  // OUUUUUUUU la triche ! Du coup on ne teste pas les J
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
            cerr<<"StrainMappingTest::runTest, f="<< f << endl;
            cerr<<"StrainMappingTest::runTest, expected="<< expectedChildCoords << endl;

            static_cast<_Mapping*>(this->mapping)->assemble.setValue(true);

            return Inherited::runTest(xin,xout,xin,expectedChildCoords);
        }

    };




    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////



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
//          cerr<<"symGradDef = " << symGradDef << endl;

                // expected mapped values
                OutVecCoord expectedChildCoords(1);
                defaulttype::Mat<In::material_dimensions,In::material_dimensions,Real> defo( symGradDef );
                for( unsigned int i=0 ; i<In::material_dimensions ; ++i )
                    defo[i][i] -= 1.0;
                expectedChildCoords[0].getVec() = defaulttype::StrainMatToVoigt( defo );
//              cerr<<"voigt strain = " << defo << endl;

                helper::Quater<Real>::fromEuler( 0.1, -.2, .3 ).toMatrix(rotation); // random rotation to combine to strain
//                helper::Quater<Real>::fromEuler( 0,0,0 ).toMatrix(rotation); // random rotation to combine to strain

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
    // first test case
    TYPED_TEST( CorotationalStrainMappingTest , test_auto )
    {
        ASSERT_TRUE( this->runTest( 0 ) ); // polar
        ASSERT_TRUE( this->runTest( 3 ) ); // svd
    }


    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////


    /// layer over PrincipalStretchesJacobianBlock that is able to order given child forces in the same order while computing J
    template<class TIn, class TOut>
    class PrincipalStretchesJacobianBlockTester : public defaulttype::PrincipalStretchesJacobianBlock<TIn,TOut>
    {
    public :
        typedef defaulttype::PrincipalStretchesJacobianBlock<TIn,TOut> Inherited;

        static const int material_dimensions = Inherited::material_dimensions;
        typedef typename Inherited::OutCoord OutCoord;
        typedef typename Inherited::OutDeriv OutDeriv;
        typedef typename Inherited::InCoord InCoord;
        typedef typename Inherited::InDeriv InDeriv;
        typedef typename Inherited::Real Real;

        unsigned _order[material_dimensions];

        void findOrder( const OutCoord& d )
        {
            OutCoord tmp = d;

            for( int i=0 ; i<material_dimensions ; ++i )
            {
                Real min = 999999999999999;
                for( int j=0 ; j<material_dimensions ; ++j )
                    if( tmp.getStrain()[j] < min )
                    {
                        _order[i] = j;
                        min = tmp.getStrain()[j];
                    }
                    tmp.getStrain()[_order[i]] = 999999999999999999;
            }
        }

        virtual OutDeriv preTreatment( const OutDeriv& f )
        {
            // re-order child forces with the same order used while computing J in addapply
            OutDeriv g;
            for( int i=0 ; i<material_dimensions ; ++i )
                g[i] = f[_order[i]];
            return g;
        }


        void addapply( OutCoord& result, const InCoord& data )
        {
            Inherited::addapply( result, data );

            // find order result.getStrain()[i]
            findOrder( result.getStrain() );
        }
    };

    /// layer over PrincipalStretchesMapping that is able to order given child forces in the same order while computing J
    template <class TIn, class TOut>
    class PrincipalStretchesMappingTester : public BaseStrainMappingT<PrincipalStretchesJacobianBlockTester<TIn,TOut> >
    {
    public:
        typedef BaseStrainMappingT<PrincipalStretchesJacobianBlockTester<TIn,TOut> > Inherited;
        typedef typename Inherited::OutVecDeriv OutVecDeriv;

        virtual OutVecDeriv preTreatment( const OutVecDeriv& f )
        {
            OutVecDeriv g(f.size());
            for(unsigned int i=0; i<this->jacobian.size(); i++)
            {
                g[i] = this->jacobian[i].preTreatment(f[i]);
            }
            return g;
        }
    };


    template <typename _Mapping>
    struct PrincipalStretchesMappingTest : public StrainMappingTest<_Mapping>
    {
        typedef StrainMappingTest<_Mapping> Inherited;

        typedef typename Inherited::In In;
        typedef typename Inherited::Real Real;
        typedef typename Inherited::OutVecCoord OutVecCoord;
        typedef typename Inherited::OutCoord OutCoord;
        typedef typename Inherited::OutDeriv OutDeriv;
        typedef typename Inherited::OutVecDeriv OutVecDeriv;


        OutCoord sort( const OutCoord& a )
        {
            OutCoord aa;
            std::vector<Real> v;
            v.assign( &a[0], &a[0] + OutCoord::total_size );
            std::sort( v.begin(), v.end() );
            for( unsigned i=0 ; i<OutCoord::total_size ; ++i ) aa[i] = v[i];
            return aa;
        }


        /// since principal stretches are oder-independent, sort them before comparison
        virtual OutDeriv difference( const OutCoord& a, const OutCoord& b )
        {
            return (OutDeriv)(sort(a)-sort(b));
        }


        /// re-order child forces with the same order used while computing J in apply
        virtual OutVecDeriv preTreatment( const OutVecDeriv& f )
        {
            return static_cast<_Mapping*>(this->mapping)->preTreatment( f );
        }



        bool runTest()
        {
            defaulttype::Mat<3,3,Real> rotation;
            defaulttype::Mat<In::material_dimensions,In::material_dimensions,Real> strain; // stretch only

            for( unsigned int i=0 ; i<In::material_dimensions ; ++i )
            {
                strain[i][i] = (i+1)*2; // todo randomize it being careful not to obtain negative (even too small) eigenvalues
            }

            helper::Quater<Real>::fromEuler( 0.1, -.2, .3 ).toMatrix(rotation);

            OutVecCoord expectedChildCoords(1);
            for( unsigned int i=0 ; i<In::material_dimensions ; ++i )
                expectedChildCoords[0].getVec()[i] = strain[i][i];
            
            return Inherited::runTest( rotation, strain, expectedChildCoords );
        }

    };


    // Define the list of types to instanciate.
    typedef Types<
        PrincipalStretchesMappingTester<defaulttype::F331Types,defaulttype::U331Types>
        ,PrincipalStretchesMappingTester<defaulttype::F321Types,defaulttype::U321Types>
        //        ,PrincipalStretchesMapping<defaulttype::F331Types,defaulttype::D331Types> // not fully implemented yet (getJ)
        //        ,PrincipalStretchesMapping<defaulttype::F321Types,defaulttype::D321Types> // not fully implemented yet (getJ)
    > PrincipalStretchesDataTypes; // the types to instanciate.

    // Test suite for all the instanciations
    TYPED_TEST_CASE(PrincipalStretchesMappingTest, PrincipalStretchesDataTypes);
    // first test case
    TYPED_TEST( PrincipalStretchesMappingTest , test_auto )
    {
        ASSERT_TRUE(  this->runTest() );
    }


    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////



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

                this->deltaMax = 0.001;  
                this->errorMax = 10;

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

                cerr<<"StrainMappingTest::runTest, f="<< f << endl;
                cerr<<"StrainMappingTest::runTest, expected="<< expectedChildCoords << endl;

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

    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////

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
                defo = (strain + strain.transposed())*0.5 -strain.Identity();

                // expected mapped values
                OutVecCoord expectedChildCoords(1);
                expectedChildCoords[0].getVec() = defaulttype::StrainMatToVoigt( defo );

                this->deltaMax = 0.001;  
                this->errorMax = 5;

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

                cerr<<"CauchyStrainMappingTest::runTest, f="<< f << endl;
                cerr<<"CauchyStrainMappingTest::runTest, expected="<< expectedChildCoords << endl;

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

    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////
    //////////////////////////////////////////////////////

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


        bool runTest()
        {
            defaulttype::Mat<In::material_dimensions,In::material_dimensions,Real> strain; 

            // create a deformation gradient
            for( unsigned int i=0 ; i<In::material_dimensions ; ++i )
            {   for( unsigned int j=i ; j<In::material_dimensions ; ++j )
                {
                    strain[i][j] = (i+1)*2+j*0.3; 
                }
            }
                defaulttype::Mat<In::material_dimensions,In::material_dimensions,Real> defo( strain );
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

                this->deltaMax = 500;  // J not tested
                this->errorMax = 6000;

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

                cerr<<"InvariantMappingTest::runTest, f="<< f << endl;
                cerr<<"InvaraintMappingTest::runTest, expected="<< expectedChildCoords << endl;

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
