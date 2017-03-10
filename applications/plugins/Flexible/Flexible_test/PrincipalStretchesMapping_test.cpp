#include "../strainMapping/PrincipalStretchesMapping.h"

#include "StrainMapping_test.h"
#include <sofa/helper/logging/Messaging.h>

namespace sofa {


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
                Real min = std::numeric_limits<Real>::max();
                for( int j=0 ; j<material_dimensions ; ++j )
                {
                    if( tmp.getStrain()[j] == min && min != std::numeric_limits<Real>::max() )
                    {
                        msg_warning("PrincipalStretchesJacobianBlockTester") << "Several strain components are identical, the test cannot find the comparison order, try with another data set.";
                        std::cerr<<tmp.getStrain()<<std::endl;
                    }

                    if( tmp.getStrain()[j] < min )
                    {
                        _order[i] = j;
                        min = tmp.getStrain()[j];
                    }
                    tmp.getStrain()[j] = std::numeric_limits<Real>::max();
                }
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

        template<class V>
        V sort( const V& a )
        {
            V aa( a );
            std::sort( &aa[0], &aa[0]+V::total_size );
            return aa;
        }


        /// since principal stretches are oder-independent, sort them before comparison
        virtual OutDeriv difference( const OutCoord& a, const OutCoord& b )
        {
            return (OutDeriv)(sort(a)-sort(b));
        }

        /// since principal stretches are oder-independent, sort them before comparison
        virtual OutVecDeriv difference( const OutVecDeriv& a, const OutVecDeriv& b )
        {
            if( a.size()!=b.size() ){
                ADD_FAILURE() << "OutVecDeriv have different sizes";
                return OutVecDeriv();
            }

            OutVecDeriv c(a.size());
            for (size_t i=0; i<(size_t)a.size() ; ++i)
            {
                c[i] = sort(a[i])-sort(b[i]);
            }
            return c;
        }


        /// re-order child forces with the same order used while computing J in apply
        virtual OutVecDeriv preTreatment( const OutVecDeriv& f )
        {
            return static_cast<_Mapping*>(this->mapping)->preTreatment( f );
        }



        bool runTest()
        {
            this->deltaRange = std::make_pair( 100, 10000 );
            this->errorMax = this->deltaRange.second;
            this->errorFactorDJ = 10;

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


} // namespace sofa
