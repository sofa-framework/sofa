#ifndef SOFA_COMPLIANT_GEARMAPPING_H
#define SOFA_COMPLIANT_GEARMAPPING_H

#include <Compliant/config.h>

#include "AssembledMultiMapping.h"

namespace sofa
{
	
namespace component
{

namespace mapping
{


/**
 Multi-maps two kinematic dofs to their scaled difference:

 (v1, v2) -> ( v2[x] - v1[y] ) * ratio

 This is used to model any kind of gear joints (gear/belt/chain/rack...)

 @warning: it is a pure velocity mapping, and no positions are computed

 @author Matthieu Nesme

*/

    template <class TIn, class TOut >
    class SOFA_Compliant_API GearMultiMapping : public AssembledMultiMapping<TIn, TOut>
    {

    public:

        SOFA_CLASS(SOFA_TEMPLATE2(GearMultiMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMultiMapping,TIn,TOut));

        typedef AssembledMultiMapping<TIn, TOut> Inherit;
        typedef TIn In;
        typedef TOut Out;
        typedef typename Out::VecCoord OutVecCoord;
        typedef typename Out::VecDeriv OutVecDeriv;
        typedef typename Out::Coord OutCoord;
        typedef typename Out::Deriv OutDeriv;
        typedef typename Out::MatrixDeriv OutMatrixDeriv;
        typedef typename Out::Real Real;
        typedef typename In::Deriv InDeriv;
        typedef typename In::MatrixDeriv InMatrixDeriv;
        typedef typename In::Coord InCoord;
        typedef typename In::VecCoord InVecCoord;
        typedef typename In::VecDeriv InVecDeriv;
        typedef linearsolver::EigenSparseMatrix<TIn,TOut>    SparseMatrixEigen;

        typedef typename helper::vector <const InVecCoord*> vecConstInVecCoord;
        typedef typename helper::vector<OutVecCoord*> vecOutVecCoord;

        enum {Nin = In::deriv_total_size, Nout = Out::deriv_total_size };

        virtual void init()
        {
            this->getToModels()[0]->resize( d_pairs.getValue().size() );
            Inherit::init();
        }



        virtual void apply(typename Inherit::out_pos_type& /*out*/,
                           const helper::vector<typename Inherit::in_pos_type>& in)  {
            // macro_trace;
            assert( in.size() == 2 );
            assert( this->Nout == 1 );

            (void) in;
        }

        typedef defaulttype::Vec<2, unsigned> index_type;
        typedef defaulttype::Vec<2, index_type> index_pair;
        typedef helper::vector< index_pair > pairs_type;

        Data< pairs_type > d_pairs; ///< index pairs for computing deltas, 4 values per pair (dofindex0,kinematicdofindex0,dofindex1,kinematicdofindex1) 
        Data< Real > d_ratio; ///< a different ratio for each pair

    protected:

        GearMultiMapping()
            : d_pairs( initData(&d_pairs, "pairs", "index pairs for computing deltas, 4 values per pair (dofindex0,kinematicdofindex0,dofindex1,kinematicdofindex1) ") )
            , d_ratio( initData(&d_ratio, (Real)1, "ratio", "gear link ratio (can be negative)") )
        {}

        void assemble(const helper::vector<typename Inherit::in_pos_type>& in ) {

            const Real& ratio = d_ratio.getValue();

            const pairs_type& p = d_pairs.getValue();
            assert( !p.empty() );

            for(unsigned i = 0, n = in.size(); i < n; ++i) {

                typename Inherit::jacobian_type::CompressedMatrix& J = this->jacobian(i).compressedMatrix;

                J.resize( Nout * p.size(), Nin * in[i].size());
                J.reserve( Nout * p.size() );

                const Real sign = (i == 0) ? -ratio : 1;

                for(unsigned k = 0, n = p.size(); k < n; ++k) {

                    const index_type& index = p[k][i];

                    unsigned c = index[0] * Nin + index[1];

                    J.startVec(k);
                    J.insertBack(k, c) = sign;

                }

                J.finalize();
            }
        }

        virtual void updateForceMask()
        {
            const pairs_type& p = d_pairs.getValue();

            for( size_t i = 0, iend = p.size(); i < iend; ++i )
            {
                if( this->maskTo[0]->getEntry(i) )
                {
                    const index_pair& indices = p[i];
                    this->maskFrom[0]->insertEntry(indices[0][0]);
                    this->maskFrom[1]->insertEntry(indices[1][0]);
                }
            }
        }

    };


}
}
}


#endif
