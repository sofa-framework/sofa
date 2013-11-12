#ifndef DIFFERENCEMAPPING_H
#define DIFFERENCEMAPPING_H

#include "../initCompliant.h"

#include "AssembledMapping.h"
#include "AssembledMultiMapping.h"

namespace sofa
{
	
namespace component
{

namespace mapping
{

template <class TIn, class TOut >
class SOFA_Compliant_API DifferenceMapping : public AssembledMapping<TIn, TOut>
{
  public:
    SOFA_CLASS(SOFA_TEMPLATE2(DifferenceMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMapping,TIn,TOut));
	
	typedef DifferenceMapping self;
	
	typedef defaulttype::Vec<2, unsigned> index_pair;
	typedef vector< index_pair > pairs_type;
	
	Data< pairs_type > pairs;
	
	DifferenceMapping() 
		: pairs( initData(&pairs, "pairs", "index pairs for computing deltas") ) {
		
	}

	enum {Nin = TIn::deriv_total_size, Nout = TOut::deriv_total_size };

	virtual void apply(typename self::out_pos_type& out, 
	                   const typename self::in_pos_type& in )  {
		assert( this->Nout == this->Nin );

		pairs_type& p = *pairs.beginEdit();
		assert( !p.empty() );

		for( unsigned j = 0, m = p.size(); j < m; ++j) {
			if( p[j][1] > p[j][0] ) std::swap( p[j][1], p[j][0] );

			out[j] = in[p[j][1]] - in[p[j][0]];
		}

		pairs.endEdit();

	}

	virtual void assemble( const typename self::in_pos_type& in ) {
		// jacobian matrix assembly
		pairs_type& p = *pairs.beginEdit();
		assert( !p.empty() );

		typename self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;

		J.resize( Nout * p.size(), Nin * in.size());
		J.setZero();

		for(unsigned k = 0, n = p.size(); k < n; ++k) {
			
			for(unsigned i = 0; i < Nout; ++i) {
				unsigned row = k * Nout + i;
				
				// needs to be sorted !
				if( p[k][1] > p[k][0] ) std::swap( p[k][1], p[k][0] );
				
				if(p[k][1] == p[k][0]) continue;
				
				J.startVec( row );

				for( unsigned u = 0; u < 2; ++u ) {
					SReal sign = (u == 0) ? -1 : 1;
					
					for( unsigned j = 0; j < Nin; ++j) {
						unsigned col = p[k][u] * Nin + j;
						J.insertBack(row, col) = sign;
					}
				}
			}
		}
		J.finalize();
		
		pairs.endEdit();
	}

	
};



//////////////////////




/**
 Multi-maps two vec dofs to their difference:

 (p1, p2) -> p2 - p1

 This is used in compliant contacts to obtain relative
 distance dofs, on which a contact compliance may be applied
 (= seamless penalty/constraint force transition)

 @author: maxime.tournier@inria.fr
*/

    template <class TIn, class TOut >
    class SOFA_Compliant_API DifferenceMultiMapping : public AssembledMultiMapping<TIn, TOut>
    {
        typedef DifferenceMultiMapping self;

    public:
        SOFA_CLASS(SOFA_TEMPLATE2(DifferenceMultiMapping,TIn,TOut), SOFA_TEMPLATE2(core::MultiMapping,TIn,TOut));

        typedef core::Mapping<TIn, TOut> Inherit;
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



        virtual void apply(typename self::out_pos_type& out,
                           const vector<typename self::in_pos_type>& in)  {
            // macro_trace;
            assert( in.size() == 2 );

            assert( this->Nout == this->Nin );

            const pairs_type& p = pairs.getValue();
            assert( !p.empty() );

            for( unsigned j = 0, m = p.size(); j < m; ++j) {
                out[j] = in[1] [p[j][1]] - in[0] [p[j][0]];
            }

        }


        typedef defaulttype::Vec<2, unsigned> index_pair;
        typedef vector< index_pair > pairs_type;

        Data< pairs_type > pairs;

    protected:

        DifferenceMultiMapping()
            : pairs( initData(&pairs, "pairs", "index pairs for computing deltas") ) {

        }

        void assemble(const vector<typename self::in_pos_type>& in ) {

            for(unsigned i = 0, n = in.size(); i < n; ++i) {
                // jacobian matrix assembly
                const pairs_type& p = pairs.getValue();
                assert( !p.empty() );

                this->jacobian(i).compressedMatrix.resize( Nout * p.size(), Nin * in[i].size());
                this->jacobian(i).compressedMatrix.setZero();

                Real sign = (i == 0) ? -1 : 1;

                for(unsigned k = 0, n = p.size(); k < n; ++k) {
                    write_block(i, k, p[k][i], sign);
                }

                this->jacobian(i).compressedMatrix.finalize();
            }
        }



        // write sign * identity in jacobian(obj)
        void write_block(unsigned int obj,
                         unsigned row, unsigned col,
                         SReal sign) {
            assert( Nout == Nin );

            // for each coordinate in matrix block
            for( unsigned i = 0, n = Nout; i < n; ++i ) {
                unsigned r = row * Nout + i;
                unsigned c = col * Nin + i;

                this->jacobian( obj ).compressedMatrix.startVec(r);
                this->jacobian( obj ).compressedMatrix.insertBack(r, c) = sign;
            }
        }

    };


}
}
}


#endif
