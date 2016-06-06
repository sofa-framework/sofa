#ifndef WinchMapping_H
#define WinchMapping_H

#include <Compliant/config.h>
#include <Compliant/mapping/AssembledMultiMapping.h>

namespace sofa
{
	
namespace component
{

namespace mapping
{

/**
 Multi-maps a distance and a rotation vectors, to an error for associating a rotation to an elongation. 

 (d, r) -> a*r - d
 (Where a is a real factor that should depends on )

 This is used to obtain relative dofs
 on which a stiffness/compliance may be applied
*/

    template <class TIn, class TOut >
    class SOFA_Compliant_API WinchMultiMapping : public AssembledMultiMapping<TIn, TOut>
    {
        typedef WinchMultiMapping self;

    public:
        SOFA_CLASS(SOFA_TEMPLATE2(WinchMultiMapping,TIn,TOut), SOFA_TEMPLATE2(core::MultiMapping,TIn,TOut));

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
            this->getToModels()[0]->resize( this->getFromModels()[0]->getSize() );
            AssembledMultiMapping<TIn, TOut>::init();
        }

        virtual void reinit()
        {
            this->getToModels()[0]->resize( this->getFromModels()[0]->getSize() );
            AssembledMultiMapping<TIn, TOut>::reinit();
        }

        virtual void apply(typename self::out_pos_type& out,
                           const helper::vector<typename self::in_pos_type>& in)  {
            Real f = factor.getValue();

            for( unsigned j = 0, m = in[0].size(); j < m; ++j) {
                out[j] = f * TIn::getCPos( in[1] [j] ) - TIn::getCPos( in[0] [j] );
            }

        }

        Data< Real > factor;

    protected:

        WinchMultiMapping()
            : factor( initData(&factor, Real(1.0),"factor", "factor representing the ratio between a rotation and an elongation") ) {

        }

        void assemble(const helper::vector<typename self::in_pos_type>& in ) {

            Real f = factor.getValue();

            for(unsigned i = 0, n = in.size(); i < n; ++i) {

                typename Inherit::jacobian_type::CompressedMatrix& J = this->jacobian(i).compressedMatrix;

                J.resize( Nout * in[i].size(), Nin * in[i].size());
                J.setZero();

                Real sign = (i == 0) ? -1 : f;

                for(unsigned k = 0, n = in[i].size(); k < n; ++k) {
                    write_block(J, k, k, sign);
                }

                J.finalize();
            }
        }



        // write sign * identity in jacobian(obj)
        void write_block(typename Inherit::jacobian_type::CompressedMatrix& J,
                         unsigned row, unsigned col,
                         SReal sign) {
            assert( Nout == Nin );

            // for each coordinate in matrix block
            for( unsigned i = 0, n = Nout; i < n; ++i ) {
                unsigned r = row * Nout + i;
                unsigned c = col * Nin + i;

                J.startVec(r);
                J.insertBack(r, c) = sign;
            }
        }


        virtual void updateForceMask()
        {

        }

    };


}
}
}


#endif
