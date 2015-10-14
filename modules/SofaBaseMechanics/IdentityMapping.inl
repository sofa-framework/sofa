/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MAPPING_IDENTITYMAPPING_INL
#define SOFA_COMPONENT_MAPPING_IDENTITYMAPPING_INL

#include <SofaBaseMechanics/IdentityMapping.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>


namespace sofa
{

namespace component
{

namespace mapping
{


template<class TIn, class TOut>
void IdentityMapping<TIn, TOut>::init()
{
    this->toModel->resize( this->fromModel->getSize() );

    Inherit::init();
}

template <class TIn, class TOut>
void IdentityMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/, Data<VecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteOnlyAccessor< Data<VecCoord> > out = dOut;
    helper::ReadAccessor< Data<InVecCoord> > in = dIn;

//    out.resize(in.size());

    for(unsigned int i=0; i<out.size(); i++)
    {
        helper::eq(out[i], in[i]);
    }
}

template <class TIn, class TOut>
void IdentityMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/, Data<VecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    helper::WriteOnlyAccessor< Data<VecDeriv> > out = dOut;
    helper::ReadAccessor< Data<InVecDeriv> > in = dIn;

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( !this->maskTo->isActivated() || this->maskTo->getEntry(i) )
            helper::eq(out[i], in[i]);
    }
}

template<class TIn, class TOut>
void IdentityMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/, Data<InVecDeriv>& dOut, const Data<VecDeriv>& dIn)
{
    helper::WriteAccessor< Data<InVecDeriv> > out = dOut;
    helper::ReadAccessor< Data<VecDeriv> > in = dIn;

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( this->maskTo->getEntry(i) )
            helper::peq(out[i], in[i]);
    }
}

template <class TIn, class TOut>
void IdentityMapping<TIn, TOut>::applyJT(const core::ConstraintParams * /*cparams*/, Data<InMatrixDeriv>& dOut, const Data<MatrixDeriv>& dIn)
{
    InMatrixDeriv& out = *dOut.beginEdit();
    const MatrixDeriv& in = dIn.getValue();

    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        // Creates a constraints if the input constraint is not empty.
        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            while (colIt != colItEnd)
            {
                InDeriv data;
                helper::eq(data, colIt.val());

                o.addCol(colIt.index(), data);

                ++colIt;
            }
        }
    }

    dOut.endEdit();
}

template <class TIn, class TOut>
void IdentityMapping<TIn, TOut>::handleTopologyChange()
{
    if ( this->toModel && this->fromModel && this->toModel->getSize() != this->fromModel->getSize()) this->init();
}

template <class TIn, class TOut>
const sofa::defaulttype::BaseMatrix* IdentityMapping<TIn, TOut>::getJ()
{
    const unsigned int outStateSize = this->toModel->getSize();
    const unsigned int  inStateSize = this->fromModel->getSize();
    assert(outStateSize == inStateSize);

    if (matrixJ.get() == 0 || updateJ)
    {
        updateJ = false;
        if (matrixJ.get() == 0 || (unsigned int)matrixJ->rowBSize() != outStateSize || (unsigned int)matrixJ->colBSize() != inStateSize)
        {
            matrixJ.reset(new MatrixType(outStateSize * NOut, inStateSize * NIn));
        }
        else
        {
            matrixJ->clear();
        }

        for( size_t i=0 ; i<this->maskTo->size() ; ++i)
        {
            if( this->maskTo->getEntry(i) )
            {
                MBloc& block = *matrixJ->wbloc(i, i, true);
                IdentityMappingMatrixHelper<NOut, NIn, Real>::setMatrix(block);
            }
        }
    }
    return matrixJ.get();
}

template<int N, int M, class Real>
struct IdentityMappingMatrixHelper
{
    template <class Matrix>
    static void setMatrix(Matrix& mat)
    {
        for(int r = 0; r < N; ++r)
        {
            for(int c = 0; c < M; ++c)
            {
                mat[r][c] = (Real) 0;
            }
            if( r<M ) mat[r][r] = (Real) 1.0;
        }
    }
};


template <class TIn, class TOut>
const typename IdentityMapping<TIn, TOut>::js_type* IdentityMapping<TIn, TOut>::getJs()
{
    if( !eigen.compressedMatrix.nonZeros() || updateJ ) {
		updateJ = false;

		assert( this->fromModel->getSize() == this->toModel->getSize());

		const unsigned n = this->fromModel->getSize();

		// each block (input i, output j) has only its top-left
		// principal submatrix filled with identity
		
		const unsigned rows = n * NOut;
        const unsigned cols = n * NIn;

        static const unsigned N = std::min<unsigned>(NIn, NOut);


        eigen.compressedMatrix.resize( rows, cols );
        eigen.compressedMatrix.setZero();
        eigen.compressedMatrix.reserve( rows );

        for( size_t i=0 ; i<this->maskTo->size() ; ++i)
        {
            if( !this->maskTo->getEntry(i) )
            {
                // do not forget to add empty rows (mandatory for Eigen)
                for(unsigned r = 0; r < N; ++r) {
                    const unsigned row = NOut * i + r;
                    eigen.compressedMatrix.startVec( row );
                }
                continue;
            }

            for(unsigned r = 0; r < N; ++r) {
				const unsigned row = NOut * i + r;

				eigen.compressedMatrix.startVec( row );

				const unsigned col = NIn * i + r;
				eigen.compressedMatrix.insertBack( row, col ) = 1;
			}
			
		}
		
		eigen.compressedMatrix.finalize();
    }

	// std::cout << eigen.compressedMatrix << std::endl;

	return &js;
}


template<class TIn, class TOut>
void IdentityMapping<TIn, TOut>::updateForceMask()
{
    for( size_t i = 0 ; i<this->maskTo->size() ; ++i )
        if( this->maskTo->getEntry(i) ) this->maskFrom->insertEntry( i );

}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
