/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once
#include <sofa/linearalgebra/config.h>

#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/linearalgebra/FullVector.h>
#include <sofa/type/Mat.h>

namespace sofa::linearalgebra
{

/// Simple block full matrix container (used for InvMatrixType)
template< std::size_t N, typename T>
class BlockFullMatrix : public linearalgebra::BaseMatrix
{
public:

    enum { BSIZE = N };
    typedef T Real;

    class TransposedBlock{

    public:
        const type::Mat<BSIZE,BSIZE,Real>& m;

        TransposedBlock(const sofa::type::Mat<BSIZE, BSIZE, Real>& m_a) : m(m_a){}

        type::Vec<BSIZE,Real> operator*(const type::Vec<BSIZE,Real>& v)
        {
            return m.multTranspose(v);
        }

        type::Mat<BSIZE,BSIZE,Real> operator-() const
        {
            return -m.transposed();
        }
    };

    class Block : public type::Mat<BSIZE,BSIZE,Real>
    {
    public:
        Index Nrows() const;
        Index Ncols() const;
        void resize(Index, Index);
        const T& element(Index i, Index j) const;
        void set(Index i, Index j, const T& v);
        void add(Index i, Index j, const T& v);
        void operator=(const type::Mat<BSIZE,BSIZE,Real>& v)
        {
            type::Mat<BSIZE,BSIZE,Real>::operator=(v);
        }
        type::Mat<BSIZE,BSIZE,Real> operator-() const
        {
            return type::Mat<BSIZE,BSIZE,Real>::operator-();
        }
        type::Mat<BSIZE,BSIZE,Real> operator-(const type::Mat<BSIZE,BSIZE,Real>& m) const
        {
            return type::Mat<BSIZE,BSIZE,Real>::operator-(m);
        }
        type::Vec<BSIZE,Real> operator*(const type::Vec<BSIZE,Real>& v)
        {
            return type::Mat<BSIZE,BSIZE,Real>::operator*(v);
        }
        type::Mat<BSIZE,BSIZE,Real> operator*(const type::Mat<BSIZE,BSIZE,Real>& m)
        {
            return sofa::type::operator*(*this, m);
        }
        type::Mat<BSIZE,BSIZE,Real> operator*(const Block& m)
        {
            return sofa::type::operator*(*this, m);
        }
        type::Mat<BSIZE,BSIZE,Real> operator*(const TransposedBlock& mt)
        {
            return operator*(mt.m.transposed());
        }
        TransposedBlock t() const;
        Block i() const;
    };
    typedef Block SubMatrixType;

    // return the dimension of submatrices
    constexpr static Index getSubMatrixDim()
    {
        return BSIZE;
    }

    SOFA_ATTRIBUTE_DISABLED__GETSUBMATRIXSIZE("Use directly getSubMatrixDim(), without any parameter")
    constexpr static Index getSubMatrixDim(Index)
    {
        return getSubMatrixDim();
    }

protected:
    Block* data;
    Index nTRow,nTCol;
    Index nBRow,nBCol;
    Index allocsize;

public:

    BlockFullMatrix();

    BlockFullMatrix(Index nbRow, Index nbCol);

    ~BlockFullMatrix() override;

    Block* ptr() { return data; }
    const Block* ptr() const { return data; }

    const Block& bloc(Index bi, Index bj) const;

    Block& bloc(Index bi, Index bj);

    void resize(Index nbRow, Index nbCol) override;

    Index rowSize(void) const override;

    Index colSize(void) const override;

    SReal element(Index i, Index j) const override;

    const Block& asub(Index bi, Index bj, Index, Index) const;

    const Block& sub(Index i, Index j, Index, Index) const;

    Block& asub(Index bi, Index bj, Index, Index);

    Block& sub(Index i, Index j, Index, Index);

    template<class B>
    void getSubMatrix(Index i, Index j, Index nrow, Index ncol, B& m);

    template<class B>
    void getAlignedSubMatrix(Index bi, Index bj, Index nrow, Index ncol, B& m);

    template<class B>
    void setSubMatrix(Index i, Index j, Index nrow, Index ncol, const B& m);

    template<class B>
    void setAlignedSubMatrix(Index bi, Index bj, Index nrow, Index ncol, const B& m);

    void set(Index i, Index j, double v) override;

    void add(Index i, Index j, double v) override;

    void clear(Index i, Index j) override;

    void clearRow(Index i) override;

    void clearCol(Index j) override;

    void clearRowCol(Index i) override;

    void clear() override;

    template<class Real2>
    FullVector<Real2> operator*(const FullVector<Real2>& v) const
    {
        FullVector<Real2> res(rowSize());
        for (Index bi=0; bi<nBRow; ++bi)
        {
            Index bj = 0;
            for (Index i=0; i<BSIZE; ++i)
            {
                Real r = 0;
                for (Index j=0; j<BSIZE; ++j)
                {
                    r += bloc(bi,bj)[i][j] * v[(bi + bj - 1)*BSIZE + j];
                }
                res[bi*BSIZE + i] = r;
            }
            for (++bj; bj<nBCol; ++bj)
            {
                for (Index i=0; i<BSIZE; ++i)
                {
                    Real r = 0;
                    for (Index j=0; j<BSIZE; ++j)
                    {
                        r += bloc(bi,bj)[i][j] * v[(bi + bj - 1)*BSIZE + j];
                    }
                    res[bi*BSIZE + i] += r;
                }
            }
        }
        return res;
    }


    static const char* Name();
};

#if !defined(SOFA_LINEARALGEBRA_BLOCKFULLMATRIX_CPP)
extern template class SOFA_LINEARALGEBRA_API BlockFullMatrix<6, SReal>;
#endif

} // namespace sofa::linearalgebra
