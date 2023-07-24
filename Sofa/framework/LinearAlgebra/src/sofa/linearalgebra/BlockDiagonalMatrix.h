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

#include <sofa/linearalgebra/FullVector.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/linearalgebra/matrix_bloc_traits.h>

namespace sofa::linearalgebra
{

/// Simple full matrix container
template<std::size_t LC, typename T = SReal>
class BlockDiagonalMatrix : public linearalgebra::BaseMatrix
{
public:
    typedef T Real;
    typedef type::Mat<LC,LC,Real> Block;
    typedef matrix_bloc_traits<Block, Index> traits;

    enum { BSIZE = LC };

    typedef BlockDiagonalMatrix<LC,T> Expr;
    typedef BlockDiagonalMatrix<LC,T> matrix_type;
    enum { category = MATRIX_BAND };
    enum { operand = 1 };


protected:
    std::vector< Block > data;
    sofa::Index cSize;

public:

    BlockDiagonalMatrix()
        : cSize(0)
    {
    }

    ~BlockDiagonalMatrix() override {}

    void resize(Index nbRow, Index ) override
    {
        cSize = nbRow;
        data.resize((cSize+LC-1) / LC);
        //for (Index i=0;i<data.size();i++) data[i].ReSize(LC,LC);
    }

    Index rowSize(void) const override
    {
        return cSize;
    }

    Index colSize(void) const override
    {
        return cSize;
    }

    Index rowBSize(void) const
    {
        return data.size();
    }

    Index colBSize(void) const
    {
        return data.size();
    }

    const Block& bloc(Index i) const
    {
        return data[i];
    }

    const Block& bloc(Index i, Index j) const
    {
        static Block empty;
        if (i != j)
            return empty;
        else
            return bloc(i);
    }

    Block* wbloc(Index i)
    {
        return &(data[i]);
    }

    Block* wbloc(Index i, Index j)
    {
        if (i != j)
            return nullptr;
        else
            return wbloc(i);
    }

    SReal element(Index i, Index j) const override
    {
        Index bi=0, bj=0; traits::split_row_index(i, bi); traits::split_col_index(j, bj);
        if (i != j) return 0;
        else return traits::v(data[i], bi, bj);
    }

    void set(Index i, Index j, double v) override
    {
        Index bi=0, bj=0; traits::split_row_index(i, bi); traits::split_col_index(j, bj);
        if (i == j) traits::v(data[i], bi, bj) = (Real)v;
    }

    void setB(Index i, const Block& b)
    {
        data[i] = b;
    }

    void setB(Index i, Index j, const Block& b)
    {
        if (i == j)
            setB(i, b);
    }

    using BaseMatrix::add;
    void add(Index i, Index j, double v) override
    {
        Index bi=0, bj=0; traits::split_row_index(i, bi); traits::split_col_index(j, bj);
        if (i == j) traits::v(data[i], bi, bj) += (Real)v;
    }

    void addB(Index i, const Block& b)
    {
        data[i] += b;
    }

    void addB(Index i, Index j, const Block& b)
    {
        if (i == j)
            addB(i, b);
    }

    void clear(Index i, Index j) override
    {
        Index bi=0, bj=0; traits::split_row_index(i, bi); traits::split_col_index(j, bj);
        if (i == j) traits::v(data[i], bi, bj) = (Real)0;
    }

    void clearRow(Index i) override
    {
        Index bi=0; traits::split_row_index(i, bi);
        for (Index bj=0; bj<Index(LC); ++bj)
            traits::v(data[i], bi, bj) = (Real)0;
    }

    void clearCol(Index j) override
    {
        Index bj=0; traits::split_col_index(j, bj);
        for (Index bi=0; bi<Index(LC); ++bi)
            traits::v(data[j], bi, bj) = (Real)0;
    }

    void clearRowCol(Index i) override
    {
        Index bi=0; traits::split_row_index(i, bi);
        for (Index bj=0; bj<Index(LC); ++bj)
            traits::v(data[i], bi, bj) = (Real)0;
        for (Index bj=0; bj<Index(LC); ++bj)
            traits::v(data[i], bj, bi) = (Real)0;
    }

    void clear() override
    {
        for (Index b=0; b<(Index)data.size(); b++)
            traits::clear(data[b]);
    }

    void invert()
    {
        for (Index b=0; b<(Index)data.size(); b++)
        {
            const Block m = data[b];
            traits::invert(data[b], m);
        }
    }

    template<class Real2>
    void mul(FullVector<Real2>& res, const FullVector<Real2>& v) const
    {
        res.resize(cSize);
        Index nblocs = cSize;
        Index szlast = 0;
        traits::split_row_index(nblocs, szlast);
        for (sofa::Index b=0; b<(sofa::Size) nblocs; b++)
        {
            const sofa::Index i = b*LC;
            for (sofa::Index bj=0; bj<Index(LC); bj++)
            {
                Real2 r = 0;
                for (sofa::Index bi=0; bi<Index(LC); bi++)
                {
                    r += (Real2)(traits::v(data[b],bi,bj) * v[i+bi]);
                }
                res[i+bj] = r;
            }
        }
        if (szlast)
        {
            sofa::Size b = nblocs;
            const sofa::Index i = b*LC;
            for (sofa::Index bj=0; bj<(sofa::Size) szlast; bj++)
            {
                Real2 r = 0;
                for (sofa::Index bi=0; bi<(sofa::Size) szlast; bi++)
                {
                    r += (Real2)(traits::v(data[b],bi,bj) * v[i+bi]);
                }
                res[i+bj] = r;
            }
        }
    }

    template<class Real2>
    FullVector<Real2> operator*(const FullVector<Real2>& v) const
    {
        FullVector<Real2> res;
        mul(res, v);
        return res;
    }
    
    friend std::ostream& operator << (std::ostream& out, const BlockDiagonalMatrix<LC, T>& v )
    {
        out << "[";
        for (Index i=0; i<(Index)v.data.size(); i++) out << " " << v.data[i];
        out << " ]";
        return out;
    }

    static const char* Name()
    {
        static std::string name = std::string("BlockDiagonalMatrix") + traits::Name();
        return name.c_str();
    }

};

typedef BlockDiagonalMatrix<3> BlockDiagonalMatrix3;
typedef BlockDiagonalMatrix<6> BlockDiagonalMatrix6;
typedef BlockDiagonalMatrix<9> BlockDiagonalMatrix9;
typedef BlockDiagonalMatrix<12> BlockDiagonalMatrix12;

#if !defined(SOFA_LINEARALGEBRA_BLOCKDIAGONALMATRIX_CPP)
extern template class SOFA_LINEARALGEBRA_API BlockDiagonalMatrix<3, SReal>;
#endif


} // namespace sofa::linearalgebra
