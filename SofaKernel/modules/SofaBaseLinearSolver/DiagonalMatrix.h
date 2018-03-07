/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_DIAGONALMATRIX_H
#define SOFA_COMPONENT_LINEARSOLVER_DIAGONALMATRIX_H
#include "config.h"

#include "MatrixExpr.h"
#include <SofaBaseLinearSolver/matrix_bloc_traits.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

/// Simple full matrix container
template<typename T>
class DiagonalMatrix : public defaulttype::BaseMatrix
{
public:
    typedef T Real;

    typedef DiagonalMatrix<T> Expr;
    typedef DiagonalMatrix<double> matrix_type;
    enum { category = MATRIX_DIAGONAL };
    enum { operand = 1 };

protected:
    FullVector<T> data;

public:

    DiagonalMatrix()
    {
    }

    DiagonalMatrix(Index nbRow, Index /*nbCol*/)
        : data(new T[nbRow])
    {
    }

    DiagonalMatrix(Real* p, Index /*nbRow*/, Index /*nbCol*/)
        : data(p)
    {
    }

    ~DiagonalMatrix() {}

    Real* ptr() { return data.ptr(); }
    const Real* ptr() const { return data.ptr(); }

    Real* operator[](Index i)
    {
        return data+i;
    }

    const Real* operator[](Index i) const
    {
        return data+i;
    }

    void resize(Index nbRow, Index /*nbCol*/)
    {
        data.resize(nbRow);
    }

    Index rowSize(void) const
    {
        return data.size();
    }

    Index colSize(void) const
    {
        return data.size();
    }

    SReal element(Index i, Index j) const
    {
        if (i!=j) return (Real)0;
        return data[i];
    }

    void set(Index i, Index j, double v)
    {
        if (i==j) data[i] = (Real)v;
    }

    void add(Index i, Index j, double v)
    {
        if (i==j) data[i] += (Real)v;
    }

    void clear(Index i, Index j)
    {
        if (i==j) data[i] = (Real)0;
    }

    void clearRow(Index i)
    {
        data[i] = (Real)0;
    }

    void clearCol(Index j)
    {
        data[j] = (Real)0;
    }

    void clearRowCol(Index i)
    {
        data[i] = (Real)0;
    }

    void clear()
    {
        data.clear();
    }

    // operators similar to vectors

    void resize(Index nbRow)
    {
        data.resize(nbRow);
    }

    Index size() const
    {
        return data.size();
    }

    void swap(DiagonalMatrix<T>& v)
    {
        data.swap(v.data);
    }

    SReal element(Index i) const
    {
        return data[i];
    }

    void set(Index i, double v)
    {
        data[i] = (Real)v;
    }

    void add(Index i, double v)
    {
        data[i] += (Real)v;
    }

    void clear(Index i)
    {
        data[i] = (Real)0;
    }

    template<class Real2>
    FullVector<Real2> operator*(const FullVector<Real2>& v) const
    {
        FullVector<Real2> res;
        res.resize(rowSize());
        for (Index i=0; i<rowSize(); i++) res[i] = data[i] * v[i];
        return res;
    }

    void invert()
    {
        for (Index i=0; i<rowSize(); i++) data[i] = 1.0 / data[i];
    }

    template<class Real2>
    void mult(FullVector<Real2>& z,const FullVector<Real2>& v) const
    {
        for (Index i=0; i<rowSize(); i++) z[i] = data[i] * v[i];
    }

    // methods for MatrixExpr support

    template<class M2>
    bool hasRef(const M2* m) const
    {
        return (const void*)this == (const void*)m;
    }

    std::string expr() const
    {
        return std::string(Name());
    }

    bool valid() const
    {
        return true;
    }

    template<class Dest>
    void addTo(Dest* dest) const
    {
        Index ny = rowSize();
        for (Index y=0; y<ny; ++y)
            dest->add(y,y,element(y));
    }

protected:

    template<class M>
    void equal(const M& m, bool add = false)
    {
        if (m.hasRef(this))
        {
            DiagonalMatrix<T> tmp;
            tmp.resize(m.rowSize(), m.colSize());
            m.addTo(&tmp);
            if (add)
                tmp.addTo(this);
            else
                swap(tmp);
        }
        else
        {
            if (!add)
                resize(m.rowSize(), m.colSize());
            m.addTo(this);
        }
    }

    /// this += m
    template<class M>
    inline void addEqual( const M& m )
    {
        equal( m, true );
    }


public:

    template<class Real2>
    void operator=(const DiagonalMatrix<Real2>& m)
    {
        if (&m == this) return;
        resize(m.rowSize(), m.colSize());
        m.addTo(this);
    }

    template<class Real2>
    void operator+=(const DiagonalMatrix<Real2>& m)
    {
        addEqual(m);
    }

    template<class Real2>
    void operator-=(const DiagonalMatrix<Real2>& m)
    {
        addEqual(MatrixExpr< MatrixNegative< DiagonalMatrix<Real2> > >(MatrixNegative< DiagonalMatrix<Real2> >(m)));
    }

    template<class Expr2>
    void operator=(const MatrixExpr< Expr2 >& m)
    {
        equal(m, false);
    }

    template<class Expr2>
    void operator+=(const MatrixExpr< Expr2 >& m)
    {
        addEqual(m);
    }

    template<class Expr2>
    void operator-=(const MatrixExpr< Expr2 >& m)
    {
        addEqual(MatrixExpr< MatrixNegative< Expr2 > >(MatrixNegative< Expr2 >(m)));
    }

    MatrixExpr< MatrixTranspose< DiagonalMatrix<T> > > t() const
    {
        return MatrixExpr< MatrixTranspose< DiagonalMatrix<T> > >(MatrixTranspose< DiagonalMatrix<T> >(*this));
    }

    MatrixExpr< MatrixInverse< DiagonalMatrix<T> > > i() const
    {
        return MatrixExpr< MatrixInverse< DiagonalMatrix<T> > >(MatrixInverse< DiagonalMatrix<T> >(*this));
    }

    MatrixExpr< MatrixNegative< DiagonalMatrix<T> > > operator-() const
    {
        return MatrixExpr< MatrixNegative< DiagonalMatrix<T> > >(MatrixNegative< DiagonalMatrix<T> >(*this));
    }

    MatrixExpr< MatrixScale< DiagonalMatrix<T>, double > > operator*(const double& r) const
    {
        return MatrixExpr< MatrixScale< DiagonalMatrix<T>, double > >(MatrixScale< DiagonalMatrix<T>, double >(*this, r));
    }

    friend std::ostream& operator << (std::ostream& out, const DiagonalMatrix<T>& v )
    {
        Index ny = v.rowSize();
        out << "[";
        for (Index y=0; y<ny; ++y) out << " " << v.element(y);
        out << " ]";
        return out;
    }

    static const char* Name() { return "DiagonalMatrix"; }
};


/// Simple full matrix container
template<int LC, typename T = double>
class BlockDiagonalMatrix : public defaulttype::BaseMatrix
{
public:
    typedef T Real;
    typedef defaulttype::Mat<LC,LC,Real> Bloc;
    typedef matrix_bloc_traits<Bloc> traits;

    enum { BSIZE = LC };

    typedef BlockDiagonalMatrix<LC,T> Expr;
    typedef BlockDiagonalMatrix<LC,double> matrix_type;
    enum { category = MATRIX_BAND };
    enum { operand = 1 };


protected:
    std::vector< Bloc > data;
    Index cSize;

public:

    BlockDiagonalMatrix()
        : cSize(0)
    {
    }

    ~BlockDiagonalMatrix() {}

    void resize(Index nbRow, Index )
    {
        cSize = nbRow;
        data.resize((cSize+LC-1) / LC);
        //for (Index i=0;i<data.size();i++) data[i].ReSize(LC,LC);
    }

    Index rowSize(void) const
    {
        return cSize;
    }

    Index colSize(void) const
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

    const Bloc& bloc(Index i) const
    {
        return data[i];
    }

    const Bloc& bloc(Index i, Index j) const
    {
        static Bloc empty;
        if (i != j)
            return empty;
        else
            return bloc(i);
    }

    Bloc* wbloc(Index i)
    {
        return &(data[i]);
    }

    Bloc* wbloc(Index i, Index j)
    {
        if (i != j)
            return NULL;
        else
            return wbloc(i);
    }

    SReal element(Index i, Index j) const
    {
        Index bi=0, bj=0; traits::split_row_index(i, bi); traits::split_col_index(j, bj);
        if (i != j) return 0;
        else return traits::v(data[i], bi, bj);
    }

    void set(Index i, Index j, double v)
    {
        Index bi=0, bj=0; traits::split_row_index(i, bi); traits::split_col_index(j, bj);
        if (i == j) traits::v(data[i], bi, bj) = (Real)v;
    }

    void setB(Index i, const Bloc& b)
    {
        data[i] = b;
    }

    void setB(Index i, Index j, const Bloc& b)
    {
        if (i == j)
            setB(i, b);
    }

    void add(Index i, Index j, double v)
    {
        Index bi=0, bj=0; traits::split_row_index(i, bi); traits::split_col_index(j, bj);
        if (i == j) traits::v(data[i], bi, bj) += (Real)v;
    }

    void addB(Index i, const Bloc& b)
    {
        data[i] += b;
    }

    void addB(Index i, Index j, const Bloc& b)
    {
        if (i == j)
            addB(i, b);
    }

    void clear(Index i, Index j)
    {
        Index bi=0, bj=0; traits::split_row_index(i, bi); traits::split_col_index(j, bj);
        if (i == j) traits::v(data[i], bi, bj) = (Real)0;
    }

    void clearRow(Index i)
    {
        Index bi=0; traits::split_row_index(i, bi);
        for (Index bj=0; bj<LC; ++bj)
            traits::v(data[i], bi, bj) = (Real)0;
    }

    void clearCol(Index j)
    {
        Index bj=0; traits::split_col_index(j, bj);
        for (Index bi=0; bi<LC; ++bi)
            traits::v(data[j], bi, bj) = (Real)0;
    }

    void clearRowCol(Index i)
    {
        Index bi=0; traits::split_row_index(i, bi);
        for (Index bj=0; bj<LC; ++bj)
            traits::v(data[i], bi, bj) = (Real)0;
        for (Index bj=0; bj<LC; ++bj)
            traits::v(data[i], bj, bi) = (Real)0;
    }

    void clear()
    {
        for (Index b=0; b<(Index)data.size(); b++)
            traits::clear(data[b]);
    }

    void invert()
    {
        for (Index b=0; b<(Index)data.size(); b++)
        {
            const Bloc m = data[b];
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
        for (Index b=0; b<nblocs; b++)
        {
            Index i = b*LC;
            for (Index bj=0; bj<LC; bj++)
            {
                Real2 r = 0;
                for (Index bi=0; bi<LC; bi++)
                {
                    r += (Real2)(traits::v(data[b],bi,bj) * v[i+bi]);
                }
                res[i+bj] = r;
            }
        }
        if (szlast)
        {
            Index b = nblocs;
            Index i = b*LC;
            for (Index bj=0; bj<szlast; bj++)
            {
                Real2 r = 0;
                for (Index bi=0; bi<szlast; bi++)
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

    friend std::ostream& operator << (std::ostream& out, const BlockDiagonalMatrix<LC>& v )
    {
        out << "[";
        for (Index i=0; i<(Index)v.data.size(); i++) out << " " << v.data[i];
        out << " ]";
        return out;
    }

    static const char* Name()
    {
        static std::string name = std::string("BlockDiagonalMatrix") + std::string(traits::Name());
        return name.c_str();
    }

};

typedef BlockDiagonalMatrix<3> BlockDiagonalMatrix3;
typedef BlockDiagonalMatrix<6> BlockDiagonalMatrix6;
typedef BlockDiagonalMatrix<9> BlockDiagonalMatrix9;
typedef BlockDiagonalMatrix<12> BlockDiagonalMatrix12;

// trivial product and inverse operations for diagonal matrices

template<class R1, class M2>
class MatrixProductOp<DiagonalMatrix<R1>, M2>
{
    typedef typename M2::Index Index;
protected:
    template<class Dest>
    class MyDest
    {
    public:
        const DiagonalMatrix<R1>& m1;
        Dest* d;
        MyDest(const DiagonalMatrix<R1>& m1, Dest* d) : m1(m1), d(d) {}
        void add(Index l, Index c, double v) { d->add(l,c,m1.element(l)*v); }
    };
public:
    typedef typename M2::matrix_type matrix_type;
    enum { category = M2::category };

    template<class Dest>
    void operator()(const DiagonalMatrix<R1>& m1, const M2& m2, Dest* d)
    {
        MyDest<Dest> myd(m1,d);
        std::cout << "EXPR using diagonal pre-product: " << m1.expr() << " * " << m2.expr() << std::endl;
        m2.addTo(&myd);
    }
};

template<class M1, class R2>
class MatrixProductOp<M1, DiagonalMatrix<R2> >
{
    typedef typename M1::Index Index;
protected:
    template<class Dest>
    class MyDest
    {
    public:
        const DiagonalMatrix<R2>& m2;
        Dest* d;
        MyDest(const DiagonalMatrix<R2>& m2, Dest* d) : m2(m2), d(d) {}
        void add(Index l, Index c, double v) { d->add(l,c,v*m2.element(c)); }
    };
public:
    typedef typename M1::matrix_type matrix_type;
    enum { category = M1::category };

    template<class Dest>
    void operator()(const M1& m1, const DiagonalMatrix<R2>& m2, Dest* d)
    {
        MyDest<Dest> myd(m2,d);
        std::cout << "EXPR using diagonal post-product: " << m1.expr() << " * " << m2.expr() << std::endl;
        m1.addTo(&myd);
    }
};

template<class R1, class R2>
class MatrixProductOp<DiagonalMatrix<R1>, DiagonalMatrix<R2> >
{
public:
    typedef DiagonalMatrix<R1> M1;
    typedef DiagonalMatrix<R2> M2;
    typedef typename M1::Index Index;
    typedef typename type_selector<(sizeof(R2)>sizeof(R1)),M1,M2>::T matrix_type;
    enum { category = matrix_type::category };

    template<class Dest>
    void operator()(const DiagonalMatrix<R1>& m1, const DiagonalMatrix<R2>& m2, Dest* d)
    {
        Index n = m1.size();
        std::cout << "EXPR using diagonal product: " << m1.expr() << " * " << m2.expr() << std::endl;
        for (Index i=0; i<n; ++i)
            d->add(i,i,m1.element(i)*m2.element(i));
    }
};

template<class R1>
class MatrixInvertOp<DiagonalMatrix<R1> >
{
public:
    typedef DiagonalMatrix<R1> matrix_type;
    typedef typename matrix_type::Index Index;
    enum { category = matrix_type::category };

    template<class Dest>
    void operator()(const DiagonalMatrix<R1>& m1, Dest* d)
    {
        Index n = m1.size();
        for (Index i=0; i<n; ++i)
            d->add(i,i,1.0/m1.element(i));
    }
};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
