/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_MATRIXEXPR_H
#define SOFA_COMPONENT_LINEARSOLVER_MATRIXEXPR_H
#include "config.h"

#include <sofa/defaulttype/BaseMatrix.h>
#include <sstream>

namespace sofa
{

namespace component
{

namespace linearsolver
{

//#define SPARSEMATRIX_CHECK
//#define SPARSEMATRIX_VERBOSE

template<class M1, class M2>
class MatrixProduct;

template<class M1, class M2>
class MatrixAddition;

template<class M1, class M2>
class MatrixSubstraction;

template<class M1>
class MatrixTranspose;

template<class M1>
class MatrixNegative;

template<class M1, class R2>
class MatrixScale;

template<class T>
class MatrixExpr : public T
{
public:
    typedef T Expr;

    MatrixExpr(const Expr& e) : Expr(e) {}

    template<class M2>
    MatrixExpr< MatrixProduct< Expr, typename M2::Expr > > operator*(const M2& m) const
    {
        return MatrixExpr< MatrixProduct< Expr, typename M2::Expr > >(MatrixProduct< Expr, typename M2::Expr >(*this, m));
    }
    template<class M2>
    MatrixExpr< MatrixAddition< Expr, typename M2::Expr > > operator+(const M2& m) const
    {
        return MatrixExpr< MatrixAddition< Expr, typename M2::Expr > >(MatrixAddition< Expr, typename M2::Expr >(*this, m));
    }
    template<class M2>
    MatrixExpr< MatrixSubstraction< Expr, typename M2::Expr > > operator-(const M2& m) const
    {
        return MatrixExpr< MatrixSubstraction< Expr, typename M2::Expr > >(MatrixSubstraction< Expr, typename M2::Expr >(*this, m));
    }
    MatrixExpr< MatrixNegative< Expr > > operator-() const
    {
        return MatrixExpr< MatrixNegative< Expr > >(MatrixNegative< Expr >(*this));
    }
    MatrixExpr< MatrixTranspose< Expr > > t() const
    {
        return MatrixExpr< MatrixTranspose< Expr > >(MatrixTranspose< Expr >(*this));
    }

    MatrixExpr< MatrixScale< Expr, double > > operator*(double d) const
    {
        return MatrixExpr< MatrixScale< Expr, double > >(MatrixScale< Expr, double >(*this, d));
    }
    friend MatrixExpr< MatrixScale< Expr, double > > operator*(double d, const MatrixExpr<Expr>& m)
    {
        return MatrixExpr< MatrixScale< Expr, double > >(MatrixScale< Expr, double >(m, d));
    }
    template<class M1>
    friend MatrixExpr< MatrixProduct< typename M1::Expr, Expr > > operator*(const M1& m1, const MatrixExpr<Expr>& m2)
    {
        return MatrixExpr< MatrixProduct< typename M1::Expr, Expr > >(MatrixProduct< typename M1::Expr, Expr >(m1,m2));
    }
    template<class M1>
    friend MatrixExpr< MatrixAddition< typename M1::Expr, Expr > > operator+(const M1& m1, const MatrixExpr<Expr>& m2)
    {
        return MatrixExpr< MatrixAddition< typename M1::Expr, Expr > >(MatrixAddition< typename M1::Expr, Expr >(m1,m2));
    }
    template<class M1>
    friend MatrixExpr< MatrixSubstraction< typename M1::Expr, Expr > > operator-(const M1& m1, const MatrixExpr<Expr>& m2)
    {
        return MatrixExpr< MatrixSubstraction< typename M1::Expr, Expr > >(MatrixSubstraction< typename M1::Expr, Expr >(m1,m2));
    }
};

enum MatrixCategory
{
    MATRIX_IDENTITY = 0,
    MATRIX_DIAGONAL,
    MATRIX_BAND,
    MATRIX_SPARSE,
    MATRIX_FULL
};

template<int op1, int op2, class M1, class M2>
class DefaultMatrixProductOp;

template<class M1, class M2>
class MatrixProductOp : public DefaultMatrixProductOp<M1::operand, M2::operand, M1, M2>
{
};

template<class M1, class M2>
class DefaultMatrixProductOp<0, 0, M1, M2>
{
public:
    typedef typename M1::matrix_type matrix1_type;
    typedef typename M2::matrix_type matrix2_type;
    typedef MatrixProductOp<matrix1_type, matrix2_type> final_op;
    typedef typename final_op::matrix_type matrix_type;
    enum { category = final_op::category };

    template<class Dest>
    void operator()(const M1& m1, const M2& m2, Dest* d)
    {
        matrix1_type tmp1;
        matrix2_type tmp2;
        tmp1 = MatrixExpr<M1>(m1);
        tmp2 = MatrixExpr<M2>(m2);
        final_op op;
        op(tmp1, tmp2, d);
    }
};

template<class M1, class M2>
class DefaultMatrixProductOp<1, 0, M1, M2>
{
public:
    typedef typename M2::matrix_type matrix2_type;
    typedef MatrixProductOp<M1, matrix2_type> final_op;
    typedef typename final_op::matrix_type matrix_type;
    enum { category = final_op::category };

    template<class Dest>
    void operator()(const M1& m1, const M2& m2, Dest* d)
    {
        matrix2_type tmp2;
        tmp2 = MatrixExpr<M2>(m2);
        final_op op;
        op(m1, tmp2, d);
    }
};

template<class M1, class M2>
class DefaultMatrixProductOp<0, 1, M1, M2>
{
public:
    typedef typename M1::matrix_type matrix1_type;
    typedef MatrixProductOp<matrix1_type, M2> final_op;
    typedef typename final_op::matrix_type matrix_type;
    enum { category = final_op::category };

    template<class Dest>
    void operator()(const M1& m1, const M2& m2, Dest* d)
    {
        matrix1_type tmp1;
        tmp1 = MatrixExpr<M1>(m1);
        final_op op;
        op(tmp1, m2, d);
    }
};

template<int op1, class M1>
class DefaultMatrixInvertOp;

template<class M1>
class MatrixInvertOp : public DefaultMatrixInvertOp<M1::operand,M1>
{
};

template<class M1>
class DefaultMatrixInvertOp<0,M1>
{
public:
    typedef typename M1::matrix_type matrix1_type;
    typedef MatrixInvertOp<matrix1_type> final_op;
    typedef typename final_op::matrix_type matrix_type;
    enum { category = final_op::category };

    template<class Dest>
    void operator()(const M1& m1, Dest* d)
    {
        matrix1_type tmp1;
        tmp1 = MatrixExpr<M1>(m1);
        final_op op;
        op(tmp1, d);
    }
};

template<class M1>
class MatrixNegative
{
public:
    typedef MatrixNegative<M1> Expr;
    enum { operand = 0 };
    enum { category = M1::category };
    typedef typename M1::matrix_type matrix_type;

    const M1& m1;
    MatrixNegative(const M1& m1) : m1(m1)
    {}

    bool valid() const
    {
        return &m1 && m1.valid();
    }

    template<class M>
    bool hasRef(const M* m) const
    {
        return m1.hasRef(m);
    }

    unsigned int rowSize(void) const
    {
        return m1.colSize();
    }

    unsigned int colSize(void) const
    {
        return m1.rowSize();
    }

    std::string expr() const
    {
        return std::string("-(")+m1.expr()+std::string(")");
    }

protected:
    template<class Dest>
    class MyDest
    {
    public:
        Dest* d;
        MyDest(Dest* d) : d(d) {}
        void add(int l, int c, double v) { d->add(l,c,-v); }
    };

public:
    template<class Dest>
    void addTo(Dest* d) const
    {
        MyDest<Dest> myd(d);
        m1.addTo(&myd);
    }
};

template<class M1>
class MatrixTranspose
{
public:
    typedef MatrixTranspose<M1> Expr;
    enum { operand = 0 };
    enum { category = M1::category };
    typedef typename M1::matrix_type matrix_type;

    const M1& m1;
    MatrixTranspose(const M1& m1) : m1(m1)
    {}

    bool valid() const
    {
        return &m1 && m1.valid();
    }

    template<class M>
    bool hasRef(const M* m) const
    {
        return m1.hasRef(m);
    }

    unsigned int rowSize(void) const
    {
        return m1.colSize();
    }

    unsigned int colSize(void) const
    {
        return m1.rowSize();
    }

    std::string expr() const
    {
        return std::string("(")+m1.expr()+std::string(")^t");
    }

protected:
    template<class Dest>
    class MyDest
    {
    public:
        Dest* d;
        MyDest(Dest* d) : d(d) {}
        void add(int l, int c, double v) { d->add(c,l,v); }
    };

public:
    template<class Dest>
    void addTo(Dest* d) const
    {
        std::cout << "EXPR using transposed computation: " << expr() << std::endl;
        MyDest<Dest> myd(d);
        m1.addTo(&myd);
    }
};

template<class M1, class R2>
class MatrixScale
{
public:
    typedef MatrixScale<M1,R2> Expr;
    enum { operand = 0 };
    enum { category = M1::category };
    typedef typename M1::matrix_type matrix_type;

    const M1& m1;
    const R2 r2;
    MatrixScale(const M1& m1, const R2& r2) : m1(m1), r2(r2)
    {}

    bool valid() const
    {
        return &m1 && m1.valid();
    }

    template<class M>
    bool hasRef(const M* m) const
    {
        return m1.hasRef(m);
    }

    unsigned int rowSize(void) const
    {
        return m1.colSize();
    }

    unsigned int colSize(void) const
    {
        return m1.rowSize();
    }

    std::string expr() const
    {
        std::ostringstream o;
        o << "(" << m1.expr() << ")*" << r2;
        return o.str();
    }

protected:
    template<class Dest>
    class MyDest
    {
    public:
        Dest* d;
        const R2 r2;
        MyDest(const R2& r2, Dest* d) : d(d), r2(r2) {}
        void add(int l, int c, double v) { d->add(l,c,v*r2); }
    };

public:
    template<class Dest>
    void addTo(Dest* d) const
    {
        MyDest<Dest> myd(r2,d);
        m1.addTo(&myd);
    }
};

template<int index, class T0, class T2> class type_selector;
template<class T0, class T1> class type_selector<0,T0,T1> { public: typedef T0 T; };
template<class T0, class T1> class type_selector<1,T0,T1> { public: typedef T1 T; };

template<class M1, class M2>
class MatrixAddition
{
public:
    typedef MatrixAddition<M1, M2> Expr;
    enum { operand = 0 };
    enum { category = ((int)M1::category>(int)M2::category) ? (int)M1::category : (int)M2::category };
    enum { m_index = (((int)M1::category>(int)M2::category) ? 0 : 1) };
    typedef typename type_selector<m_index,typename M1::matrix_type,typename M2::matrix_type>::T matrix_type;

    const M1& m1;
    const M2& m2;
    MatrixAddition(const M1& m1, const M2& m2) : m1(m1), m2(m2)
    {}

    bool valid() const
    {
        return &m1 && &m2 && m1.colSize() == m2.colSize() && m1.rowSize() == m2.rowSize() && m1.valid() && m2.valid();
    }

    template<class M>
    bool hasRef(const M* m) const
    {
        return m1.hasRef(m) || m2.hasRef(m);
    }

    unsigned int rowSize(void) const
    {
        return m1.rowSize();
    }

    unsigned int colSize(void) const
    {
        return m1.colSize();
    }

    std::string expr() const
    {
        return std::string("(")+m1.expr()+std::string(")+(")+m2.expr()+std::string(")");
    }

    template<class Dest>
    void addTo(Dest* d) const
    {
        m1.addTo(d);
        m2.addTo(d);
    }
};

template<class M1, class M2>
class MatrixSubstraction
{
public:
    typedef MatrixSubstraction<M1, M2> Expr;
    enum { operand = 0 };
    enum { category = ((int)M1::category>(int)M2::category) ? (int)M1::category : (int)M2::category };
    enum { m_index = (((int)M1::category>(int)M2::category)?0:1) };
    typedef typename type_selector<m_index,typename M1::matrix_type,typename M2::matrix_type>::T matrix_type;

    const M1& m1;
    const M2& m2;
    MatrixSubstraction(const M1& m1, const M2& m2) : m1(m1), m2(m2)
    {}

    bool valid() const
    {
        return &m1 && &m2 && m1.colSize() == m2.colSize() && m1.rowSize() == m2.rowSize() && m1.valid() && m2.valid();
    }

    template<class M>
    bool hasRef(const M* m) const
    {
        return m1.hasRef(m) || m2.hasRef(m);
    }

    unsigned int rowSize(void) const
    {
        return m1.rowSize();
    }

    unsigned int colSize(void) const
    {
        return m1.colSize();
    }

    std::string expr() const
    {
        return std::string("(")+m1.expr()+std::string(")-(")+m2.expr()+std::string(")");
    }

protected:
    template<class Dest>
    class MyDest
    {
    public:
        Dest* d;
        MyDest(Dest* d) : d(d) {}
        void add(int l, int c, double v) { d->add(l,c,-v); }
    };

public:
    template<class Dest>
    void addTo(Dest* d) const
    {
        m1.addTo(d);
        MyDest<Dest> myd(d);
        m2.addTo(&myd);
    }
};

template<class M1, class M2>
class MatrixProduct
{
public:
    typedef MatrixProduct<M1, M2> Expr;
    typedef MatrixProductOp<M1,M2> Op;
    enum { operand = 0 };
    enum { category = Op::category };
    typedef typename Op::matrix_type matrix_type;

    const M1& m1;
    const M2& m2;
    MatrixProduct(const M1& m1, const M2& m2) : m1(m1), m2(m2)
    {}

    bool valid() const
    {
        return &m1 && &m2 && m1.colSize() == m2.rowSize() && m1.valid() && m2.valid();
    }

    template<class M>
    bool hasRef(const M* m) const
    {
        return m1.hasRef(m) || m2.hasRef(m);
    }

    unsigned int rowSize(void) const
    {
        return m1.rowSize();
    }

    unsigned int colSize(void) const
    {
        return m2.colSize();
    }

    std::string expr() const
    {
        return std::string("(")+m1.expr()+std::string(")*(")+m2.expr()+std::string(")");
    }

    template<class Dest>
    void addTo(Dest* d) const
    {
        Op op;
        op(m1, m2, d);
    }
};

template<class M1>
class MatrixInverse
{
public:
    typedef MatrixInverse<M1> Expr;
    typedef MatrixInvertOp<M1> Op;
    enum { operand = 0 };
    enum { category = Op::category };
    typedef typename Op::matrix_type matrix_type;

    const M1& m1;
    MatrixInverse(const M1& m1) : m1(m1)
    {}

    bool valid() const
    {
        return &m1 && m1.valid() && m1.rowSize() == m1.colSize();
    }

    template<class M>
    bool hasRef(const M* m) const
    {
        return m1.hasRef(m);
    }

    unsigned int rowSize(void) const
    {
        return m1.colSize();
    }

    unsigned int colSize(void) const
    {
        return m1.rowSize();
    }

    std::string expr() const
    {
        return std::string("(")+m1.expr()+std::string(")^{-1}");
    }

    template<class Dest>
    void addTo(Dest* d) const
    {
        Op op;
        op(m1, d);
    }
};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
