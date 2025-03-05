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
#include <sofa/linearalgebra/MatrixExpr.h>

#include <map>

//////////////////// FORWARD DEFINITION ////////////////////////////////////////////////////////////
namespace sofa::linearalgebra 
{
    template<typename T>
    class SparseMatrix ;
}

MSG_REGISTER_CLASS(sofa::linearalgebra::SparseMatrix<float>, "SparseMatrix<float>")
MSG_REGISTER_CLASS(sofa::linearalgebra::SparseMatrix<double>, "SparseMatrix<double>")


///////////////////////////  DEFINITION ////////////////////////////////////////////////////////////
namespace sofa::linearalgebra
{

/** This is basically a map of map of T, wrapped in a linearalgebra::BaseMatrix interface.
 The const access methods avoid creating the entries when they do not exist.
*/
template<typename T>
class SparseMatrix : public linearalgebra::BaseMatrix
{
public:
    typedef T Real;
    typedef std::map<Index,Real> Line;
    typedef std::map<Index,Line> Data;
    typedef typename Line::iterator LElementIterator;
    typedef typename Line::const_iterator LElementConstIterator;
    typedef typename Data::iterator LineIterator;
    typedef typename Data::const_iterator LineConstIterator;

    typedef SparseMatrix<T> Expr;
    typedef SparseMatrix<double> matrix_type;
    enum { category = MATRIX_SPARSE };
    enum { operand = 1 };

protected:
    Data data;
    Index nRow,nCol;

public:

    SparseMatrix()
        : nRow(0), nCol(0)
    {
    }

    SparseMatrix(Index nbRow, Index nbCol)
        : nRow(nbRow), nCol(nbCol)
    {
    }

    LineIterator begin() { return data.begin(); }
    LineIterator end()   { return data.end();   }
    LineConstIterator begin() const { return data.begin(); }
    LineConstIterator end()   const { return data.end();   }

    Line& operator[](Index i)
    {
        return data[i];
    }

    const Line& operator[](Index i) const
    {
        static const Line empty;
        LineConstIterator it = data.find(i);
        if (it==data.end())
            return empty;
        else
            return it->second;
    }

    void resize(Index nbRow, Index nbCol) override
    {
        if ( SPARSEMATRIX_VERBOSE && (nbRow != rowSize() || nbCol != colSize()) )
        {
            msg_info() << ": resize(" << nbRow << "," << nbCol << ")" ;
        }
        data.clear();
        nRow = nbRow;
        nCol = nbCol;
    }

    Index rowSize(void) const override
    {
        return nRow;
    }

    Index colSize(void) const override
    {
        return nCol;
    }

    SReal element(Index i, Index j) const override
    {
        if ( SPARSEMATRIX_CHECK && (i >= rowSize() || j >= colSize()) )
        {
            dmsg_error() << "ERROR: invalid read access to element (" << i << "," << j << ") in " << /* this->Name() <<*/" of size (" << rowSize() << "," << colSize() << ")" ;
            return 0.0;
        }
        LineConstIterator it = data.find(i);
        if (it==data.end())
            return 0.0;
        LElementConstIterator ite = it->second.find(j);
        if (ite == it->second.end())
            return 0.0;
        return (SReal)ite->second;
    }

    void set(Index i, Index j, double v) override
    {
        msg_info_when(SPARSEMATRIX_VERBOSE) << "(" << rowSize() << "," << colSize() << "): element(" << i << "," << j << ") = " ;
        if ( SPARSEMATRIX_CHECK && (i >= rowSize() || j >= colSize()) )
        {
            msg_error() << "Invalid write access to element (" << i << "," << j << ") in " << /* this->Name() <<*/" of size (" << rowSize() << "," << colSize() << ")" ;
            return;
        }
        data[i][j] = (Real)v;
    }

    using BaseMatrix::add;
    void add(Index i, Index j, double v) override
    {
        msg_info_when(SPARSEMATRIX_VERBOSE) << "(" << rowSize() << "," << colSize() << "): element(" << i << "," << j << ") += " << v ;
        if ( SPARSEMATRIX_CHECK && (i >= rowSize() || j >= colSize()) )
        {
            msg_error() << "Invalid write access to element (" << i << "," << j << ") in " << /* this->Name() <<*/" of size (" << rowSize() << "," << colSize() << ")" ;
            return;
        }
        data[i][j] += (Real)v;
    }

    void clear(Index i, Index j) override
    {
        msg_info_when(SPARSEMATRIX_VERBOSE) << "(" << rowSize() << "," << colSize() << "): element(" << i << "," << j << ") = 0" ;
        if ( SPARSEMATRIX_CHECK && (i >= rowSize() || j >= colSize()) )
        {
            msg_error() << "Invalid write access to element (" << i << "," << j << ") in " << /* this->Name() <<*/" of size (" << rowSize() << "," << colSize() << ")";
            return;
        }
        LineIterator it = data.find(i);
        if (it==data.end())
            return;
        LElementIterator ite = it->second.find(j);
        if (ite == it->second.end())
            return;
        it->second.erase(ite);
        if (it->second.empty())
            data.erase(it);
    }

    void clearRow(Index i) override
    {
        msg_info_when(SPARSEMATRIX_VERBOSE) << "(" << rowSize() << "," << colSize() << "): row(" << i << ") = 0" ;
        if ( SPARSEMATRIX_CHECK && (i >= rowSize()) )
        {
            msg_error() << "Invalid write access to row " << i << " in " << /* this->Name() <<*/" of size (" << rowSize() << "," << colSize() << ")" ;
            return;
        }

        LineIterator it = data.find(i);
        if (it==data.end())
            return;
        data.erase(it);
    }

    void clearCol(Index j) override
    {
        msg_info_when(SPARSEMATRIX_VERBOSE) << "(" << rowSize() << "," << colSize() << "): col(" << j << ") = 0" ;
        if ( SPARSEMATRIX_CHECK && (j >= colSize()) )
        {
            msg_error() << "Invalid write access to column " << j << " in " << /* this->Name() <<*/" of size (" << rowSize() << "," << colSize() << ")" ;
            return;
        }
        for(LineIterator it=data.begin(),itend=data.end(); it!=itend; ++it)
        {
            LElementIterator ite = it->second.find(j);
            if (ite != it->second.end())
                it->second.erase(ite);
        }
    }

    void clearRowCol(Index i) override
    {
        msg_info_when(SPARSEMATRIX_VERBOSE) << "(" << rowSize() << "," << colSize() << "): row(" << i << ") = 0 and col(" << i << ") = 0" ;
        if ( SPARSEMATRIX_CHECK && (i >= rowSize() || i >= colSize()) )
        {
            msg_error() << "Invalid write access to row and column " << i << " in " << /* this->Name() <<*/" of size (" << rowSize() << "," << colSize() << ")";
            return;
        }
        clearRow(i);
        clearCol(i);
    }

    void clear() override { data.clear(); }

    template<class Real2>
    void mul(FullVector<Real2>& res, const FullVector<Real2>& v) const
    {
        res.resize(rowSize());
        for (LineConstIterator itl = begin(), itlend=end(); itl!=itlend; ++itl)
        {
            Real2 r = 0;
            for (LElementConstIterator ite = itl->second.begin(), iteend=itl->second.end(); ite!=iteend; ++ite)
                r += (Real2)ite->second * v[ite->first];
            res[itl->first] = r;
        }
    }

    template<class Real2>
    void addMulTranspose(FullVector<Real2>& res, const FullVector<Real2>& v) const
    {
        res.resize(colSize());
        for (LineConstIterator itl = begin(), itlend=end(); itl!=itlend; ++itl)
        {
            Real2 vi = v[itl->first];
            for (LElementConstIterator ite = itl->second.begin(), iteend=itl->second.end(); ite!=iteend; ++ite)
                res[ite->first] += (Real2)ite->second * vi;
        }
    }

    template<class Real2>
    void mul(FullVector<Real2>& res, const linearalgebra::BaseVector* v) const
    {
        res.resize(rowSize());
        for (LineConstIterator itl = begin(), itlend=end(); itl!=itlend; ++itl)
        {
            Real2 r = 0;
            for (LElementConstIterator ite = itl->second.begin(), iteend=itl->second.end(); ite!=iteend; ++ite)
                r += (Real2)ite->second * (Real2)v->element(ite->first);
            res[itl->first] = r;
        }
    }

    template<class Real2>
    void addMulTranspose(FullVector<Real2>& res, const linearalgebra::BaseVector* v) const
    {
        res.resize(colSize());
        for (LineConstIterator itl = begin(), itlend=end(); itl!=itlend; ++itl)
        {
            Real2 vi = v->element(itl->first);
            for (LElementConstIterator ite = itl->second.begin(), iteend=itl->second.end(); ite!=iteend; ++ite)
                res[ite->first] += (Real2)ite->second * vi;
        }
    }

    template<class Real2>
    void mul(linearalgebra::BaseVector* res, const FullVector<Real2>& v) const
    {
        res->resize(rowSize());
        for (LineConstIterator itl = begin(), itlend=end(); itl!=itlend; ++itl)
        {
            Real2 r = 0;
            for (LElementConstIterator ite = itl->second.begin(), iteend=itl->second.end(); ite!=iteend; ++ite)
                r += (Real2)ite->second * v[ite->first];
            res->set(itl->first, r);
        }
    }

    template<class Real2>
    void addMulTranspose(linearalgebra::BaseVector* res, const FullVector<Real2>& v) const
    {
        res->resize(colSize());
        for (LineConstIterator itl = begin(), itlend=end(); itl!=itlend; ++itl)
        {
            Real2 vi = v[itl->first];
            for (LElementConstIterator ite = itl->second.begin(), iteend=itl->second.end(); ite!=iteend; ++ite)
                res->add(ite->first, ite->second * vi);
        }
    }

    void mul(linearalgebra::BaseVector* res, const linearalgebra::BaseVector* v) const
    {
        res->resize(rowSize());
        for (LineConstIterator itl = begin(), itlend=end(); itl!=itlend; ++itl)
        {
            Real r = 0;
            for (LElementConstIterator ite = itl->second.begin(), iteend=itl->second.end(); ite!=iteend; ++ite)
                r += ite->second * (Real)v->element(ite->first);
            res->set(itl->first, r);
        }
    }

    void addMulTranspose(linearalgebra::BaseVector* res, const linearalgebra::BaseVector* v) const
    {
        res->resize(colSize());
        for (LineConstIterator itl = begin(), itlend=end(); itl!=itlend; ++itl)
        {
            Real vi = (Real)v->element(itl->first);
            for (LElementConstIterator ite = itl->second.begin(), iteend=itl->second.end(); ite!=iteend; ++ite)
                res->add(ite->first, ite->second * vi);
        }
    }

    template<class Real2>
    FullVector<Real2> operator*(const FullVector<Real2>& v) const
    {
        FullVector<Real2> res;
        mul(res,v);
        return res;
    }

    MatrixExpr< MatrixTranspose< SparseMatrix<T> > > t() const
    {
        return MatrixExpr { MatrixTranspose< SparseMatrix<T> >(*this) };
    }

    MatrixExpr< MatrixNegative< SparseMatrix<T> > > operator-() const
    {
        return MatrixExpr { MatrixNegative< SparseMatrix<T> >(*this) };
    }

    template<class Real2>
    MatrixExpr< MatrixProduct< SparseMatrix<T>, SparseMatrix<Real2> > > operator*(const SparseMatrix<Real2>& m) const
    {
        return MatrixExpr { MatrixProduct< SparseMatrix<T>, SparseMatrix<Real2> >(*this, m) };
    }

    MatrixExpr< MatrixScale< SparseMatrix<T>, double > > operator*(const double& r) const
    {
        return MatrixExpr { MatrixScale< SparseMatrix<T>, double >(*this, r) };
    }

    template<class Real2>
    MatrixExpr< MatrixAddition< SparseMatrix<T>, SparseMatrix<Real2> > > operator+(const SparseMatrix<Real2>& m) const
    {
        return MatrixExpr { MatrixAddition< SparseMatrix<T>, SparseMatrix<Real2> >(*this, m) };
    }

    template<class Real2>
    MatrixExpr< MatrixAddition< SparseMatrix<T>, SparseMatrix<Real2> > > operator-(const SparseMatrix<Real2>& m) const
    {
        return MatrixExpr { MatrixAddition< SparseMatrix<T>, SparseMatrix<Real2> >(*this, m) };
    }

    void swap(SparseMatrix<T>& m)
    {
        data.swap(m.data);
        Index t;
        t = nRow; nRow = m.nRow; m.nRow = t;
        t = nCol; nCol = m.nCol; m.nCol = t;
    }

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
        for (LineConstIterator itl = begin(), itlend=end(); itl!=itlend; ++itl)
        {
            const Index l = itl->first;
            for (LElementConstIterator ite = itl->second.begin(), iteend=itl->second.end(); ite!=iteend; ++ite)
            {
                const Index c = ite->first;
                Real v = ite->second;
                dest->add(l,c,v);
            }
        }
    }

protected:

    template<class M>
    void equal(const M& m, bool add = false)
    {
        if (m.hasRef(this))
        {
            SparseMatrix<T> tmp;
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
    void operator=(const SparseMatrix<Real2>& m)
    {
        if (&m == this) return;
        resize(m.rowSize(), m.colSize());
        m.addTo(this);
    }

    template<class Real2>
    void operator+=(const SparseMatrix<Real2>& m)
    {
        addEqual(m);
    }

    template<class Real2>
    void operator-=(const SparseMatrix<Real2>& m)
    {
        equal(MatrixExpr< MatrixNegative< SparseMatrix<Real2> > >(MatrixNegative< SparseMatrix<Real2> >(m)), true);
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

    static const char* Name();
};

template<> inline const char* SparseMatrix<double>::Name() { return "SparseMatrix"; }
template<> inline const char* SparseMatrix<float>::Name() { return "SparseMatrixf"; }


template<class R1, class R2>
class MatrixProductOp<SparseMatrix<R1>, SparseMatrix<R2> >
{
public:
    typedef SparseMatrix<R1> M1;
    typedef SparseMatrix<R2> M2;
    typedef typename type_selector<(sizeof(R2)>sizeof(R1)),M1,M2>::T matrix_type;
    enum { category = matrix_type::category };

    template<class Dest>
    void operator()(const SparseMatrix<R1>& m1, const SparseMatrix<R2>& m2, Dest* d)
    {
        for (typename SparseMatrix<R1>::LineConstIterator itl = m1.begin(), itlend=m1.end(); itl!=itlend; ++itl)
        {
            const int l1 = itl->first;
            for (typename SparseMatrix<R1>::LElementConstIterator ite = itl->second.begin(), iteend=itl->second.end(); ite!=iteend; ++ite)
            {
                const int c1 = ite->first;
                R1 v = ite->second;
                const typename SparseMatrix<R2>::Line& m2l = m2[c1];
                for (typename SparseMatrix<R2>::LElementConstIterator ite2 = m2l.begin(), ite2end=m2l.end(); ite2!=ite2end; ++ite2)
                {
                    R2 v2 = ite2->second;
                    const int c2 = ite2->first;
                    d->add(l1, c2, (v*v2));
                }
            }
        }
    }
};



} // namespace sofa::linearalgebra
