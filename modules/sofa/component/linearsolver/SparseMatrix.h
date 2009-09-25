/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_SPARSEMATRIX_H
#define SOFA_COMPONENT_LINEARSOLVER_SPARSEMATRIX_H

#include <sofa/defaulttype/BaseMatrix.h>
#include "FullVector.h"

#include <map>

namespace sofa
{

namespace component
{

namespace linearsolver
{

//#define SPARSEMATRIX_CHECK
//#define SPARSEMATRIX_VERBOSE

template<typename T>
class SparseMatrix : public defaulttype::BaseMatrix
{
public:
    typedef T Real;
    typedef int Index;
    typedef std::map<Index,Real> Line;
    typedef std::map<Index,Line> Data;
    typedef typename Line::iterator LElementIterator;
    typedef typename Line::const_iterator LElementConstIterator;
    typedef typename Data::iterator LineIterator;
    typedef typename Data::const_iterator LineConstIterator;

protected:
    Data data;
    Index nRow,nCol;

public:

    SparseMatrix()
        : nRow(0), nCol(0)
    {
    }

    SparseMatrix(int nbRow, int nbCol)
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

    void resize(int nbRow, int nbCol)
    {

#ifdef SPARSEMATRIX_VERBOSE
        if (nbRow != rowSize() || nbCol != colSize())
            std::cout << /* this->Name()  <<  */": resize("<<nbRow<<","<<nbCol<<")"<<std::endl;
#endif
        data.clear();
        nRow = nbRow;
        nCol = nbCol;
    }

    unsigned int rowSize(void) const
    {
        return nRow;
    }

    unsigned int colSize(void) const
    {
        return nCol;
    }

    SReal element(int i, int j) const
    {
#ifdef SPARSEMATRIX_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            serr << "ERROR: invalid read access to element ("<<i<<","<<j<<") in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return 0.0;
        }
#endif
        LineConstIterator it = data.find(i);
        if (it==data.end())
            return 0.0;
        LElementConstIterator ite = it->second.find(j);
        if (ite == it->second.end())
            return 0.0;
        return ite->second;
    }

    void set(int i, int j, double v)
    {
#ifdef SPARSEMATRIX_VERBOSE
        std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") = "<<v<<std::endl;
#endif
#ifdef SPARSEMATRIX_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            serr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        data[i][j] = (Real)v;
    }

    void add(int i, int j, double v)
    {
#ifdef SPARSEMATRIX_VERBOSE
        std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") += "<<v<<std::endl;
#endif
#ifdef SPARSEMATRIX_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            serr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        data[i][j] += (Real)v;
    }

    void clear(int i, int j)
    {
#ifdef SPARSEMATRIX_VERBOSE
        std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): element("<<i<<","<<j<<") = 0"<<std::endl;
#endif
#ifdef SPARSEMATRIX_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)j >= (unsigned)colSize())
        {
            serr << "ERROR: invalid write access to element ("<<i<<","<<j<<") in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
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

    void clearRow(int i)
    {
#ifdef SPARSEMATRIX_VERBOSE
        std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): row("<<i<<") = 0"<<std::endl;
#endif
#ifdef SPARSEMATRIX_CHECK
        if ((unsigned)i >= (unsigned)rowSize())
        {
            serr << "ERROR: invalid write access to row "<<i<<" in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        LineIterator it = data.find(i);
        if (it==data.end())
            return;
        data.erase(it);
    }

    void clearCol(int j)
    {
#ifdef SPARSEMATRIX_VERBOSE
        std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): col("<<j<<") = 0"<<std::endl;
#endif
#ifdef SPARSEMATRIX_CHECK
        if ((unsigned)j >= (unsigned)colSize())
        {
            serr << "ERROR: invalid write access to column "<<j<<" in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        for(LineIterator it=data.begin(),itend=data.end(); it!=itend; ++it)
        {
            LElementIterator ite = it->second.find(j);
            if (ite != it->second.end())
                it->second.erase(ite);
        }
    }

    void clearRowCol(int i)
    {
#ifdef SPARSEMATRIX_VERBOSE
        std::cout << /* this->Name()  <<  */"("<<rowSize()<<","<<colSize()<<"): row("<<i<<") = 0 and col("<<i<<") = 0"<<std::endl;
#endif
#ifdef SPARSEMATRIX_CHECK
        if ((unsigned)i >= (unsigned)rowSize() || (unsigned)i >= (unsigned)colSize())
        {
            serr << "ERROR: invalid write access to row and column "<<i<<" in "<</* this->Name() <<*/" of size ("<<rowSize()<<","<<colSize()<<")"<<std::endl;
            return;
        }
#endif
        clearRow(i);
        clearCol(i);
    }

    void clear() { data.clear(); }

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
    void mulTranspose(FullVector<Real2>& res, const FullVector<Real2>& v) const
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
    void mul(FullVector<Real2>& res, const defaulttype::BaseVector* v) const
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
    void mulTranspose(FullVector<Real2>& res, const defaulttype::BaseVector* v) const
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
    void mul(defaulttype::BaseVector* res, const FullVector<Real2>& v) const
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
    void mulTranspose(defaulttype::BaseVector* res, const FullVector<Real2>& v) const
    {
        res->resize(colSize());
        for (LineConstIterator itl = begin(), itlend=end(); itl!=itlend; ++itl)
        {
            Real2 vi = v[itl->first];
            for (LElementConstIterator ite = itl->second.begin(), iteend=itl->second.end(); ite!=iteend; ++ite)
                res->add(ite->first, ite->second * vi);
        }
    }

    void mul(defaulttype::BaseVector* res, const defaulttype::BaseVector* v) const
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

    void mulTranspose(defaulttype::BaseVector* res, const defaulttype::BaseVector* v) const
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

    template<class Real2>
    void mul(SparseMatrix<T>* res, const SparseMatrix<Real2>& m) const
    {
        res->resize(rowSize(), m.colSize());
        for (LineConstIterator itl = begin(), itlend=end(); itl!=itlend; ++itl)
        {
            const int this_line = itl->first;
            for (LElementConstIterator ite = itl->second.begin(), iteend=itl->second.end(); ite!=iteend; ++ite)
            {
                Real v = ite->second;
                const typename SparseMatrix<Real2>::Line& ml = m[ite->first];
                for (typename SparseMatrix<Real2>::LElementConstIterator ite2 = ml.begin(), ite2end=ml.end(); ite2!=ite2end; ++ite2)
                {
                    Real2 v2 = ite2->second;
                    const int m_col = ite2->first;
                    res->add(this_line, m_col, (Real)(v*v2));
                }
            }
        }
    }

    template<class Real2>
    void addmul(SparseMatrix<T>* res, const SparseMatrix<Real2>& m) const
    {
        //res->resize(rowSize(), m.colSize());
        for (LineConstIterator itl = begin(), itlend=end(); itl!=itlend; ++itl)
        {
            const int this_line = itl->first;
            for (LElementConstIterator ite = itl->second.begin(), iteend=itl->second.end(); ite!=iteend; ++ite)
            {
                Real v = ite->second;
                const typename SparseMatrix<Real2>::Line& ml = m[ite->first];
                for (typename SparseMatrix<Real2>::LElementConstIterator ite2 = ml.begin(), ite2end=ml.end(); ite2!=ite2end; ++ite2)
                {
                    Real2 v2 = ite2->second;
                    const int m_col = ite2->first;
                    res->add(this_line, m_col, (Real)(v*v2));
                }
            }
        }
    }

    friend std::ostream& operator << (std::ostream& out, const SparseMatrix<T>& v )
    {
        int nx = v.colSize();
        int ny = v.rowSize();
        out << "[";
        for (int y=0; y<ny; ++y)
        {
            out << "\n[";
            for (int x=0; x<nx; ++x)
            {
                out << " " << v.element(y,x);
            }
            out << " ]";
        }
        out << " ]";
        return out;
    }

    static const char* Name() { return "SparseMatrix"; }
};

#ifdef SPARSEMATRIX_CHECK
#undef SPARSEMATRIX_CHECK
#endif
#ifdef SPARSEMATRIX_VERBOSE
#undef SPARSEMATRIX_VERBOSE
#endif

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
