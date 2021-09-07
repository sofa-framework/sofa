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

namespace sofa::linearalgebra
{

/// Simple full matrix container
template<typename T>
class SOFA_LINEARALGEBRA_API FullMatrix : public linearalgebra::BaseMatrix
{
public:
    typedef T Real;
    typedef typename linearalgebra::BaseMatrix::Index Index;
    typedef FullVector<Real> Line;

    class LineConstIterator
    {
    public:
        Index first;
        Line second;
        LineConstIterator(Real* p, Index i, Index size, Index pitch) : first(i), second(p+i*pitch,size,pitch) { }
        const Line& operator*() const { return second; }
        const Line* operator->() const { return &second; }
        void operator++() { ++first; second.setptr(second.ptr() + second.capacity()); }
        void operator++(int) { ++first; second.setptr(second.ptr() + second.capacity()); }
        void operator--() { --first; second.setptr(second.ptr() - second.capacity()); }
        void operator--(int) { --first; second.setptr(second.ptr() - second.capacity()); }
        bool operator==(const LineConstIterator& i) const { return i.second.ptr() == second.ptr(); }
        bool operator!=(const LineConstIterator& i) const { return i.second.ptr() != second.ptr(); }
    };
    class LineIterator : public LineConstIterator
    {
    public:
        LineIterator(Real* p, Index i, Index size, Index pitch) : LineConstIterator(p,i,size,pitch) {}
        Line& operator*() { return this->second; }
        Line* operator->() { return &this->second; }
    };
    typedef typename Line::iterator LElementIterator;
    typedef typename Line::const_iterator LElementConstIterator;

protected:
    Real* data;
    Index nRow,nCol;
    Index pitch;
    Index allocsize;

public:

    FullMatrix();
    FullMatrix(Index nbRow, Index nbCol);
    FullMatrix(Real* p, Index nbRow, Index nbCol);
    FullMatrix(Real* p, Index nbRow, Index nbCol, Index pitch);

    ~FullMatrix() override;

    Real* ptr() { return data; }
    const Real* ptr() const { return data; }

    LineIterator begin();
    LineIterator end();
    LineConstIterator begin() const;
    LineConstIterator end() const;

    Real* operator[](Index i) { return data+i*pitch; }
    const Real* operator[](Index i) const { return data+i*pitch; }

    void resize(Index nbRow, Index nbCol) override;

    Index rowSize(void) const override { return nRow; }
    Index colSize(void) const override { return nCol; }

    SReal element(Index i, Index j) const override;
    void set(Index i, Index j, double v) override;
    using BaseMatrix::add;
    void add(Index i, Index j, double v) override;
    void clear(Index i, Index j) override;

    void clearRow(Index i) override;
    void clearCol(Index j) override;
    void clearRowCol(Index i) override;
    void clear() override;

    /// matrix-vector product
    /// @returns this * v
    FullVector<Real> operator*( const FullVector<Real>& v ) const ;

    /// matrix-vector product
    /// res = this * v
    void mul( FullVector<Real>& res,const FullVector<Real>& b ) const ;

    /// transposed matrix-vector product
    /// res = this^T * v
    void mulT( FullVector<Real>& res, const FullVector<Real>& b ) const;

    /// matrix multiplication
    /// @returns this * m
    FullMatrix<Real> operator*( const FullMatrix<Real>& m ) const;

    /// matrix multiplication
    /// res = this * m
    void mul( FullMatrix<Real>& res, const FullMatrix<Real>& m ) const;

    /// transposed matrix multiplication
    /// res = this^T * m
    void mulT( FullMatrix<Real>& res, const FullMatrix<Real>& m ) const;


    static const char* Name();
};

/// Simple full matrix container, with an additionnal pointer per line, to be able do get a T** pointer and use [i][j] directly
template<typename T>
class SOFA_LINEARALGEBRA_API LPtrFullMatrix : public FullMatrix<T>
{
public:
    typedef typename FullMatrix<T>::Index Index;
protected:
    T** ldata;
    Index lallocsize;
public:
    LPtrFullMatrix();

    ~LPtrFullMatrix() override;
    void resize(Index nbRow, Index nbCol) override;

    T** lptr() { return ldata; }
};

template<> const char* FullMatrix<double>::Name();
template<> const char* FullMatrix<float>::Name();

SOFA_LINEARALGEBRA_API std::ostream& operator << (std::ostream& out, const FullMatrix<double>& v );
SOFA_LINEARALGEBRA_API std::ostream& operator << (std::ostream& out, const FullMatrix<float>& v );

SOFA_LINEARALGEBRA_API std::ostream& operator << (std::ostream& out, const LPtrFullMatrix<double>& v );
SOFA_LINEARALGEBRA_API std::ostream& operator << (std::ostream& out, const LPtrFullMatrix<float>& v );

#if !defined(SOFABASELINEARSOLVER_FULLMATRIX_DEFINITION)
extern template class SOFA_LINEARALGEBRA_API FullMatrix<double>;
extern template class SOFA_LINEARALGEBRA_API FullMatrix<float>;

extern template class SOFA_LINEARALGEBRA_API LPtrFullMatrix<double>;
extern template class SOFA_LINEARALGEBRA_API LPtrFullMatrix<float>;
#endif /// SOFABASELINEARSOLVER_FULLMATRIX_DEFINITION

} // namespace sofa::linearalgebra
