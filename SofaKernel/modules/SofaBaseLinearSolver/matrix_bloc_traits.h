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
#ifndef SOFA_COMPONENT_LINEARSOLVER_MATRIXBLOCTRAITS_H
#define SOFA_COMPONENT_LINEARSOLVER_MATRIXBLOCTRAITS_H
#include "config.h"

#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/BaseMatrix.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

template<Size TN, typename T> class bloc_index_func
{
public:
    enum { N = TN };
    static void split(T& index, T& modulo)
    {
        if constexpr (N == 1)
        {
            ; // nothing
            return;
        }
        if constexpr (N == 2)
        {
            modulo = index & 1;
            index = index >> 1;
            return;
        }
        if constexpr (N == 4)
        {
            modulo = index & 3;
            index = index >> 2;
            return;
        }
        if constexpr (N == 8)
        {
            modulo = index & 7;
            index = index >> 3;
            return;
        }
        else
        {
            modulo = index % N;
            index = index / N;
            return;
        }
    }
};

// by default, supposing T is a defaulttype::Mat (useful for type derivated from defaulttype::Mat)
template<class T, typename IndexType>
class matrix_bloc_traits
{
public:
    typedef T Bloc;
    typedef typename T::Real Real;
    enum { NL = T::nbLines };
    enum { NC = T::nbCols };
    static Real& v(Bloc& b, IndexType row, IndexType col) { return b[row][col]; }
    static const Real& v(const Bloc& b, IndexType row, IndexType col) { return b[row][col]; }
    static void clear(Bloc& b) { b.clear(); }
    static bool empty(const Bloc& b)
    {
        for (IndexType i=0; i<NL; ++i)
            for (IndexType j=0; j<NC; ++j)
                if (b[i][j] != 0) return false;
        return true;
    }
    static void invert(Bloc& result, const Bloc& b) { result.invert(b); }

    static void split_row_index(IndexType& index, IndexType& modulo) { bloc_index_func<NL, IndexType>::split(index, modulo); }
    static void split_col_index(IndexType& index, IndexType& modulo) { bloc_index_func<NC, IndexType>::split(index, modulo); }

    static sofa::defaulttype::BaseMatrix::ElementType getElementType() { return matrix_bloc_traits<Real, IndexType>::getElementType(); }
    static const char* Name();
};

template <Size L, Size C, class real, typename IndexType>
class matrix_bloc_traits < defaulttype::Mat<L,C,real>, IndexType>
{
public:
    typedef defaulttype::Mat<L,C,real> Bloc;
    typedef real Real;
    enum { NL = L };
    enum { NC = C };
    static Real& v(Bloc& b, Index row, Index col) { return b[row][col]; }
    static const Real& v(const Bloc& b, Index row, Index col) { return b[row][col]; }
    static void clear(Bloc& b) { b.clear(); }
    static bool empty(const Bloc& b)
    {
        for (Index i=0; i<NL; ++i)
            for (Index j=0; j<NC; ++j)
                if (b[i][j] != 0) return false;
        return true;
    }
    static void invert(Bloc& result, const Bloc& b) { result.invert(b); }

    static void split_row_index(IndexType& index, IndexType& modulo) { bloc_index_func<NL, IndexType>::split(index, modulo); }
    static void split_col_index(IndexType& index, IndexType& modulo) { bloc_index_func<NC, IndexType>::split(index, modulo); }

    static sofa::defaulttype::BaseMatrix::ElementType getElementType() { return matrix_bloc_traits<Real, IndexType>::getElementType(); }
    static const char* Name();
};

template<> inline const char* matrix_bloc_traits<defaulttype::Mat<1,1,float >, int >::Name() { return "1f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<1,1,double>, int >::Name() { return "1d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<2,2,float >, int >::Name() { return "2f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<2,2,double>, int >::Name() { return "2d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<3,3,float >, int >::Name() { return "3f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<3,3,double>, int >::Name() { return "3d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<4,4,float >, int >::Name() { return "4f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<4,4,double>, int >::Name() { return "4d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<6,6,float >, int >::Name() { return "6f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<6,6,double>, int >::Name() { return "6d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<8,8,float >, int >::Name() { return "8f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<8,8,double>, int >::Name() { return "8d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<9,9,float >, int >::Name() { return "9f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<9,9,double>, int >::Name() { return "9d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<12,12,float >, int >::Name() { return "12f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<12,12,double>, int >::Name() { return "12d"; }

template<> inline const char* matrix_bloc_traits<defaulttype::Mat<1,1,float >, Size >::Name() { return "1f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<1,1,double>, Size >::Name() { return "1d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<2,2,float >, Size >::Name() { return "2f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<2,2,double>, Size >::Name() { return "2d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<3,3,float >, Size >::Name() { return "3f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<3,3,double>, Size >::Name() { return "3d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<4,4,float >, Size >::Name() { return "4f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<4,4,double>, Size >::Name() { return "4d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<6,6,float >, Size >::Name() { return "6f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<6,6,double>, Size >::Name() { return "6d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<8,8,float >, Size >::Name() { return "8f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<8,8,double>, Size >::Name() { return "8d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<9,9,float >, Size >::Name() { return "9f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<9,9,double>, Size >::Name() { return "9d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<12,12,float >, Size >::Name() { return "12f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<12,12,double>, Size >::Name() { return "12d"; }

template <typename IndexType>
class matrix_bloc_traits < float, IndexType >
{
public:
    typedef float Bloc;
    typedef float Real;
    enum { NL = 1 };
    enum { NC = 1 };
    static Real& v(Bloc& b, IndexType, IndexType) { return b; }
    static const Real& v(const Bloc& b, IndexType, IndexType) { return b; }
    static void clear(Bloc& b) { b = 0; }
    static bool empty(const Bloc& b)
    {
        return b == 0;
    }
    static void invert(Bloc& result, const Bloc& b) { result = 1.0f/b; }

    static void split_row_index(IndexType& index, IndexType& modulo) { bloc_index_func<NL, IndexType>::split(index, modulo); }
    static void split_col_index(IndexType& index, IndexType& modulo) { bloc_index_func<NC, IndexType>::split(index, modulo); }

    static const char* Name() { return "f"; }
    static sofa::defaulttype::BaseMatrix::ElementType getElementType() { return sofa::defaulttype::BaseMatrix::ELEMENT_FLOAT; }
    static IndexType getElementSize() { return sizeof(Real); }
};

template <typename IndexType>
class matrix_bloc_traits < double, IndexType >
{
public:
    typedef double Bloc;
    typedef double Real;
    enum { NL = 1 };
    enum { NC = 1 };
    static Real& v(Bloc& b, IndexType, IndexType) { return b; }
    static const Real& v(const Bloc& b, IndexType, IndexType) { return b; }
    static void clear(Bloc& b) { b = 0; }
    static bool empty(const Bloc& b)
    {
        return b == 0;
    }
    static void invert(Bloc& result, const Bloc& b) { result = 1.0/b; }

    static void split_row_index(IndexType& index, IndexType& modulo) { bloc_index_func<NL, IndexType>::split(index, modulo); }
    static void split_col_index(IndexType& index, IndexType& modulo) { bloc_index_func<NC, IndexType>::split(index, modulo); }

    static sofa::defaulttype::BaseMatrix::ElementType getElementType() { return sofa::defaulttype::BaseMatrix::ELEMENT_FLOAT; }
    static const char* Name() { return "d"; }
};

template <typename IndexType>
class matrix_bloc_traits < int, IndexType >
{
public:
    typedef float Bloc;
    typedef float Real;
    enum { NL = 1 };
    enum { NC = 1 };
    static Real& v(Bloc& b, int, int) { return b; }
    static const Real& v(const Bloc& b, int, int) { return b; }
    static void clear(Bloc& b) { b = 0; }
    static bool empty(const Bloc& b)
    {
        return b == 0;
    }
    static void invert(Bloc& result, const Bloc& b) { result = 1.0f/b; }

    static void split_row_index(int& index, int& modulo) { bloc_index_func<NL, IndexType>::split(index, modulo); }
    static void split_col_index(int& index, int& modulo) { bloc_index_func<NC, IndexType>::split(index, modulo); }

    static sofa::defaulttype::BaseMatrix::ElementType getElementType() { return sofa::defaulttype::BaseMatrix::ELEMENT_INT; }
    static const char* Name() { return "f"; }
};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
