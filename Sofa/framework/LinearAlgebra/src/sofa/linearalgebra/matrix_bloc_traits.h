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

#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/linearalgebra/BaseMatrix.h>

namespace sofa::linearalgebra
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

// by default, supposing T is a type::Mat (useful for type derivated from type::Mat)
template<class T, typename IndexType>
class matrix_bloc_traits
{
public:
    typedef T Block;
    typedef T BlockTranspose;

    typedef typename T::Real Real;
    enum { NL = T::nbLines };
    enum { NC = T::nbCols };
    static Real& v(Block& b, IndexType row, IndexType col) { return b[row][col]; }
    static const Real& v(const Block& b, IndexType row, IndexType col) { return b[row][col]; }
    static void vset(Block& b, int row, int col, Real val) { b[row][col] = val; }
    static void vadd(Block& b, int row, int col, Real val) { b[row][col] += val; }
    static void clear(Block& b) { b.clear(); }
    static bool empty(const Block& b)
    {
        for (IndexType i=0; i<NL; ++i)
            for (IndexType j=0; j<NC; ++j)
                if (b[i][j] != 0) return false;
        return true;
    }
    static void invert(Block& result, const Block& b) { result.invert(b); }

    static void split_row_index(IndexType& index, IndexType& modulo) { bloc_index_func<NL, IndexType>::split(index, modulo); }
    static void split_col_index(IndexType& index, IndexType& modulo) { bloc_index_func<NC, IndexType>::split(index, modulo); }

    static sofa::linearalgebra::BaseMatrix::ElementType getElementType() { return matrix_bloc_traits<Real, IndexType>::getElementType(); }
};

template <Size L, Size C, class real, typename IndexType>
class matrix_bloc_traits < type::Mat<L,C,real>, IndexType>
{
public:
    typedef type::Mat<L, C, real> Block;
    typedef type::Mat<C, L, real> BlockTranspose;

    typedef real Real;
    enum { NL = L };
    enum { NC = C };
    static Real& v(Block& b, Index row, Index col) { return b[row][col]; }
    static const Real& v(const Block& b, Index row, Index col) { return b[row][col]; }
    static void vset(Block& b, int row, int col, Real val) { b[row][col] = val; }
    static void vadd(Block& b, int row, int col, Real val) { b[row][col] += val; }
    static void clear(Block& b) { b.clear(); }
    static bool empty(const Block& b)
    {
        for (Index i=0; i<NL; ++i)
            for (Index j=0; j<NC; ++j)
                if (b[i][j] != 0) return false;
        return true;
    }
    static void invert(Block& result, const Block& b)
    {
        const bool canInvert = result.invert(b);
        assert(canInvert);
        SOFA_UNUSED(canInvert);
    }

    static void split_row_index(IndexType& index, IndexType& modulo) { bloc_index_func<NL, IndexType>::split(index, modulo); }
    static void split_col_index(IndexType& index, IndexType& modulo) { bloc_index_func<NC, IndexType>::split(index, modulo); }

    static sofa::linearalgebra::BaseMatrix::ElementType getElementType() { return matrix_bloc_traits<Real, IndexType>::getElementType(); }
    static const std::string Name()
    {
        std::ostringstream o;
        o << "Mat" << L << "x" << C;
        if constexpr (std::is_same_v<float, real>)
        {
            o << "f";
        }
        if constexpr (std::is_same_v<double, real>)
        {
            o << "d";
        }
        if constexpr (std::is_same_v<int, real>)
        {
            o << "i";
        }

        return o.str();
    }
};

template <Size N, class T, typename IndexType >
class matrix_bloc_traits < sofa::type::Vec<N, T>, IndexType >
{
public:
    typedef sofa::type::Vec<N, T> Block;
    typedef T Real;
    typedef Block BlockTranspose;

    enum { NL = 1 };
    enum { NC = N };

    static Real& v(Block& b, int /*row*/, int col) { return b[col]; }
    static const Real& v(const Block& b, int /*row*/, int col) { return b[col]; }
    static void vset(Block& b, int /*row*/, int col, Real v) { b[col] = v; }
    static void vadd(Block& b, int /*row*/, int col, Real v) { b[col] += v; }
    static void clear(Block& b) { b.clear(); }
    static bool empty(const Block& b)
    {
        for (int i = 0; i < NC; ++i)
            if (b[i] != 0) return false;
        return true;
    }

    static Block transposed(const Block& b) { return b; }

    static void transpose(Block& res, const Block& b) { res = b; }

    static sofa::linearalgebra::BaseMatrix::ElementType getElementType() { return matrix_bloc_traits<Real, IndexType>::getElementType(); }
    static const std::string Name()
    {
        std::ostringstream o;
        o << "V" << N;
        if constexpr (std::is_same_v<float, Real>)
        {
            o << "f";
        }
        if constexpr (std::is_same_v<double, Real>)
        {
            o << "d";
        }
        if constexpr (std::is_same_v<int, Real>)
        {
            o << "i";
        }

        return o.str();
    }
};

//template<> inline const char* matrix_bloc_traits<type::Mat<1,1,float >, sofa::SignedIndex >::Name() { return "1f"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<1,1,double>, sofa::SignedIndex  >::Name() { return "1d"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<2,2,float >, sofa::SignedIndex >::Name() { return "2f"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<2,2,double>, sofa::SignedIndex >::Name() { return "2d"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<3,3,float >, sofa::SignedIndex >::Name() { return "3f"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<3,3,double>, sofa::SignedIndex >::Name() { return "3d"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<4,4,float >, sofa::SignedIndex >::Name() { return "4f"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<4,4,double>, sofa::SignedIndex >::Name() { return "4d"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<6,6,float >, sofa::SignedIndex >::Name() { return "6f"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<6,6,double>, sofa::SignedIndex >::Name() { return "6d"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<8,8,float >, sofa::SignedIndex >::Name() { return "8f"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<8,8,double>, sofa::SignedIndex >::Name() { return "8d"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<9,9,float >, sofa::SignedIndex >::Name() { return "9f"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<9,9,double>, sofa::SignedIndex >::Name() { return "9d"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<12,12,float >, sofa::SignedIndex >::Name() { return "12f"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<12,12,double>, sofa::SignedIndex >::Name() { return "12d"; }

//template<> inline const char* matrix_bloc_traits<type::Mat<1,1,float >, Size >::Name() { return "1f"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<1,1,double>, Size >::Name() { return "1d"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<2,2,float >, Size >::Name() { return "2f"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<2,2,double>, Size >::Name() { return "2d"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<3,3,float >, Size >::Name() { return "3f"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<3,3,double>, Size >::Name() { return "3d"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<4,4,float >, Size >::Name() { return "4f"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<4,4,double>, Size >::Name() { return "4d"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<6,6,float >, Size >::Name() { return "6f"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<6,6,double>, Size >::Name() { return "6d"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<8,8,float >, Size >::Name() { return "8f"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<8,8,double>, Size >::Name() { return "8d"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<9,9,float >, Size >::Name() { return "9f"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<9,9,double>, Size >::Name() { return "9d"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<12,12,float >, Size >::Name() { return "12f"; }
//template<> inline const char* matrix_bloc_traits<type::Mat<12,12,double>, Size >::Name() { return "12d"; }

template <typename IndexType>
class matrix_bloc_traits < float, IndexType >
{
public:
    typedef float Block;
    typedef Block BlockTranspose;

    typedef float Real;
    enum { NL = 1 };
    enum { NC = 1 };
    static Real& v(Block& b, IndexType, IndexType) { return b; }
    static const Real& v(const Block& b, IndexType, IndexType) { return b; }
    static void vset(Block& b, int, int, Real val) { b = val; }
    static void vadd(Block& b, int, int, Real val) { b += val; }
    static void clear(Block& b) { b = 0; }
    static bool empty(const Block& b)
    {
        return b == 0;
    }
    static void invert(Block& result, const Block& b) { result = 1.0f/b; }

    static void split_row_index(IndexType& index, IndexType& modulo) { bloc_index_func<NL, IndexType>::split(index, modulo); }
    static void split_col_index(IndexType& index, IndexType& modulo) { bloc_index_func<NC, IndexType>::split(index, modulo); }

    static const std::string Name() { return "f"; }
    static sofa::linearalgebra::BaseMatrix::ElementType getElementType() { return sofa::linearalgebra::BaseMatrix::ELEMENT_FLOAT; }
    static IndexType getElementSize() { return sizeof(Real); }
};

template <typename IndexType>
class matrix_bloc_traits < double, IndexType >
{
public:
    typedef double Block;
    typedef Block BlockTranspose;

    typedef double Real;
    enum { NL = 1 };
    enum { NC = 1 };
    static Real& v(Block& b, IndexType, IndexType) { return b; }
    static void vset(Block& b, int, int, Real val) { b = val; }
    static void vadd(Block& b, int, int, Real val) { b += val; }
    static const Real& v(const Block& b, IndexType, IndexType) { return b; }
    static void clear(Block& b) { b = 0; }
    static bool empty(const Block& b)
    {
        return b == 0;
    }
    static void invert(Block& result, const Block& b) { result = 1.0/b; }

    static void split_row_index(IndexType& index, IndexType& modulo) { bloc_index_func<NL, IndexType>::split(index, modulo); }
    static void split_col_index(IndexType& index, IndexType& modulo) { bloc_index_func<NC, IndexType>::split(index, modulo); }

    static sofa::linearalgebra::BaseMatrix::ElementType getElementType() { return sofa::linearalgebra::BaseMatrix::ELEMENT_FLOAT; }
    static const std::string Name() { return "d"; }
};

template <typename IndexType>
class matrix_bloc_traits < int, IndexType >
{
public:
    typedef float Block;
    typedef Block BlockTranspose;

    typedef float Real;
    enum { NL = 1 };
    enum { NC = 1 };
    static Real& v(Block& b, int, int) { return b; }
    static const Real& v(const Block& b, int, int) { return b; }
    static void vset(Block& b, int, int, Real val) { b = val; }
    static void vadd(Block& b, int, int, Real val) { b += val; }
    static void clear(Block& b) { b = 0; }
    static bool empty(const Block& b)
    {
        return b == 0;
    }
    static void invert(Block& result, const Block& b) { result = 1.0f/b; }

    static void split_row_index(int& index, int& modulo) { bloc_index_func<NL, IndexType>::split(index, modulo); }
    static void split_col_index(int& index, int& modulo) { bloc_index_func<NC, IndexType>::split(index, modulo); }

    static sofa::linearalgebra::BaseMatrix::ElementType getElementType() { return sofa::linearalgebra::BaseMatrix::ELEMENT_INT; }
    static const std::string Name() { return "f"; }
};


} // namespace sofa::linearalgebra
