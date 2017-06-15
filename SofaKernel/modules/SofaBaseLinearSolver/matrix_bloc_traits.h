/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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

template<int TN> class bloc_index_func
{
public:
    enum { N = TN };
    static void split(int& index, int& modulo)
    {
        modulo = index % N;
        index  = index / N;
    }
};

template<> class bloc_index_func<1>
{
public:
    enum { N = 1 };
    static void split(int&, int&)
    {
    }
};

template<> class bloc_index_func<2>
{
public:
    enum { N = 2 };
    static void split(int& index, int& modulo)
    {
        modulo = index & 1;
        index  = index >> 1;
    }
};

template<> class bloc_index_func<4>
{
public:
    enum { N = 2 };
    static void split(int& index, int& modulo)
    {
        modulo = index & 3;
        index  = index >> 2;
    }
};

template<> class bloc_index_func<8>
{
public:
    enum { N = 2 };
    static void split(int& index, int& modulo)
    {
        modulo = index & 7;
        index  = index >> 3;
    }
};


// by default, supposing T is a defaulttype::Mat (useful for type derivated from defaulttype::Mat)
template<class T>
class matrix_bloc_traits
{
public:
    typedef T Bloc;
    typedef typename T::Real Real;
    enum { NL = T::nbLines };
    enum { NC = T::nbCols };
    static Real& v(Bloc& b, int row, int col) { return b[row][col]; }
    static const Real& v(const Bloc& b, int row, int col) { return b[row][col]; }
    static void clear(Bloc& b) { b.clear(); }
    static bool empty(const Bloc& b)
    {
        for (int i=0; i<NL; ++i)
            for (int j=0; j<NC; ++j)
                if (b[i][j] != 0) return false;
        return true;
    }
    static void invert(Bloc& result, const Bloc& b) { result.invert(b); }

    static void split_row_index(int& index, int& modulo) { bloc_index_func<NL>::split(index, modulo); }
    static void split_col_index(int& index, int& modulo) { bloc_index_func<NC>::split(index, modulo); }

    static sofa::defaulttype::BaseMatrix::ElementType getElementType() { return matrix_bloc_traits<Real>::getElementType(); }
    static const char* Name();
};

template <int L, int C, class real>
class matrix_bloc_traits < defaulttype::Mat<L,C,real> >
{
public:
    typedef defaulttype::Mat<L,C,real> Bloc;
    typedef real Real;
    enum { NL = L };
    enum { NC = C };
    static Real& v(Bloc& b, int row, int col) { return b[row][col]; }
    static const Real& v(const Bloc& b, int row, int col) { return b[row][col]; }
    static void clear(Bloc& b) { b.clear(); }
    static bool empty(const Bloc& b)
    {
        for (int i=0; i<NL; ++i)
            for (int j=0; j<NC; ++j)
                if (b[i][j] != 0) return false;
        return true;
    }
    static void invert(Bloc& result, const Bloc& b) { result.invert(b); }

    static void split_row_index(int& index, int& modulo) { bloc_index_func<NL>::split(index, modulo); }
    static void split_col_index(int& index, int& modulo) { bloc_index_func<NC>::split(index, modulo); }

    static sofa::defaulttype::BaseMatrix::ElementType getElementType() { return matrix_bloc_traits<Real>::getElementType(); }
    static const char* Name();
};

template<> inline const char* matrix_bloc_traits<defaulttype::Mat<1,1,float > >::Name() { return "1f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<1,1,double> >::Name() { return "1d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<2,2,float > >::Name() { return "2f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<2,2,double> >::Name() { return "2d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<3,3,float > >::Name() { return "3f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<3,3,double> >::Name() { return "3d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<4,4,float > >::Name() { return "4f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<4,4,double> >::Name() { return "4d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<6,6,float > >::Name() { return "6f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<6,6,double> >::Name() { return "6d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<8,8,float > >::Name() { return "8f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<8,8,double> >::Name() { return "8d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<9,9,float > >::Name() { return "9f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<9,9,double> >::Name() { return "9d"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<12,12,float > >::Name() { return "12f"; }
template<> inline const char* matrix_bloc_traits<defaulttype::Mat<12,12,double> >::Name() { return "12d"; }

template <>
class matrix_bloc_traits < float >
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

    static void split_row_index(int& index, int& modulo) { bloc_index_func<NL>::split(index, modulo); }
    static void split_col_index(int& index, int& modulo) { bloc_index_func<NC>::split(index, modulo); }

    static const char* Name() { return "f"; }
    static sofa::defaulttype::BaseMatrix::ElementType getElementType() { return sofa::defaulttype::BaseMatrix::ELEMENT_FLOAT; }
    static std::size_t getElementSize() { return sizeof(Real); }
};

template <>
class matrix_bloc_traits < double >
{
public:
    typedef double Bloc;
    typedef double Real;
    enum { NL = 1 };
    enum { NC = 1 };
    static Real& v(Bloc& b, int, int) { return b; }
    static const Real& v(const Bloc& b, int, int) { return b; }
    static void clear(Bloc& b) { b = 0; }
    static bool empty(const Bloc& b)
    {
        return b == 0;
    }
    static void invert(Bloc& result, const Bloc& b) { result = 1.0/b; }

    static void split_row_index(int& index, int& modulo) { bloc_index_func<NL>::split(index, modulo); }
    static void split_col_index(int& index, int& modulo) { bloc_index_func<NC>::split(index, modulo); }

    static sofa::defaulttype::BaseMatrix::ElementType getElementType() { return sofa::defaulttype::BaseMatrix::ELEMENT_FLOAT; }
    static const char* Name() { return "d"; }
};

template <>
class matrix_bloc_traits < int >
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

    static void split_row_index(int& index, int& modulo) { bloc_index_func<NL>::split(index, modulo); }
    static void split_col_index(int& index, int& modulo) { bloc_index_func<NC>::split(index, modulo); }

    static sofa::defaulttype::BaseMatrix::ElementType getElementType() { return sofa::defaulttype::BaseMatrix::ELEMENT_INT; }
    static const char* Name() { return "f"; }
};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
