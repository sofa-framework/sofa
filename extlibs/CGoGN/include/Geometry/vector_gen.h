/*******************************************************************************
 * CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
 * version 0.1                                                                  *
 * Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
 *                                                                              *
 * This library is free software; you can redistribute it and/or modify it      *
 * under the terms of the GNU Lesser General Public License as published by the *
 * Free Software Foundation; either version 2.1 of the License, or (at your     *
 * option) any later version.                                                   *
 *                                                                              *
 * This library is distributed in the hope that it will be useful, but WITHOUT  *
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
 * for more details.                                                            *
 *                                                                              *
 * You should have received a copy of the GNU Lesser General Public License     *
 * along with this library; if not, write to the Free Software Foundation,      *
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
 *                                                                              *
 * Web site: http://cgogn.unistra.fr/                                           *
 * Contact information: cgogn@unistra.fr                                        *
 *                                                                              *
 *******************************************************************************/

#ifndef __VECTOR_GEN__
#define __VECTOR_GEN__

#include <cassert>
#include <iostream>
#include <math.h>
#include <string.h>

#include "Utils/static_assert.h"
#include "Utils/nameTypes.h"
#include <sofa/defaulttype/Vec.h>


namespace CGoGN
{

namespace Geom
{

/*
 * Class for the representation of vectors
 */
template <unsigned int DIM, typename T>
struct Vector {
    typedef sofa::defaulttype::Vec<DIM,T> type;
};


/***
 * Test if x is null within precision.
 * Two cases are possible :
 *  - precision == 0 : x is null if (x == 0)
 *  - precision > 0 : x is null if (|x| < 1/precision) or (precision * |x| < 1)
 */
template <typename T>
bool isNull(T x, int precision = 0) ;

/***
 * Test if the square root of x is null within precision.
 * In other words, test if x is null within precision*precision
 */
template <typename T>
bool isNull2(T x, int precision = 0) ;

// template <unsigned int DIM, typename T>
// typename Vector<DIM, T>::type operator*(T a, const typename Vector<DIM, T>::type& v) ;

template <unsigned int DIM, typename T>
typename Vector<DIM, T>::type operator/(T a, const typename Vector<DIM, T>::type& v) ;

// returns the signed volume of the parallelepiped spanned by vectors v1, v2 and v3
template <unsigned int DIM, typename T>
T tripleProduct(const sofa::defaulttype::Vec<DIM, T>& v1, const sofa::defaulttype::Vec<DIM, T>& v2, const sofa::defaulttype::Vec<DIM, T>& v3) ;

// returns a spherical interpolation of two vectors considering parameter t ((0 <= t <= 1) => result between v1 and v2)
template <unsigned int DIM, typename T>
typename Vector<DIM, T>::type slerp(const typename Vector<DIM, T>::type &v1, const typename Vector<DIM, T>::type &v2, const T &t) ;

template <unsigned int DIM, typename T, typename T2>
typename Vector<DIM, T>::type operator*(T2 b, const typename Vector<DIM, T>::type& v);


/**********************************************/
/*           SOME USEFUL TYPEDEFS             */
/**********************************************/

//typedef Vector<2, float> Vec2f ;
//typedef Vector<2, double> Vec2d ;
//typedef Vector<2, unsigned int> Vec2ui ;
//typedef Vector<2, int> Vec2i ;
//typedef Vector<2, unsigned char> Vec2uc ;

//typedef Vector<3, float> Vec3f ;
//typedef Vector<3, double> Vec3d ;
//typedef Vector<3, unsigned int> Vec3ui ;
//typedef Vector<3, int> Vec3i ;
//typedef Vector<3, unsigned char> Vec3uc ;

//typedef Vector<4, float> Vec4f ;
//typedef Vector<4, double> Vec4d ;
//typedef Vector<4, unsigned int> Vec4ui ;
//typedef Vector<4, int> Vec4i ;
//typedef Vector<4, unsigned char> Vec4uc ;

typedef typename Vector<2, float>::type Vec2f ;
typedef typename Vector<2, double>::type Vec2d ;
typedef typename Vector<2, unsigned int>::type Vec2ui ;
typedef typename Vector<2, int>::type Vec2i ;
typedef typename Vector<2, unsigned char>::type Vec2uc ;

typedef typename Vector<3, float>::type Vec3f ;
typedef typename Vector<3, double>::type Vec3d ;
typedef typename Vector<3, unsigned int>::type Vec3ui ;
typedef typename Vector<3, int>::type Vec3i ;
typedef typename Vector<3, unsigned char>::type Vec3uc ;

typedef typename Vector<4, float>::type Vec4f ;
typedef typename Vector<4, double>::type Vec4d ;
typedef typename Vector<4, unsigned int>::type Vec4ui ;
typedef typename Vector<4, int>::type Vec4i ;
typedef typename Vector<4, unsigned char>::type Vec4uc ;

typedef typename Vector<6, float>::type Vec6f ;
typedef typename Vector<6, double>::type Vec6d ;
typedef typename Vector<6, unsigned int>::type Vec6ui ;
typedef typename Vector<6, int>::type Vec6i ;
typedef typename Vector<6, unsigned char>::type Vec6uc ;
}

}

#include "vector_gen.hpp"

#endif
