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

#ifndef __TRANSFO__H__
#define __TRANSFO__H__

#include "Geometry/matrix.h"
#include <cmath>

namespace CGoGN
{

namespace Geom
{

/**
 * Apply a scale to matrix
 * @param sx scale in x axis
 * @param sy scale in y axis
 * @param sz scale in z axis
 * @param mat current matrix
 */
template <typename T>
void scale(T sx, T sy, T sz, Matrix<4,4,T>& mat);

/**
 * Apply a translation to matrix
 * @param tx scale in x axis
 * @param ty scale in y axis
 * @param tz scale in z axis
 * @param mat current matrix
 */
template <typename T>
void translate(T tx, T ty, T tz, Matrix<4,4,T>& mat);

/**
 * Apply a rotation around Z axis to matrix
 * @param angle angle of rotation in radian
 * @param mat current matrix
 */
template <typename T>
void rotateZ(T angle, Matrix<4,4,T>& mat);

/**
 * Apply a rotation around Y axis to matrix
 * @param angle angle of rotation in radian
 * @param mat current matrix
 */
template <typename T>
void rotateY(T angle, Matrix<4,4,T>& mat);

/**
 * Apply a rotation around X axis to matrix
 * @param angle angle of rotation in radian
 * @param mat current matrix
 */
template <typename T>
void rotateX(T angle, Matrix<4,4,T>& mat);

/**
 * Apply a rotation around axis to matrix
 * @param axis_x axis x coord
 * @param axis_y axis y coord
 * @param axis_z axis z coord
 * @param angle angle of rotation in radian
 * @param mat current matrix
 */
template <typename T>
void rotate(T axis_x, T axis_y, T axis_z, T angle, Matrix<4,4,T>& mat);

template <typename T>
void rotate(typename Vector<3,T>::type& axis, T angle, Matrix<4,4,T>& mat);

/**
 * Apply a transformation (stored in matrix) to a 3D point
 * @param P the point to transfo
 * @param mat the transformation matrix
 */
template <typename T>
typename Vector<3,T>::type transform(const typename Vector<3,T>::type& P, const Matrix<4,4,T>& mat);

} // namespace Geom

} // namespace CGoGN

#include "Geometry/transfo.hpp"

#endif
