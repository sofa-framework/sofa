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

namespace CGoGN
{

namespace Geom
{

template <typename T>
void scale(T sx, T sy, T sz, Matrix<4,4,T>& mat)
{
	Matrix<4,4,T>  mat_scale;

	mat_scale(0,0) = sx;
	mat_scale(0,1) = T(0);
	mat_scale(0,2) = T(0);
	mat_scale(0,3) = T(0);

	mat_scale(1,0) = T(0);
	mat_scale(1,1) = sy;
	mat_scale(1,2) = T(0);
	mat_scale(1,3) = T(0);

	mat_scale(2,0) = T(0);
	mat_scale(2,1) = T(0);
	mat_scale(2,2) = sz;
	mat_scale(2,3) = T(0);

	mat_scale(3,0) = T(0);
	mat_scale(3,1) = T(0);
	mat_scale(3,2) = T(0);
	mat_scale(3,3) = T(1);

	mat = mat_scale * mat;
}


template <typename T>
void translate(T tx, T ty, T tz, Matrix<4,4,T>& mat)
{
	Matrix<4,4,T>  mat_trans;

	mat_trans(0,0) = T(1);
	mat_trans(0,1) = T(0);
	mat_trans(0,2) = T(0);
	mat_trans(0,3) = tx;

	mat_trans(1,0) = T(0);
	mat_trans(1,1) = T(1);
	mat_trans(1,2) = T(0);
	mat_trans(1,3) = ty;

	mat_trans(2,0) = T(0);
	mat_trans(2,1) = T(0);
	mat_trans(2,2) = T(1);
	mat_trans(2,3) = tz;

	mat_trans(3,0) = T(0);
	mat_trans(3,1) = T(0);
	mat_trans(3,2) = T(0);
	mat_trans(3,3) = T(1);

	mat = mat_trans * mat;
}

template <typename T>
void rotateZ(T angle, Matrix<4,4,T>& mat)
{
	Matrix<4,4,T>  mat_rot;

	T c = std::cos(angle);
	T s = std::sin(angle);

	mat_rot(0,0) = c;
	mat_rot(0,1) = -s;
	mat_rot(0,2) = T(0);
	mat_rot(0,3) = T(0);

	mat_rot(1,0) = s;
	mat_rot(1,1) = c;
	mat_rot(1,2) = T(0);
	mat_rot(1,3) = T(0);

	mat_rot(2,0) = T(0);
	mat_rot(2,1) = T(0);
	mat_rot(2,2) = T(1);
	mat_rot(2,3) = T(0);

	mat_rot(3,0) = T(0);
	mat_rot(3,1) = T(0);
	mat_rot(3,2) = T(0);
	mat_rot(3,3) = T(1);

	mat = mat_rot * mat;
}

template <typename T>
void rotateY(T angle, Matrix<4,4,T>& mat)
{
	Matrix<4,4,T>  mat_rot;

	T c = std::cos(angle);
	T s = std::sin(angle);

	mat_rot(0,0) = c;
	mat_rot(0,1) = T(0);
	mat_rot(0,2) = s;
	mat_rot(0,3) = T(0);

	mat_rot(1,0) = T(0);
	mat_rot(1,1) = T(1);
	mat_rot(1,2) = T(0);
	mat_rot(1,3) = T(0);

	mat_rot(2,0) = -s;
	mat_rot(2,1) = T(0);
	mat_rot(2,2) = c;
	mat_rot(2,3) = T(0);

	mat_rot(3,0) = T(0);
	mat_rot(3,1) = T(0);
	mat_rot(3,2) = T(0);
	mat_rot(3,3) = T(1);

	mat = mat_rot * mat;
}

template <typename T>
void rotateX(T angle, Matrix<4,4,T>& mat)
{
	Matrix<4,4,T>  mat_rot;

	T c = std::cos(angle);
	T s = std::sin(angle);

	mat_rot(0,0) = T(1);
	mat_rot(0,1) = T(0);
	mat_rot(0,2) = T(0);
	mat_rot(0,3) = T(0);

	mat_rot(1,0) = T(0);
	mat_rot(1,1) = c;
	mat_rot(1,2) = -s;
	mat_rot(1,3) = T(0);

	mat_rot(2,0) = T(0);
	mat_rot(2,1) = s;
	mat_rot(2,2) = c;
	mat_rot(2,3) = T(0);

	mat_rot(3,0) = T(0);
	mat_rot(3,1) = T(0);
	mat_rot(3,2) = T(0);
	mat_rot(3,3) = T(1);

	mat = mat_rot * mat;
}

template <typename T>
void rotate(T axis_x, T axis_y, T axis_z, T angle, Matrix<4,4,T>& mat)
{
	T length = T( std::sqrt(double(axis_x*axis_x + axis_y*axis_y + axis_z*axis_z)));
	axis_x /=length;
	axis_y /=length;
	axis_z /=length;

	T c = std::cos(angle);
	T s = std::sin(angle);

	T xx = axis_x * axis_x;
	T yx = axis_y * axis_x;
	T zx = axis_z * axis_x;

	T yy = axis_y * axis_y;
	T zy = axis_z * axis_y;

	T zz = axis_z * axis_z;

	Matrix<4,4,T>  mat_rot;

	mat_rot(0,0)  = xx + c * (T(1) - xx);
	mat_rot(1,0)  = yx - c * yx + s * axis_z;
	mat_rot(2,0)  = zx - c * zx - s * axis_y;
	mat_rot(3,0)  = T(0);

	mat_rot(0,1)  = yx - c * yx - s * axis_z;
	mat_rot(1,1)  = yy + c * (T(1) - yy);
	mat_rot(2,1)  = zy - c * zy + s * axis_x;
	mat_rot(3,1)  = T(0);

	mat_rot(0,2)  = zx - c * zx + s * axis_y;
	mat_rot(1,2)  = zy - c * zy - s * axis_x;
	mat_rot(2,2) = zz + c * (T(1) - zz);
	mat_rot(3,2) = T(0);

	mat_rot(0,3) = T(0);
	mat_rot(1,3) = T(0);
	mat_rot(2,3) = T(0);
	mat_rot(3,3) = T(1);

	mat = mat_rot * mat;
}

template <typename T>
void rotate(Vector<3,T>& axis, T angle, Matrix<4,4,T>& mat)
{
	rotate(axis[0], axis[1], axis[2], angle, mat) ;
}

template <typename T>
Vector<3,T> transform(const Vector<3,T>& P,const Matrix<4,4,T>& mat)
{
	Vector<4,T> Q(P[0], P[1], P[2], T(1));
	Vector<4,T> R = mat * Q;
	return Vector<3,T>(R[0]/R[3], R[1]/R[3], R[2]/R[3]);
}

} // namespace Geom

} // namespace CGoGN
