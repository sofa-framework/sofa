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


#ifndef __PARAMETERS_
#define __PARAMETERS_

#include "Geometry/vector_gen.h"
#include "Geometry/matrix.h"
#include "attributeHandler.h"

namespace CGoGN
{

struct PFP_STANDARD
{
	typedef float REAL;
	typedef Geom::Vector<3,REAL> VEC3;
	typedef Geom::Vector<4,REAL> VEC4;
	typedef Geom::Vector<6,REAL> VEC6;
	typedef Geom::Matrix<3,3,REAL> MATRIX33;
	typedef Geom::Matrix<4,4,REAL> MATRIX44;
	typedef Geom::Matrix<3,6,REAL> MATRIX36;

	static inline Geom::Vec3f toVec3f(const VEC3& P)
	{
		return P;
	}
};


struct PFP_DOUBLE
{
	typedef double REAL;
	typedef Geom::Vector<3,REAL> VEC3;
	typedef Geom::Vector<4,REAL> VEC4;
	typedef Geom::Vector<6,REAL> VEC6;
	typedef Geom::Matrix<3,3,REAL> MATRIX33;
	typedef Geom::Matrix<4,4,REAL> MATRIX44;
	typedef Geom::Matrix<3,6,REAL> MATRIX36;

	static inline Geom::Vec3f toVec3f(const VEC3& P)
	{
		return Geom::Vec3f(float(P[0]),float(P[1]),float(P[2]));
	}
};



}

#endif
