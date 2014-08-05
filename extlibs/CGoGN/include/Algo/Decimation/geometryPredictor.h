/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2013, IGG Team, ICube, University of Strasbourg           *
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

#ifndef __GEOMETRY_PREDICTOR_H__
#define __GEOMETRY_PREDICTOR_H__

#include "Algo/Decimation/predictor.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Decimation
{

template <typename PFP>
class Predictor_HalfCollapse : public Predictor<PFP, typename PFP::VEC3>
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	Predictor_HalfCollapse(MAP& m, VertexAttribute<VEC3, MAP>& pos) :
		Predictor<PFP, VEC3>(m, pos)
	{}
	PredictorType getType() { return P_HalfCollapse ; }
	bool init() { return true ; }
	void predict(Dart d2, Dart dd2) ;
} ;

template <typename PFP>
class Predictor_CornerCutting : public Predictor<PFP, typename PFP::VEC3>
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	Predictor_CornerCutting(MAP& m, VertexAttribute<VEC3, MAP>& pos) :
		Predictor<PFP, VEC3>(m, pos)
	{}
	PredictorType getType() { return P_CornerCutting ; }
	bool init() { return true ; }
	REAL autoAlpha(Dart d2, Dart dd2) ;
	void predict(Dart d2, Dart dd2) ;
} ;

template <typename PFP>
class Predictor_TangentPredict1 : public Predictor<PFP, typename PFP::VEC3>
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	Predictor_TangentPredict1(MAP& m, VertexAttribute<VEC3, MAP>& pos) :
		Predictor<PFP, VEC3>(m, pos)
	{}
	PredictorType getType() { return P_TangentPredict1 ; }
	bool init() { return true ; }
	void predictedTangent(Dart d2, Dart dd2, VEC3& displ, REAL& k1, REAL& k2) ;
	void predict(Dart d2, Dart dd2) ;
} ;

template <typename PFP>
class Predictor_TangentPredict2 : public Predictor<PFP, typename PFP::VEC3>
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	Predictor_TangentPredict2(MAP& m, VertexAttribute<VEC3, MAP>& pos) :
		Predictor<PFP, VEC3>(m, pos)
	{}
	PredictorType getType() { return P_TangentPredict2 ; }
	bool init() { return true ; }
	void predictedTangent(Dart d2, Dart dd2, VEC3& displ, REAL& k1, REAL& k2) ;
	void predict(Dart d2, Dart dd2) ;
} ;

} // namespace Decimation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/Decimation/geometryPredictor.hpp"

#endif
