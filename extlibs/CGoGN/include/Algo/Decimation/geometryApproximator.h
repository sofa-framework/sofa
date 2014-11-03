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

#ifndef __GEOMETRY_APPROXIMATOR_H__
#define __GEOMETRY_APPROXIMATOR_H__

#include "Algo/Decimation/approximator.h"
#include "Utils/convertType.h"
#include <Eigen/Dense>

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Decimation
{

template <typename PFP>
class Approximator_QEM : public Approximator<PFP, typename PFP::VEC3, EDGE>
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

protected:
	VertexAttribute<Utils::Quadric<REAL>, MAP> m_quadric ;

public:
	Approximator_QEM(MAP& m, std::vector<VertexAttribute<VEC3, MAP>*> pos, Predictor<PFP, VEC3>* pred = NULL) :
		Approximator<PFP, VEC3, EDGE>(m, pos, pred)
	{
		assert(pos.size() > 0 || !"Approximator_QEM: attribute vector is empty") ;
	}
	~Approximator_QEM()
	{}
	ApproximatorType getType() const { return A_QEM ; }
	bool init() ;
	void approximate(Dart d) ;
} ;

template <typename PFP>
class Approximator_QEMhalfEdge : public Approximator<PFP, typename PFP::VEC3, DART>
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

protected:
	VertexAttribute<Utils::Quadric<REAL>, MAP> m_quadric ;

public:
	Approximator_QEMhalfEdge(MAP& m, std::vector<VertexAttribute<VEC3, MAP>*> pos, Predictor<PFP, VEC3>* pred = NULL) :
		Approximator<PFP, VEC3, DART>(m, pos, pred)
	{
		assert(pos.size() > 0 || !"Approximator_QEMhalfEdge: attribute vector is empty") ;
	}
	~Approximator_QEMhalfEdge()
	{}
	ApproximatorType getType() const { return A_hQEM ; }
	bool init() ;
	void approximate(Dart d) ;
} ;

template <typename PFP>
class Approximator_MidEdge : public Approximator<PFP, typename PFP::VEC3, EDGE>
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	Approximator_MidEdge(MAP& m, std::vector<VertexAttribute<VEC3, MAP>*> pos, Predictor<PFP, VEC3>* pred = NULL) :
		Approximator<PFP, VEC3, EDGE>(m, pos, pred)
	{
		assert(pos.size() > 0 || !"Approximator_MidEdge: attribute vector is empty") ;
	}
	~Approximator_MidEdge()
	{}
	ApproximatorType getType() const { return A_MidEdge ; }
	bool init() ;
	void approximate(Dart d) ;
} ;

template <typename PFP>
class Approximator_HalfCollapse : public Approximator<PFP, typename PFP::VEC3, DART>
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	Approximator_HalfCollapse(MAP& m, std::vector<VertexAttribute<VEC3, MAP>*> pos, Predictor<PFP, VEC3>* pred = NULL) :
		Approximator<PFP, VEC3, DART>(m, pos, pred)
	{
		assert(pos.size() > 0 || !"Approximator_HalfCollapse: attribute vector is empty") ;
	}
	~Approximator_HalfCollapse() {}

	ApproximatorType getType() const { return A_hHalfCollapse ; }
	bool init() ;
	void approximate(Dart d) ;
} ;

template <typename PFP>
class Approximator_CornerCutting : public Approximator<PFP, typename PFP::VEC3, EDGE>
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	Approximator_CornerCutting(MAP& m, std::vector<VertexAttribute<VEC3, MAP>*> pos, Predictor<PFP, VEC3>* pred = NULL) :
		Approximator<PFP, VEC3, EDGE>(m, pos, pred)
	{
		assert(pos.size() > 0 || !"Approximator_CornerCutting: attribute vector is empty") ;
	}
	~Approximator_CornerCutting()
	{}
	ApproximatorType getType() const { return A_CornerCutting ; }
	bool init() ;
	void approximate(Dart d) ;
} ;

template <typename PFP>
class Approximator_NormalArea : public Approximator<PFP, typename PFP::VEC3, EDGE>
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

protected:
	EdgeAttribute<Geom::Matrix<3,3,REAL>, MAP> edgeMatrix ;

public:
	Approximator_NormalArea(MAP& m, std::vector<VertexAttribute<VEC3, MAP>*> pos, Predictor<PFP, VEC3>* pred = NULL) :
		Approximator<PFP, VEC3, EDGE>(m, pos, pred)
	{
		assert(pos.size() > 0 || !"Approximator_NormalArea: attribute vector is empty") ;
	}
	~Approximator_NormalArea()
	{}
	ApproximatorType getType() const { return A_NormalArea ; }
	bool init() ;
	void approximate(Dart d) ;
} ;

} // namespace Decimation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/Decimation/geometryApproximator.hpp"

#endif
