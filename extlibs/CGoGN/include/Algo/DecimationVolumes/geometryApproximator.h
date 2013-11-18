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

#ifndef __GEOMETRY_APPROXIMATOR_VOLUMES_H__
#define __GEOMETRY_APPROXIMATOR_VOLUMES_H__

#include "Algo/DecimationVolumes/approximator.h"

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace Decimation
{

template <typename PFP>
class Approximator_MidEdge : public Approximator<PFP, typename PFP::VEC3, EDGE>
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	Approximator_MidEdge(MAP& m, std::vector<VertexAttribute<VEC3>* > pos, Predictor<PFP, VEC3>* pred = NULL) :
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
class Approximator_MidFace : public Approximator<PFP, typename PFP::VEC3, FACE>
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	Approximator_MidFace(MAP& m, std::vector<VertexAttribute<VEC3>* > pos, Predictor<PFP, VEC3>* pred = NULL) :
		Approximator<PFP, VEC3, FACE>(m, pos, pred)
	{
		assert(pos.size() > 0 || !"Approximator_MidFace: attribute vector is empty") ;
	}
	~Approximator_MidFace()
	{}
	ApproximatorType getType() const { return A_MidFace ; }
	bool init() ;
	void approximate(Dart d) ;
} ;

template <typename PFP>
class Approximator_MidVolume : public Approximator<PFP, typename PFP::VEC3, VOLUME>
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	Approximator_MidVolume(MAP& m, std::vector<VertexAttribute<VEC3>* > pos, Predictor<PFP, VEC3>* pred = NULL) :
		Approximator<PFP, VEC3, VOLUME>(m, pos, pred)
	{
		assert(pos.size() > 0 || !"Approximator_MidVolume: attribute vector is empty") ;
	}
	~Approximator_MidVolume()
	{}
	ApproximatorType getType() const { return A_MidVolume ; }
	bool init() ;
	void approximate(Dart d) ;
} ;

template <typename PFP>
class Approximator_HalfEdgeCollapse : public Approximator<PFP, typename PFP::VEC3, DART>
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	Approximator_HalfEdgeCollapse(MAP& m, std::vector<VertexAttribute<VEC3>* > pos, Predictor<PFP, VEC3>* pred = NULL) :
		Approximator<PFP, VEC3, DART>(m, pos, pred)
	{
		assert(pos.size() > 0 || !"Approximator_HalfEdgeCollapse: attribute vector is empty") ;
	}
	~Approximator_HalfEdgeCollapse()
	{}
	ApproximatorType getType() const { return A_hHalfEdgeCollapse ; }
	bool init() ;
	void approximate(Dart d) ;
} ;

//template <typename PFP>
//class Approximator_QEM : public Approximator<PFP, typename PFP::VEC3, EDGE>
//{
//public:
//	typedef typename PFP::MAP MAP ;
//	typedef typename PFP::VEC3 VEC3 ;
//	typedef typename PFP::REAL REAL ;
//
//protected:
//	VertexAttribute<Utils::Quadric<REAL> > m_quadric ;
//
//public:
//	Approximator_QEM(MAP& m, std::vector<VertexAttribute<VEC3>* > pos, Predictor<PFP, VEC3>* pred = NULL) :
//		Approximator<PFP, VEC3, EDGE>(m, pos, pred)
//	{
//		assert(pos.size() > 0 || !"Approximator_QEM: attribute vector is empty") ;
//	}
//	~Approximator_QEM()
//	{}
//	ApproximatorType getType() const { return A_QEM ; }
//	bool init() ;
//	void approximate(Dart d) ;
//} ;



} //namespace Decimation

} //namespace Volume

} //namespace Algo

} //namespace CGoGN

#include "Algo/DecimationVolumes/geometryApproximator.hpp"

#endif
