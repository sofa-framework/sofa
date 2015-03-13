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

#ifndef __3MR_LERP_MASK__
#define __3MR_LERP_MASK__

#include <cmath>
#include "Algo/Geometry/centroid.h"

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace MR
{

namespace Primal
{

namespace Masks
{

/* Linear Interpolation
 *********************************************************************************/
template <typename PFP>
class LerpVertexVertexFunctor : public FunctorType
{
protected:
	typename PFP::MAP& m_map ;
	AttributeHandler<typename PFP::VEC3, VERTEX, typename PFP::MAP>& m_position ;

public:
	LerpVertexVertexFunctor(typename PFP::MAP& m, AttributeHandler<typename PFP::VEC3, VERTEX, typename PFP::MAP>& p) : m_map(m), m_position(p)
	{}

	bool operator() (Dart d)
	{
		m_map.decCurrentLevel() ;
		typename PFP::VEC3 p = m_position[d] ;
//		std::cout << " p du niv i-1 = " << p << std::endl;
		m_map.incCurrentLevel() ;

//		std::cout << " p du niv i+1 = " << p << std::endl;

		//m_position[d] = p;

//		m_map.decCurrentLevel() ;
//		std::cout << "dec = " <<  m_position[d] << std::endl;
//		m_map.incCurrentLevel();
//		std::cout << "inc = " <<  m_position[d] << std::endl << std::endl;

		return false ;
	}
};

template <typename PFP>
class LerpEdgeVertexFunctor : public FunctorType
{
protected:
	typename PFP::MAP& m_map ;
	AttributeHandler<typename PFP::VEC3, VERTEX, typename PFP::MAP>& m_position ;

public:
	LerpEdgeVertexFunctor(typename PFP::MAP& m, AttributeHandler<typename PFP::VEC3, VERTEX, typename PFP::MAP>& p) : m_map(m), m_position(p)
	{}

	bool operator() (Dart d)
	{
		Dart d1 = m_map.phi2(d);

		m_map.decCurrentLevel();
		Dart d2 = m_map.phi2(d1);
		typename PFP::VEC3 p = (m_position[d1] + m_position[d2]) * typename PFP::REAL(0.5);
		m_map.incCurrentLevel();

		m_position[d] = p;

		return false;
	}
} ;

template <typename PFP>
class LerpFaceVertexFunctor : public FunctorType
{
protected:
	typename PFP::MAP& m_map ;
	AttributeHandler<typename PFP::VEC3, VERTEX, typename PFP::MAP>& m_position ;

public:
	LerpFaceVertexFunctor(typename PFP::MAP& m, AttributeHandler<typename PFP::VEC3, VERTEX, typename PFP::MAP>& p) : m_map(m), m_position(p)
	{}

	bool operator() (Dart d)
	{
		Dart df = m_map.phi1(m_map.phi1(d)) ;

		m_map.decCurrentLevel() ;
		typename PFP::VEC3 p =  Algo::Surface::Geometry::faceCentroid<PFP>(m_map, df, m_position);
		m_map.incCurrentLevel() ;

		m_position[d] = p ;

		return false ;
	}
} ;

template <typename PFP>
class LerpVolumeVertexFunctor : public FunctorType
{
protected:
	typename PFP::MAP& m_map ;
	AttributeHandler<typename PFP::VEC3, VERTEX, typename PFP::MAP>& m_position ;

public:
	LerpVolumeVertexFunctor(typename PFP::MAP& m, AttributeHandler<typename PFP::VEC3, VERTEX, typename PFP::MAP>& p) : m_map(m), m_position(p)
	{}

	bool operator() (Dart d)
	{
		Dart df = m_map.phi_1(m_map.phi2(m_map.phi1(d))) ;

		m_map.decCurrentLevel() ;
		typename PFP::VEC3 p =  Algo::Surface::Geometry::volumeCentroid<PFP>(m_map, df, m_position);
		m_map.incCurrentLevel() ;

		m_position[d] = p ;


		return false ;
	}
} ;


} // namespace Masks

} // namespace Primal

} // namespace MR

} // namespace Volume

} // namespace Algo

} // namespace CGoGN


#endif /* __3MR_FUNCTORS_PRIMAL__ */
