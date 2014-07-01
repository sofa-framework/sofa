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

#ifndef __FILTERING_FUNCTORS_H__
#define __FILTERING_FUNCTORS_H__

#include "Topology/generic/functor.h"
#include "Algo/Geometry/intersection.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Filtering
{

template <typename ATTR_TYPE>
class FunctorAverage : public virtual FunctorType
{
	typedef typename ATTR_TYPE::DATA_TYPE T ;

protected:
	const ATTR_TYPE& attr ;
	T sum ;
	unsigned int count ;

public:
	FunctorAverage(const ATTR_TYPE& a) : FunctorType(), attr(a), sum(0), count(0)
	{}
	bool operator()(Dart d)
	{
		sum += attr[d] ;
		++count ;
		return false ;
	}
	inline void reset() { sum = T(0) ; count = 0 ; }
	inline T getSum() { return sum ; }
	inline unsigned int getCount() { return count ; }
	inline T getAverage() { return sum / typename T::DATA_TYPE(count) ; }
} ;

template <typename PFP, typename T>
class FunctorAverageOnSphereBorder : public FunctorMap<typename PFP::MAP>
{
	typedef typename PFP::VEC3 VEC3;

protected:
	const VertexAttribute<T, typename PFP::MAP>& attr ;
	const VertexAttribute<VEC3, typename PFP::MAP>& position ;
	VEC3 center;
	typename PFP::REAL radius;
	T sum ;
	unsigned int count ;

public:
	FunctorAverageOnSphereBorder(typename PFP::MAP& map, const VertexAttribute<T, typename PFP::MAP>& a, const VertexAttribute<VEC3, typename PFP::MAP>& p) :
		FunctorMap<typename PFP::MAP>(map), attr(a), position(p), sum(0), count(0)
	{
		center = VEC3(0);
		radius = 0;
	}
	bool operator()(Dart d)
	{
		typename PFP::REAL alpha = 0;
		Geometry::intersectionSphereEdge<PFP>(this->m_map, center, radius, d, position, alpha);
		sum += (1 - alpha) * attr[d] + alpha * attr[this->m_map.phi1(d)] ;
		++count ;
		return false ;
	}
	inline void reset(VEC3& c, typename PFP::REAL r) { center = c; radius = r; sum = T(0) ; count = 0 ; }
	inline T getSum() { return sum ; }
	inline unsigned int getCount() { return count ; }
	inline T getAverage() { return sum / typename T::DATA_TYPE(count) ; }
} ;

} // namespace Filtering

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#endif
