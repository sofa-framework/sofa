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

#include "Algo/Filtering/functors.h"
#include "Algo/Selection/collector.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Filtering
{

template <typename PFP>
void filterTaubin(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position2)
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;

	Algo::Surface::Selection::Collector_OneRing<PFP> c(map) ;

	const float lambda = 0.6307 ;
	const float mu = -0.6732 ;

	CellMarkerNoUnmark<MAP, VERTEX> mv(map) ;

	FunctorAverage<VertexAttribute<VEC3, MAP> > fa1(position) ;
	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if( !mv.isMarked(d))
		{
			mv.mark(d);

			if(!map.isBoundaryVertex(d))
			{
				c.collectBorder(d) ;
				fa1.reset() ;
				c.applyOnBorder(fa1) ;
				VEC3 p = position[d] ;
				VEC3 displ = fa1.getAverage() - p ;
				displ *= lambda ;
				position2[d] = p + displ ;
			}
			else
				position2[d] = position[d] ;
		}
	}

	// unshrinking step
	FunctorAverage<VertexAttribute<VEC3, MAP> > fa2(position2) ;
	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if( mv.isMarked(d))
		{
			mv.unmark(d);

			if(!map.isBoundaryVertex(d))
			{
				c.collectBorder(d) ;
				fa2.reset() ;
				c.applyOnBorder(fa2) ;
				VEC3 p = position2[d] ;
				VEC3 displ = fa2.getAverage() - p ;
				displ *= mu ;
				position[d] = p + displ ;
			}
			else
				position[d] = position2[d] ;
		}
	}
}

/**
 * Taubin filter modified as proposed by [Lav09]
 */
template <typename PFP>
void filterTaubin_modified(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position2, typename PFP::REAL radius)
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;

	const float lambda = 0.6307 ;
	const float mu = -0.6732 ;

	CellMarkerNoUnmark<MAP, VERTEX> mv(map) ;

	FunctorAverageOnSphereBorder<PFP, VEC3> fa1(map, position, position) ;
	Algo::Surface::Selection::Collector_WithinSphere<PFP> c1(map, position, radius) ;
	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if( !mv.isMarked(d))
		{
			mv.mark(d);

			if(!map.isBoundaryVertex(d))
			{
				c1.collectBorder(d) ;
				VEC3 center = position[d] ;
				fa1.reset(center, radius) ;
				c1.applyOnBorder(fa1) ;
				VEC3 displ = fa1.getAverage() - center ;
				displ *= lambda ;
				position2[d] = center + displ ;
			}
			else
				position2[d] = position[d] ;
		}
	}

	// unshrinking step
	FunctorAverageOnSphereBorder<PFP, VEC3> fa2(map, position2, position2) ;
	Algo::Surface::Selection::Collector_WithinSphere<PFP> c2(map, position2, radius) ;
	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if( mv.isMarked(d))
		{
			mv.unmark(d);

			if(!map.isBoundaryVertex(d))
			{
				c2.collectBorder(d) ;
				VEC3 center = position2[d] ;
				fa2.reset(center, radius) ;
				c2.applyOnBorder(fa2) ;
				VEC3 displ = fa2.getAverage() - center ;
				displ *= mu ;
				position[d] = center + displ ;
			}
			else
				position[d] = position2[d] ;
		}
	}
}

} // namespace Filtering

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
