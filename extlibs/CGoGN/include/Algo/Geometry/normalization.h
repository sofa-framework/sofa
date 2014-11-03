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

#ifndef __ALGO_GEOMETRY_NORMALIZATION_H__
#define __ALGO_GEOMETRY_NORMALIZATION_H__

#include "Geometry/vector_gen.h"
#include "Container/containerBrowser.h"

namespace CGoGN
{

namespace Algo
{

namespace Geometry
{

// Normalize the average length of given attribute
template <typename PFP, unsigned int ORBIT>
typename PFP::REAL normalizeLength(typename PFP::MAP & map, AttributeHandler<typename PFP::VEC3, ORBIT, typename PFP::MAP> & m_attr, const typename PFP::REAL scale = 1.0)
{
//	typename PFP::REAL sum = 0 ;
//	int count = 0 ;

//	MapBrowserLinked<typename PFP::MAP> mb(the_map) ;
//	the_map.foreach_orbit(m_attr.getOrbit(), mb) ;

//	for (Dart d = mb.begin(); d != mb.end(); mb.next(d))
//	{
//		typename PFP::VEC3 length = m_attr[d] ;
//		length -= m_attr[the_map.phi2(d)] ;
//        sum += length.norm() ;
//        ++count ;
//	}

//    sum /= typename PFP::REAL(count) ;

//    typename PFP::REAL div = sum / scale ; // mutiply res by scale factor

//	for (Dart d = mb.begin(); d != mb.end(); mb.next(d))
//	{
//        m_attr[d] /= div ;
//	}

//	return div ;

	typename PFP::REAL sum = 0 ;
	int count = 0 ;

	TraversorCell<typename PFP::MAP, ORBIT> trav(map);
	for (Dart d = trav.begin(); d != trav.end(); d=trav.next())
	{
		typename PFP::VEC3 length = m_attr[d] ;
		length -= m_attr[map.phi2(d)] ;
		sum += length.norm() ;
		++count ;
	}

	sum /= typename PFP::REAL(count) ;

	typename PFP::REAL div = sum / scale ; // mutiply res by scale factor

	for (Dart d = trav.begin(); d != trav.end(); d=trav.next())
	{
		m_attr[d] /= div ;
	}

	return div ;
}

} // namespace Geometry

} // namespace Algo

} // namespace CGoGN

#endif
