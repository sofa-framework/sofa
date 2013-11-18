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

#include "Topology/map/map1.h"

namespace CGoGN
{

void Map1::compactTopoRelations(const std::vector<unsigned int>& oldnew)
{
	for (unsigned int i = m_attribs[DART].begin(); i != m_attribs[DART].end(); m_attribs[DART].next(i))
	{
		unsigned int d_index = dartIndex(m_phi1->operator [](i));
		if (d_index != oldnew[d_index])
			m_phi1->operator [](i) = Dart(oldnew[d_index]);

		d_index = dartIndex(m_phi_1->operator [](i));
		if (d_index != oldnew[d_index])
			m_phi_1->operator [](i) = Dart(oldnew[d_index]);
//		{
//			Dart d = m_phi1->operator [](i);
//				unsigned int d_index = dartIndex(m_phi1->operator [](i));
//				Dart e = Dart(oldnew[d_index]);
//				if (d != e)
//					d = e;
//		}
//		{
//			Dart& d = m_phi_1->operator [](i);
//			unsigned int d_index = dartIndex(d);
//			Dart e = Dart(oldnew[d_index]);
//			if (d != e)
//				d = e;
//		}
	}
}

/*! @name Generator and Deletor
 *  To generate or delete faces in a 1-map
 *************************************************************************/

Dart Map1::newCycle(unsigned int nbEdges)
{
	assert(nbEdges > 0 || !"Cannot create a face with no edge") ;
	Dart d = newDart() ;			// Create the first edge
	for (unsigned int i = 1 ; i < nbEdges ; ++i)
		Map1::cutEdge(d) ;			// Subdivide nbEdges-1 times this edge
	return d ;
}

//Dart Map1::newBoundaryCycle(unsigned int nbEdges)
//{
//	assert(nbEdges > 0 || !"Cannot create a face with no edge") ;
//	Dart d = newDart() ;			// Create the first edge
//	boundaryMark2(d);
//	for (unsigned int i = 1 ; i < nbEdges ; ++i)
//		Map1::cutEdge(d) ;			// Subdivide nbEdges-1 times this edge
//	return d ;
//}

void Map1::deleteCycle(Dart d)
{
	Dart e = phi1(d) ;
	while (e != d)
	{
		Dart f = phi1(e) ;
		deleteDart(e) ;
		e = f ;
	}
	deleteDart(d) ;
}

/*! @name Topological Operators
 *  Topological operations on 1-maps
 *************************************************************************/

void Map1::reverseCycle(Dart d)
{
	Dart e = phi1(d) ;			// Dart e is the first edge of the new face
	if (e == d) return ;		// Only one edge: nothing to do
	if (phi1(e) == d) return ;	// Only two edges: nothing to do

	phi1unsew(d) ;				// Detach e from the face of d

	Dart dNext = phi1(d) ;		// While the face of d contains more than two edges
	while (dNext != d)
	{
		phi1unsew(d) ;			// Unsew the edge after d
		phi1sew(e, dNext) ;		// Sew it after e (thus in reverse order)
		dNext = phi1(d) ;
	}
	phi1sew(e, d) ;				// Sew the last edge
}

} // namespace CGoGN
