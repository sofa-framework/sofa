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

#include "Topology/gmap/gmap1.h"

namespace CGoGN
{

void GMap1::compactTopoRelations(const std::vector<unsigned int>& oldnew)
{
	for (unsigned int i = m_attribs[DART].begin(); i != m_attribs[DART].end(); m_attribs[DART].next(i))
	{
		{
			Dart& d = m_beta0->operator [](i);
			Dart e = Dart(oldnew[d.index]);
			if (d != e)
				d = e;
		}
		{
			Dart& d = m_beta1->operator [](i);
			Dart e = Dart(oldnew[d.index]);
			if (d != e)
				d = e;
		}
	}
}

/*! @name Constructors and Destructors
 *  To generate or delete faces in a 1-G-map
 *************************************************************************/

Dart GMap1::newCycle(unsigned int nbEdges)
{
	assert(nbEdges > 0 || !"Cannot create a face with no edge") ;

	Dart d0 =  GMap0::newEdge();	// create the first edge
	Dart dp = beta0(d0);			// store an extremity
	for (unsigned int i = 1; i < nbEdges; ++i)
	{
		Dart di = GMap0::newEdge();	// create the next edge
		beta1sew(dp,di);
		dp = beta0(di);	// change the preceding
	}
	beta1sew(dp,d0);	// sew the last with the first
	return d0;
}

//Dart GMap1::newBoundaryCycle(unsigned int nbEdges)
//{
//	assert(nbEdges > 0 || !"Cannot create a face with no edge") ;
//
//	Dart d0 =  GMap0::newEdge();	// create the first edge
//	boundaryMark2(d0);
//	boundaryMark2(beta0(d0));
//	Dart dp = beta0(d0);			// store an extremity
//	for (unsigned int i = 1; i < nbEdges; ++i)
//	{
//		Dart di = GMap0::newEdge();	// create the next edge
//		boundaryMark2(di);
//		boundaryMark2(beta0(di));
//		beta1sew(dp,di);
//		dp = beta0(di);	// change the preceding
//	}
//	beta1sew(dp,d0);	// sew the last with the first
//	return d0;
//}

void GMap1::deleteCycle(Dart d)
{
	Dart e = phi1(d);
	while (e != d)
	{
		Dart f = phi1(e);
		deleteEdge(e);
		e = f;
	}
	deleteEdge(d);
}

} // namespace CGoGN
