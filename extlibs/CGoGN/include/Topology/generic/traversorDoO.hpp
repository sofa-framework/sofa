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

#include "Topology/generic/genericmap.h"
#include "Topology/generic/functor.h"

namespace CGoGN
{

template <typename MAP, unsigned int ORBIT>
TraversorDartsOfOrbit<MAP, ORBIT>::TraversorDartsOfOrbit(const MAP& map, Dart d, unsigned int thread)
{
	m_vd.reserve(16);
	FunctorStoreNotBoundary<MAP> fs(map, m_vd);
	const_cast<MAP&>(map).template foreach_dart_of_orbit<ORBIT>(d, fs, thread);
	m_vd.push_back(NIL);
}

template <typename MAP, unsigned int ORBIT>
Dart TraversorDartsOfOrbit<MAP, ORBIT>::begin()
{
	m_current = m_vd.begin();
	return *m_current;
}

template <typename MAP, unsigned int ORBIT>
Dart TraversorDartsOfOrbit<MAP, ORBIT>::end()
{
	return NIL;
}

template <typename MAP, unsigned int ORBIT>
Dart TraversorDartsOfOrbit<MAP, ORBIT>::next()
{
	if (*m_current != NIL)
		m_current++;
	return *m_current;
}




template <typename MAP, unsigned int ORBIT>
VTraversorDartsOfOrbit<MAP, ORBIT>::VTraversorDartsOfOrbit(const MAP& map, Dart d, unsigned int thread)
{
	m_vd.reserve(16);
	FunctorStoreNotBoundary<MAP> fs(map, m_vd);
	map.template foreach_dart_of_orbit<ORBIT>(d, fs, thread);
	m_vd.push_back(NIL);
}

template <typename MAP, unsigned int ORBIT>
Dart VTraversorDartsOfOrbit<MAP, ORBIT>::begin()
{
	m_current = m_vd.begin();
	return *m_current;
}

template <typename MAP, unsigned int ORBIT>
Dart VTraversorDartsOfOrbit<MAP, ORBIT>::end()
{
	return NIL;
}

template <typename MAP, unsigned int ORBIT>
Dart VTraversorDartsOfOrbit<MAP, ORBIT>::next()
{
	if (*m_current != NIL)
		m_current++;
	return *m_current;
}



} // namespace CGoGN
