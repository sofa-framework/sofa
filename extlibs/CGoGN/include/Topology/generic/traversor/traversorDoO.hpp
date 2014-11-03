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
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/if.hpp>
#include <boost/bind.hpp>

namespace bl = boost::lambda;

namespace CGoGN
{

template <typename MAP, unsigned int ORBIT>
TraversorDartsOfOrbit<MAP, ORBIT>::TraversorDartsOfOrbit(const MAP& map, Cell<ORBIT> c, unsigned int thread) :
    m_vd(GenericMap::askDartBuffer(thread)),
    m_thread(thread)
{
    //	map.foreach_dart_of_orbit(c, [&] (Dart d) { if (!map.isBoundaryMarkedCurrent(d)) m_vd->push_back(d); }, thread);
    map.foreach_dart_of_orbit(c, bl::if_(!bl::bind(&MAP::isBoundaryMarkedCurrent, boost::cref(map), bl::_1)) [bl::bind(static_cast<void (std::vector<Dart>::*)(const Dart&)>(&std::vector<Dart>::push_back), boost::ref(m_vd), bl::_1)] , thread);
    m_vd->push_back(NIL);
}

template <typename MAP, unsigned int ORBIT>
TraversorDartsOfOrbit<MAP, ORBIT>::~TraversorDartsOfOrbit()
{
    GenericMap::releaseDartBuffer(m_vd, m_thread);
}

template <typename MAP, unsigned int ORBIT>
Dart TraversorDartsOfOrbit<MAP, ORBIT>::begin()
{
    return *(m_current = m_vd->begin());
}

template <typename MAP, unsigned int ORBIT>
Dart TraversorDartsOfOrbit<MAP, ORBIT>::end()
{
    return NIL;
}

template <typename MAP, unsigned int ORBIT>
Dart TraversorDartsOfOrbit<MAP, ORBIT>::next()
{
    //	if (*m_current != NIL)
    //        ++m_current;
    return *(++m_current);
}



template <typename MAP, unsigned int ORBIT>
VTraversorDartsOfOrbit<MAP, ORBIT>::VTraversorDartsOfOrbit(const MAP& map, Cell<ORBIT> c, unsigned int thread):
    m_thread(thread)
{
    m_vd = GenericMap::askDartBuffer(thread);
    //	map.foreach_dart_of_orbit(c, [&] (Dart d) {	if (!map.isBoundaryMarkedCurrent(d)) m_vd->push_back(d); }, thread);
    //    map.foreach_dart_of_orbit(c,  bl::if_(!boost::ref(map).isBoundaryMarkedCurrent(bl::_1) [boost::ref(m_vd).push_back(bl::_1)], thread);
    map.foreach_dart_of_orbit(c,  bl::if_(!bl::bind(&MAP::isBoundaryMarkedCurrent, boost::cref(map), bl::_1))[bl::bind(static_cast<void (std::vector<Dart>::*)(const Dart&)>(&std::vector<Dart>::push_back), boost::ref(m_vd), bl::_1)], thread);
    m_vd->push_back(NIL);
}

template <typename MAP, unsigned int ORBIT>
VTraversorDartsOfOrbit<MAP, ORBIT>::~VTraversorDartsOfOrbit()
{
    GenericMap::releaseDartBuffer(m_vd, m_thread);
}

template <typename MAP, unsigned int ORBIT>
Dart VTraversorDartsOfOrbit<MAP, ORBIT>::begin()
{
    m_current = m_vd->begin();
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
