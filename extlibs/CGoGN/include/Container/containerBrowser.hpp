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

#include "Topology/generic/dart.h"

namespace CGoGN
{

template <typename MAP>
inline DartContainerBrowserSelector<MAP>::DartContainerBrowserSelector(MAP& m, const FunctorSelect& fs):
	m_map(m)
{
	m_cont = &m.getDartContainer();
	m_selector = fs.copy();
}

template <typename MAP>
inline DartContainerBrowserSelector<MAP>::~DartContainerBrowserSelector()
{
	delete m_selector;
}

template <typename MAP>
inline unsigned int DartContainerBrowserSelector<MAP>::begin() const
{
	unsigned int it = m_cont->realBegin() ;
	while ( (it != m_cont->realEnd()) && !m_selector->operator()(m_map.indexDart(it)) )
		m_cont->realNext(it);
	return it;
}

template <typename MAP>
inline unsigned int DartContainerBrowserSelector<MAP>::end() const
{
	return m_cont->realEnd();
}

template <typename MAP>
inline void DartContainerBrowserSelector<MAP>::next(unsigned int& it) const
{
	do
	{
		m_cont->realNext(it) ;
	}
	while ( (it != m_cont->realEnd()) && !m_selector->operator()(Dart(it)) ) ;
}

template <typename MAP>
inline void DartContainerBrowserSelector<MAP>::enable()
{
	m_cont->setContainerBrowser(this);
}

template <typename MAP>
inline void DartContainerBrowserSelector<MAP>::disable()
{
	m_cont->setContainerBrowser(NULL);
}







template <typename MAP, unsigned int CELL>
inline ContainerBrowserCellMarked<MAP, CELL>::ContainerBrowserCellMarked(GenericMap& m, CellMarker<MAP, CELL>& cm):
	 m_marker(cm)
{
	m_cont = &(m.getAttributeContainer<CELL>());
}

template <typename MAP, unsigned int CELL>
inline ContainerBrowserCellMarked<MAP, CELL>::~ContainerBrowserCellMarked()
{
}

template <typename MAP, unsigned int CELL>
inline unsigned int ContainerBrowserCellMarked<MAP, CELL>::begin() const
{
	unsigned int it = m_cont->realBegin() ;
	while ( (it != m_cont->realEnd()) && !m_marker.isMarked(it) )
		m_cont->realNext(it);

	return it;
}

template <typename MAP, unsigned int CELL>
inline unsigned int ContainerBrowserCellMarked<MAP, CELL>::end() const
{
	return m_cont->realEnd();
}

template <typename MAP, unsigned int CELL>
inline void ContainerBrowserCellMarked<MAP, CELL>::next(unsigned int& it) const
{
	do
	{
		m_cont->realNext(it) ;
	}
	while ( (it != m_cont->realEnd()) && !m_marker.isMarked(it) );
}

template <typename MAP, unsigned int CELL>
inline void ContainerBrowserCellMarked<MAP, CELL>::enable()
{
	m_cont->setContainerBrowser(this);
}

template <typename MAP, unsigned int CELL>
inline void ContainerBrowserCellMarked<MAP, CELL>::disable()
{
	m_cont->setContainerBrowser(NULL);
}






inline ContainerBrowserLinked::ContainerBrowserLinked(GenericMap& m, unsigned int orbit):
	autoAttribute(true),
	m_first(0xffffffff),
	m_end(0xffffffff)
{
	m_cont = &(m.getAttributeContainer(orbit));
	m_links = m_cont->addAttribute<unsigned int>("Browser_Links") ;
}

inline ContainerBrowserLinked::ContainerBrowserLinked(AttributeContainer& c):
	m_cont(&c),
	autoAttribute(true),
	m_first(0xffffffff),
	m_end(0xffffffff)
{
	m_links = m_cont->addAttribute<unsigned int>("Browser_Links") ;
}

inline ContainerBrowserLinked::ContainerBrowserLinked(AttributeContainer& c, AttributeMultiVector<unsigned int>* links):
	m_cont(&c),
	m_links(links),
	autoAttribute(false),
	m_first(0xffffffff),
	m_end(0xffffffff)
{}


inline ContainerBrowserLinked::ContainerBrowserLinked(ContainerBrowserLinked& cbl):
	m_cont(cbl.m_cont),
	m_links(cbl.m_links),
	m_first(0xffffffff),
	m_end(0xffffffff)
{}

inline ContainerBrowserLinked::~ContainerBrowserLinked()
{
	if (autoAttribute)
		m_cont->removeAttribute<unsigned int>("Browser_Links") ;
}

inline void ContainerBrowserLinked::clear()
{
	m_first = 0xffffffff;
	m_end   = 0xffffffff;
}

inline unsigned int ContainerBrowserLinked::begin() const
{
	return m_first ;
}

inline unsigned int ContainerBrowserLinked::end() const
{
	return 0xffffffff;
}

inline void ContainerBrowserLinked::next(unsigned int& it) const
{
	it = (*m_links)[it] ;
}

inline void ContainerBrowserLinked::pushBack(unsigned int it)
{
	(*m_links)[it] = 0xffffffff ;
	if (m_first == 0xffffffff)		// empty list
	{
		m_first = it ;
		m_end = it ;
	}
	else
	{
		(*m_links)[m_end] = it ;
		m_end = it ;
	}
}

inline void ContainerBrowserLinked::enable()
{
	m_cont->setContainerBrowser(this);
}

inline void ContainerBrowserLinked::disable()
{
	m_cont->setContainerBrowser(NULL);
}

} // namespace CGoGN
