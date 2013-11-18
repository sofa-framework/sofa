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
#include "Topology/generic/attribmap.h"


namespace CGoGN
{


inline DartContainerBrowserSelector::DartContainerBrowserSelector(AttribMap& m, const FunctorSelect& fs):
	m_map(m)
{
	if (GenericMap::isMultiRes())
	{
		m_cont = &(m.getMRAttributeContainer());
	}
	else
	{
		m_cont = &(m.getAttributeContainer<DART>());
	}
	m_selector = fs.copy();
}

inline DartContainerBrowserSelector::~DartContainerBrowserSelector()
{
	delete m_selector;
}

inline unsigned int DartContainerBrowserSelector::begin() const
{
	if (GenericMap::isMultiRes())
	{
		unsigned int it = m_cont->realBegin() ;
		while ( (it != m_cont->realEnd()) && !m_selector->operator()(m_map.indexDart(it)) )
			m_cont->realNext(it);
		return it;
	}
	else
	{
		unsigned int it = m_cont->realBegin() ;
		while ( (it != m_cont->realEnd()) && !m_selector->operator()(Dart(it)) )
			m_cont->realNext(it);
		return it;
	}
}

inline unsigned int DartContainerBrowserSelector::end() const
{
	return m_cont->realEnd();
}

inline void DartContainerBrowserSelector::next(unsigned int& it) const
{
	do
	{
		m_cont->realNext(it) ;
	}
	while ( (it != m_cont->realEnd()) && !m_selector->operator()(Dart(it)) ) ;
}


inline void DartContainerBrowserSelector::enable()
{
	m_cont->setContainerBrowser(this);
}

inline void DartContainerBrowserSelector::disable()
{
	m_cont->setContainerBrowser(NULL);
}


inline ContainerBrowserLinked::ContainerBrowserLinked(AttribMap& m, unsigned int orbit):
	autoAttribute(true), m_first(0xffffffff), m_end(0xffffffff)
{
	m_cont = &(m.getAttributeContainer(orbit));
	m_links = m_cont->addAttribute<unsigned int>("Browser_Links") ;
}


inline ContainerBrowserLinked::ContainerBrowserLinked(AttributeContainer& c):
	m_cont(&c), autoAttribute(true), m_first(0xffffffff), m_end(0xffffffff)
{
	m_links = m_cont->addAttribute<unsigned int>("Browser_Links") ;
}

inline ContainerBrowserLinked::ContainerBrowserLinked(AttributeContainer& c, AttributeMultiVector<unsigned int>* links):
	m_cont(&c), m_links(links), autoAttribute(false), m_first(0xffffffff), m_end(0xffffffff)
{}


inline ContainerBrowserLinked::ContainerBrowserLinked(ContainerBrowserLinked& cbl):
	m_cont(cbl.m_cont), m_links(cbl.m_links), m_first(0xffffffff), m_end(0xffffffff)
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







template <unsigned int CELL>
inline ContainerBrowserCellMarked<CELL>::ContainerBrowserCellMarked(AttribMap& m,  CellMarker<CELL>& cm):
	 m_marker(cm)
{
	m_cont = &(m.getAttributeContainer<CELL>());
}


template <unsigned int CELL>
inline ContainerBrowserCellMarked<CELL>::~ContainerBrowserCellMarked()
{
}


template <unsigned int CELL>
inline unsigned int ContainerBrowserCellMarked<CELL>::begin() const
{
	unsigned int it = m_cont->realBegin() ;
	while ( (it != m_cont->realEnd()) && !m_marker.isMarked(it) )
		m_cont->realNext(it);

	return it;
}

template <unsigned int CELL>
inline unsigned int ContainerBrowserCellMarked<CELL>::end() const
{
	return m_cont->realEnd();
}

template <unsigned int CELL>
inline void ContainerBrowserCellMarked<CELL>::next(unsigned int& it) const
{
	do
	{
		m_cont->realNext(it) ;
	}
	while ( (it != m_cont->realEnd()) && !m_marker.isMarked(it) );
}

template <unsigned int CELL>
inline void ContainerBrowserCellMarked<CELL>::enable()
{
	m_cont->setContainerBrowser(this);
}

template <unsigned int CELL>
inline void ContainerBrowserCellMarked<CELL>::disable()
{
	m_cont->setContainerBrowser(NULL);
}


} // namespace CGoGN
