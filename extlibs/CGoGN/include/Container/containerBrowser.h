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

#ifndef __CONTAINER_BROWSER__
#define __CONTAINER_BROWSER__

#include "Container/attributeContainer.h"
#include "Topology/generic/functor.h"

class AttribMap;

namespace CGoGN
{

/**
 * Browser that traverses all darts and jumps over
 * those not selected by the selector
 */
template <typename MAP>
class DartContainerBrowserSelector : public ContainerBrowser
{
protected:
	AttributeContainer* m_cont ;
	const FunctorSelect* m_selector ;
	MAP& m_map;

public:
	DartContainerBrowserSelector(MAP& m, const FunctorSelect& fs);
	~DartContainerBrowserSelector();
	unsigned int begin() const;
	unsigned int end() const;
	void next(unsigned int& it) const;
	void enable();
	void disable();
} ;



template <typename MAP, unsigned int CELL>
class ContainerBrowserCellMarked : public ContainerBrowser
{
protected:
	AttributeContainer* m_cont ;
	CellMarker<MAP, CELL>& m_marker ;

public:
	ContainerBrowserCellMarked(GenericMap& m, CellMarker<MAP, CELL>& cm);
	~ContainerBrowserCellMarked();
	unsigned int begin() const;
	unsigned int end() const;
	void next(unsigned int& it) const;
	void enable();
	void disable();
} ;



class ContainerBrowserLinked : public ContainerBrowser
{
protected:
	// The browsed map
	AttributeContainer* m_cont ;
	AttributeMultiVector<unsigned int>* m_links ;
	bool autoAttribute ;
	unsigned int m_first ;
	unsigned int m_end ;

public:
	ContainerBrowserLinked(GenericMap& m, unsigned int orbit);
	ContainerBrowserLinked(AttributeContainer& c);
	ContainerBrowserLinked(AttributeContainer& c, AttributeMultiVector<unsigned int>* links);
	/**
	 * @brief ContainerBrowserLinked contructor that share the container and links
	 * @param cbl the ContainerBrowserLinked
	 */
	ContainerBrowserLinked(ContainerBrowserLinked& cbl);
	~ContainerBrowserLinked();

	unsigned int begin() const;
	unsigned int end() const;
	void next(unsigned int& it) const;
	void enable();
	void disable();

	void clear();
	void pushBack(unsigned int it);

//	void popFront();
//	void addSelected(const FunctorSelect& fs);
//	void append(MapBrowserLinked& mbl);
} ;

} // namespace CGoGN


#include "containerBrowser.hpp"

#endif
