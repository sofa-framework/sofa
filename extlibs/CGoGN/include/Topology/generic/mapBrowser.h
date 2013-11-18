/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
*                                                                              *
* This library is free software ; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation ; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY ; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library ; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#ifndef _MAPBROWSER_H_
#define _MAPBROWSER_H_

#include "Container/containerBrowser.h"
#include "Topology/generic/dart.h"

namespace CGoGN
{
/**
 * @brief The MapBrowserLinked class for compatibility of old code only
 */
class MapBrowserLinked : public FunctorType
{
protected:
	ContainerBrowserLinked m_cbrowser;
public:
	inline MapBrowserLinked(AttribMap& m): m_cbrowser(m, DART)
	{
	#ifndef NDEBUG
		std::cout << "DO NOT USE MAP BROWER. USE TRAVERSOR INSTEAD "<< std::endl;
	#endif
	}

	inline Dart begin() const			{return Dart(m_cbrowser.begin());}
	inline Dart end() const			{return NIL;}
	inline void next(Dart& d) const	{m_cbrowser.next(d.index);}
	inline bool operator()(Dart d)		{m_cbrowser.pushBack(d.index); return false;}
	inline ContainerBrowserLinked& getContainerBrowser() {return m_cbrowser;}
};

} // namespace CGoGN


#endif /* _MAPBROWSER_H_ */
