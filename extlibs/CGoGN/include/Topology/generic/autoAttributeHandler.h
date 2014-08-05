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

#ifndef __AUTO_ATTRIBUTE_HANDLER_H__
#define __AUTO_ATTRIBUTE_HANDLER_H__

#include "Topology/generic/attributeHandler.h"

namespace CGoGN
{

/**
 *  shortcut class for Dart AutoAttribute (Handler)
 */
template <typename T, typename MAP>
class DartAutoAttribute : public DartAttribute<T, MAP>
{
public:
	DartAutoAttribute(MAP& m, const std::string& nameAttr = "")
	{
		this->m_map = &m ;
		if(!m.template isOrbitEmbedded<DART>())
			m.template addEmbedding<DART>() ;
		AttributeMultiVector<T>* amv = this->m_map->template getAttributeContainer<DART>().template addAttribute<T>(nameAttr) ;
		this->m_attrib = amv ;
		this->valid = true ;
		this->registerInMap() ;
	}

	~DartAutoAttribute()
	{
		if (this->valid)
			this->m_map->removeAttribute(*this) ;
	}
};

/**
 *  shortcut class for Vertex AutoAttribute (Handler)
 */
template <typename T, typename MAP>
class VertexAutoAttribute : public VertexAttribute<T, MAP>
{
public:
	VertexAutoAttribute(MAP& m, const std::string& nameAttr = "")
	{
		this->m_map = &m ;
		if(!m.template isOrbitEmbedded<VERTEX>())
			m.template addEmbedding<VERTEX>() ;
		AttributeMultiVector<T>* amv = this->m_map->template getAttributeContainer<VERTEX>().template addAttribute<T>(nameAttr) ;
		this->m_attrib = amv ;
		this->valid = true ;
		this->registerInMap() ;
	}

	~VertexAutoAttribute()
	{
		if (this->valid)
			this->m_map->removeAttribute(*this) ;
	}
};

/**
 *  shortcut class for Edge AutoAttribute (Handler)
 */
template <typename T, typename MAP>
class EdgeAutoAttribute : public EdgeAttribute<T, MAP>
{
public:
	EdgeAutoAttribute(MAP& m, const std::string& nameAttr = "")
	{
		this->m_map = &m ;
		if(!m.template isOrbitEmbedded<EDGE>())
			m.template addEmbedding<EDGE>() ;
		AttributeMultiVector<T>* amv = this->m_map->template getAttributeContainer<EDGE>().template addAttribute<T>(nameAttr) ;
		this->m_attrib = amv ;
		this->valid = true ;
		this->registerInMap() ;
	}

	~EdgeAutoAttribute()
	{
		if (this->valid)
			this->m_map->removeAttribute(*this) ;
	}
};

/**
 *  shortcut class for Face AutoAttribute (Handler)
 */
template <typename T, typename MAP>
class FaceAutoAttribute : public FaceAttribute<T, MAP>
{
public:
	FaceAutoAttribute(MAP& m, const std::string& nameAttr = "")
	{
		this->m_map = &m ;
		if(!m.template isOrbitEmbedded<FACE>())
			m.template addEmbedding<FACE>() ;
		AttributeMultiVector<T>* amv = this->m_map->template getAttributeContainer<FACE>().template addAttribute<T>(nameAttr) ;
		this->m_attrib = amv ;
		this->valid = true ;
		this->registerInMap() ;
	}

	~FaceAutoAttribute()
	{
		if (this->valid)
			this->m_map->removeAttribute(*this) ;
	}
};

/**
 *  shortcut class for Volume AutoAttribute (Handler)
 */
template <typename T, typename MAP>
class VolumeAutoAttribute : public VolumeAttribute<T, MAP>
{
public:
	VolumeAutoAttribute(MAP& m, const std::string& nameAttr = "")
	{
		this->m_map = &m ;
		if(!m.template isOrbitEmbedded<VOLUME>())
			m.template addEmbedding<VOLUME>() ;
		AttributeMultiVector<T>* amv = this->m_map->template getAttributeContainer<VOLUME>().template addAttribute<T>(nameAttr) ;
		this->m_attrib = amv ;
		this->valid = true ;
		this->registerInMap() ;
	}

	~VolumeAutoAttribute()
	{
		if (this->valid)
			this->m_map->removeAttribute(*this) ;
	}
};

} // namespace CGoGN

#endif
