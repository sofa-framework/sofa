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

#include "Topology/generic/attribmap.h"
#include "Topology/generic/attributeHandler.h"

namespace CGoGN
{


/**
 *  shortcut class for Dart AutoAttribute (Handler)
 */
template <typename T>
class DartAutoAttribute : public DartAttribute<T>
{
public:
	DartAutoAttribute(AttribMap& m, const std::string& nameAttr = "")
	{
		this->m_map = &m ;
		if(!m.isOrbitEmbedded<DART>())
			m.addEmbedding<DART>() ;
		AttributeMultiVector<T>* amv = this->m_map->template getAttributeContainer<DART>().template addAttribute<T>(nameAttr) ;
		this->m_attrib = amv ;
		this->valid = true ;
		this->registerInMap() ;
	}

	~DartAutoAttribute()
	{
		if (this->valid)
			reinterpret_cast<AttribMap*>(this->m_map)->removeAttribute<T>(*this) ;
	}
};

/**
 *  shortcut class for Vertex AutoAttribute (Handler)
 */
template <typename T>
class VertexAutoAttribute : public VertexAttribute<T>
{
public:
	VertexAutoAttribute(AttribMap& m, const std::string& nameAttr = "")
	{
		this->m_map = &m ;
		if(!m.isOrbitEmbedded<VERTEX>())
			m.addEmbedding<VERTEX>() ;
		AttributeMultiVector<T>* amv = this->m_map->template getAttributeContainer<VERTEX>().template addAttribute<T>(nameAttr) ;
		this->m_attrib = amv ;
		this->valid = true ;
		this->registerInMap() ;
	}

	~VertexAutoAttribute()
	{
		if (this->valid)
			reinterpret_cast<AttribMap*>(this->m_map)->removeAttribute<T>(*this) ;
	}
};

/**
 *  shortcut class for Edge AutoAttribute (Handler)
 */
template <typename T>
class EdgeAutoAttribute : public EdgeAttribute<T>
{
public:
	EdgeAutoAttribute(AttribMap& m, const std::string& nameAttr = "")
	{
		this->m_map = &m ;
		if(!m.isOrbitEmbedded<EDGE>())
			m.addEmbedding<EDGE>() ;
		AttributeMultiVector<T>* amv = this->m_map->template getAttributeContainer<EDGE>().template addAttribute<T>(nameAttr) ;
		this->m_attrib = amv ;
		this->valid = true ;
		this->registerInMap() ;
	}

	~EdgeAutoAttribute()
	{
		if (this->valid)
			reinterpret_cast<AttribMap*>(this->m_map)->removeAttribute<T>(*this) ;
	}
};

/**
 *  shortcut class for Face AutoAttribute (Handler)
 */
template <typename T>
class FaceAutoAttribute : public FaceAttribute<T>
{
public:
	FaceAutoAttribute(AttribMap& m, const std::string& nameAttr = "")
	{
		this->m_map = &m ;
		if(!m.isOrbitEmbedded<FACE>())
			m.addEmbedding<FACE>() ;
		AttributeMultiVector<T>* amv = this->m_map->template getAttributeContainer<FACE>().template addAttribute<T>(nameAttr) ;
		this->m_attrib = amv ;
		this->valid = true ;
		this->registerInMap() ;
	}

	~FaceAutoAttribute()
	{
		if (this->valid)
			reinterpret_cast<AttribMap*>(this->m_map)->removeAttribute<T>(*this) ;
	}
};

/**
 *  shortcut class for Volume AutoAttribute (Handler)
 */
template <typename T>
class VolumeAutoAttribute : public VolumeAttribute<T>
{
public:
	VolumeAutoAttribute(AttribMap& m, const std::string& nameAttr = "")
	{
		this->m_map = &m ;
		if(!m.isOrbitEmbedded<VOLUME>())
			m.addEmbedding<VOLUME>() ;
		AttributeMultiVector<T>* amv = this->m_map->template getAttributeContainer<VOLUME>().template addAttribute<T>(nameAttr) ;
		this->m_attrib = amv ;
		this->valid = true ;
		this->registerInMap() ;
	}

	~VolumeAutoAttribute()
	{
		if (this->valid)
			reinterpret_cast<AttribMap*>(this->m_map)->removeAttribute<T>(*this) ;
	}
};


} // namespace CGoGN

#endif
