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

#ifndef _HYPER_MAP_H
#define _HYPER_MAP_H

#include "hypermap/hyperbasemap.h"

#include "point3d.h"
#include <GL/gl.h>
#include <gmtl/Output.h>

namespace CGoGN
{

template <typename DART>
class eHyperMap1 : public tBaseHyperMap<DART>
{
public:
	typedef typename tBaseHyperMap<DART>::Dart Dart ;

protected:
	Dart m_reference_dart;

public:

	/**
	* new dart in the map, with no position 
	*/
	Dart newEmbDart() 
	{
		Dart d = this->newDart();
		Emb::Point3D* p = Emb::Point3D::create();
		setEmbedding(d,0,p);
		return d;
	}

	/**
	* new dart in the map, 3D position is given
	*/
	Dart newEmbDart(float x, float y, float z) 
	{
		Dart d = newEmbDart();
		setVertexPosition(d,gmtl::Vec3f(x,y,z));
		return d;
	}

	void extractDart(Dart d)
	{
		 tBaseHyperMap<DART>::m_global_map.splice(tBaseHyperMap<DART>::m_global_map.begin(),*this,d);
	}

	void insertDart(Dart d)
	{
		this->splice(this->begin(), tBaseHyperMap<DART>::m_global_map,d);
	}


	/**
	* set reference dart
	*/
	void refDart(Dart d)
	{
		m_reference_dart = d;
	}

	/**
	* get reference dart
	*/ 
	Dart refDart()
	{
		return m_reference_dart;
	}

	/**
	* return embedding of vertex
	*/
	Emb::Embedding* getVertexEmb(Dart d)
	{
		return getEmbedding(d,DART::getVertexEmbId());
	}

	/**
	* set embedding of vertex
	*/
	void setVertexEmb(Dart d, Embedding* e)
	{
		embedOrbit(0,d,DART::getVertexEmbId(),e);
	}

	/**
	* return vertex position
	*/
	gmtl::Vec3f& getVertexPosition(Dart d)
	{
		Emb::Point3D* e = reinterpret_cast<Emb::Point3D*>(getEmbedding(d,0));
		return e->getPosition();
	}

	/**
	* set vertex position
	*/
	void setVertexPosition(Dart d, const gmtl::Vec3f& p)
	{
		Emb::Point3D* e = reinterpret_cast<Emb::Point3D*>(getEmbedding(d,0));
		e->setPosition(p);
	}

	/**
	* return the other dart of vertex, NIL if not
	*/
	Dart otherInVertex(Dart d)
	{
		Dart e = alpha1(d);
		if (e == this->nil())
			e = alpha_1(d);
		return e;
	}

	/**
	* return the other dart of edge, NIL if not
	*/
	Dart otherInEdge(Dart d)
	{
		Dart e = alpha0(d);
		if (e == this->nil())
			e = alpha_0(d);
		return e;
	}

};

} // namespace CGoGN



#endif