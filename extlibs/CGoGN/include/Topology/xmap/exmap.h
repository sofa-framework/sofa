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

#ifndef __EXMAP_H__
#define __EXMAP_H__

#include "xmap/xmap.h"

namespace CGoGN
{

template <typename DART>
class e0xmap: public tXMap<DART>
{
public:
	typedef typename tXMap<DART>::Dart Dart ;

private:
	void setSingleVertexEmb(Dart d, Embedding* e) { setEmbedding(d,DART::getVertexEmbId(),e); }

public:
	Emb::Embedding* getVertexEmb(Dart d) { return getEmbedding(d,DART::getVertexEmbId()); }

	void setVertexEmb(Dart d, Embedding* e) { embedOrbit(0,d,DART::getVertexEmbId(),e); }

	/**
	* return embedding of vertex of face
	*/
	Emb::Embedding* getFaceVertexEmb(Dart d) { return getEmbedding(d,DART::getDartEmbId());}

	/**
	* set embedding of vertex of face
	*/
	void setFaceVertexEmb(Dart d, Embedding* e) 	{setEmbedding(d,DART::getDartEmbId(),e);}
} ;

} // namespace CGoGN

#include "xmap/exmap.hpp"

#endif
