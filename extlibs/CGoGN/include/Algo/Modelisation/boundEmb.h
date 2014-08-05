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

#ifndef _BOUND_EMBD_H
#define _BOUND_EMBD_H

namespace CGoGN
{

namespace Algo
{

namespace Modelisation
{

template <typename PFP>
void sewFaceEmb(typename PFP::MAP& map, Dart d, Dart e)
{
	map.sewFaces(d, e, false) ;
	if (map.template isOrbitEmbedded<EDGE>())
		Algo::Topo::initOrbitEmbeddingNewCell<EDGE>(map, d) ;
}

template <typename PFP>
Dart newFaceEmb(typename PFP::MAP& map, unsigned int n)
{
	Dart d = map.newFace(n,false);
	if (map.template isOrbitEmbedded<FACE>())
		Algo::Topo::initOrbitEmbeddingNewCell<FACE>(map, d) ;
	return d;
}

} // namespace Modelisation

} // namespace Algo

} // namespace CGoGN

#endif
