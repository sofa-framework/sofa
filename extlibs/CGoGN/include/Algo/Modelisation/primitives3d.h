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

#ifndef PRIMITIVES3D_H
#define PRIMITIVES3D_H

#include <vector>

#include "Utils/os_spec.h"
#include "Geometry/transfo.h"

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace Modelisation
{

/**
* class of geometric primitive
* It alloaw the creation of:
* - grid 3D
* - ??
* Topological creation methods are separated from embedding to
* easily allow specific embedding.
*/
template <typename PFP>
class Primitive3D
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;
	
public:
	enum {NONE, HEXAGRID};

protected:
	/**
	* Map in which we are working
	*/
	MAP& m_map;

	AttributeHandler<typename PFP::VEC3, VERTEX>& m_positions;

	/**
	* Reference dart of primitive
	*/
	Dart m_dart;

	/**
	* Kind of primitive (grid, cylinder,  
	*/
	int m_kind;

	/**
	* Table of vertex darts (one dart per vertex)
	* Order depend on primitive kind
	*/
	std::vector<Dart> m_tableVertDarts;

	/**
	* numbers that defined the subdivision of primitives
	*/
	unsigned int m_nx;
	unsigned int m_ny;
	unsigned int m_nz;

	/**
	* Create a 3D grid 
	* @param nx nb of cubes in x
	* @return the dart of vertex (0,0,0) direction z
	*/	
	Dart HexaGrid1Topo(unsigned int nx);

	/**
	* Create a 3D grid 
	* @param nx nb of cubes in x
	* @param ny nb of cubes in y
	* @return the dart of vertex (0,0,0) direction z
	*/	
	Dart HexaGrid2Topo(unsigned int nx, unsigned int ny);

public:

	/**
	* Constructor
	* @param map the map in which we want to work
	*/
	Primitive3D(MAP& map, AttributeHandler<typename PFP::VEC3, VERTEX>& position) :
		m_map(map),
		m_positions(position),
		m_kind(NONE),
		m_nx(-1), m_ny(-1), m_nz(-1)
	{}

	/**
	* get the table of darts (one per vertex)
	*/
	const std::vector<Dart>& getVertexDarts() { return m_tableVertDarts; }

	/*
	* get the reference dart
	*/
	Dart getDart() { return m_dart; }

	/**
	* transform the primitive with transformation matrice
	* @param matrice 
	*/
	void transform(const Geom::Matrix44f& matrice);

	/**
	* mark all darts of the primitive 
	* @param m the marker to use 
	*/
//	void mark(Mark m);

	/**
	* Create a 3D grid 
	* @param nx nb of cubes in x
	* @param ny nb of cubes in y
	* @param nz nb of cubes in z
	* @return the dart of vertex (0,0,0) direction z
	*/	
	Dart hexaGrid_topo(unsigned int nx, unsigned int ny, unsigned int nz);

	/**
	* embed the topo grid primitive 
	* Grid has size x,y,z centered on 0
	* @param x
	* @param y 
	* @param z
	* @param positions handler of position attribute
	*/
	void embedHexaGrid(float x, float y, float z);

	void embedHexaGrid(typename PFP::VEC3 origin, float x, float y, float z);

	/**
	* Create a 3D grid 
	* @param nx nb of cubes in x
	* @param ny nb of cubes in y
	* @param nz nb of cubes in z
	* @return the dart of vertex (0,0,0) direction z
	*/	
// 	Dart prismGrid_topo(int nx,int nz);

	/**
	* embed the topo grid primitive 
	* Grid has size x,y,z centered on 0
	* @param x
	* @param y 
	* @param z
	*/
// 	void embedPrismGrid( float x, float z);
};

} // namespace Modelisation

}

} // namespace Algo

} // namespace CGoGN

#include "Algo/Modelisation/primitives3d.hpp"

#endif
