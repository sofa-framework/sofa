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

#include "Geometry/vector_gen.h"

namespace CGoGN
{

namespace Algo
{

namespace Render
{

template<typename PFP>
void drawerAddEdge(Utils::Drawer& dr, typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, float k)
{

	const Geom::Vec3f& P = PFP::toVec3f(positions[d]);
	Dart e = map.phi1(d);
	const Geom::Vec3f& Q = PFP::toVec3f(positions[e]);

	Geom::Vec3f C = (P+Q)/ typename PFP::REAL(2.0);

	dr.vertex(C*(1.0f-k) + k*P);
	dr.vertex(C*(1.0f-k) + k*Q);
}

template<typename PFP>
void drawerAddEdgeShrink(Utils::Drawer& dr, typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, const typename PFP::VEC3& C, float k)
{
	const Geom::Vec3f& P = PFP::toVec3f(positions[d]);
	Dart e = map.phi1(d);
	const Geom::Vec3f& Q = PFP::toVec3f(positions[e]);

	dr.vertex(C*(1.0f-k) + k*P);
	dr.vertex(C*(1.0f-k) + k*Q);
}

template<typename PFP>
void drawerAddFace(Utils::Drawer& dr, typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, float k)
{
	Geom::Vec3f C = PFP::toVec3f(Algo::Surface::Geometry::faceCentroid<PFP>(map,d,positions));

	Traversor2FE<typename PFP::MAP> trav(map,d);
	for (Dart e=trav.begin(); e!=trav.end(); e=trav.next())
	{
		drawerAddEdgeShrink<PFP>(dr,map,e,positions,C,k);
	}
}

template<typename PFP>
void drawerAddVolume(Utils::Drawer& dr, typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions,float k)
{
	Geom::Vec3f C = PFP::toVec3f(Algo::Surface::Geometry::volumeCentroid<PFP>(map,d,positions));

	Traversor3WE<typename PFP::MAP> trav(map,d);
	for (Dart e=trav.begin(); e!=trav.end(); e=trav.next())
		drawerAddEdgeShrink<PFP>(dr,map,e,positions,C,k);
}

template<typename PFP>
void drawerVertices(Utils::Drawer& dr, typename PFP::MAP& /*map*/, std::vector<Dart>& vd, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions)
{
	dr.begin(GL_POINTS);
	for (std::vector<Dart>::iterator it = vd.begin(); it !=vd.end(); ++it)
		dr.vertex(positions[*it]);
	dr.end();
}

template<typename PFP>
void drawerEdges(Utils::Drawer& dr, typename PFP::MAP& map, std::vector<Dart>& vd, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions,float k)
{
	dr.begin(GL_LINES);
	for (std::vector<Dart>::iterator it = vd.begin(); it !=vd.end(); ++it)
		drawerAddEdge<PFP>(dr,map,*it,positions,k);
	dr.end();
}

template<typename PFP>
void drawerFaces(Utils::Drawer& dr, typename PFP::MAP& map, std::vector<Dart>& vd, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions,float k)
{
	dr.begin(GL_LINES);
	for (std::vector<Dart>::iterator it = vd.begin(); it !=vd.end(); ++it)
		drawerAddFace<PFP>(dr,map,*it,positions,k);
	dr.end();
}
template<typename PFP>
void drawerVolumes(Utils::Drawer& dr, typename PFP::MAP& map, std::vector<Dart>& vd, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions,float k)
{
	dr.begin(GL_LINES);
	for (std::vector<Dart>::iterator it = vd.begin(); it !=vd.end(); ++it)
		drawerAddVolume<PFP>(dr,map,*it,positions,k);
	dr.end();
}

template<typename PFP>
void drawerVertex(Utils::Drawer& dr, typename PFP::MAP& /*map*/, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions)
{
	dr.begin(GL_POINTS);
	dr.vertex(positions[d]);
	dr.end();
}

template<typename PFP>
void drawerEdge(Utils::Drawer& dr, typename PFP::MAP& map,  Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions,float k)
{
	dr.begin(GL_LINES);
	drawerAddEdge<PFP>(dr,map,d,positions,k);
	dr.end();
}

template<typename PFP>
void drawerFace(Utils::Drawer& dr, typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions,float k)
{
	dr.begin(GL_LINES);
	drawerAddFace<PFP>(dr,map,d,positions,k);
	dr.end();
}

template<typename PFP>
void drawerVolume(Utils::Drawer& dr, typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions,float k)
{
	dr.begin(GL_LINES);
	drawerAddVolume<PFP>(dr,map,d,positions,k);
	dr.end();
}

template<typename PFP>
void drawerCells(unsigned int cell, Utils::Drawer& dr, typename PFP::MAP& map, std::vector<Dart>& vd, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, float k)
{
	switch(cell)
	{
	case VERTEX:
		drawerVertices<PFP>(dr, map, vd, positions);
		break;
	case EDGE:
		drawerEdges<PFP>(dr, map, vd, positions,k);
		break;
	case FACE:
		drawerFaces<PFP>(dr, map, vd, positions,k);
		break;
	case VOLUME:
		drawerVolumes<PFP>(dr, map, vd, positions,k);
		break;
	default:
		break;
	}
}

template<typename PFP>
void drawerCell(unsigned int cell, Utils::Drawer& dr, typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& positions, float k)
{
	switch(cell)
	{
	case VERTEX:
		drawerVertex<PFP>(dr, map, d, positions);
		break;
	case EDGE:
		drawerEdge<PFP>(dr, map, d, positions,k);
		break;
	case FACE:
		drawerFace<PFP>(dr, map, d, positions,k);
		break;
	case VOLUME:
		drawerVolume<PFP>(dr, map, d, positions,k);
		break;
	default:
		break;
	}
}

}

}

}
