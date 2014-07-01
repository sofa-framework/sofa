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

#include "Algo/Geometry/intersection.h"
#include "Algo/Geometry/orientation.h"
#include "Algo/Geometry/basic.h"
#include "Algo/Geometry/plane.h"

#include <limits>

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

template <typename PFP>
bool isConvex(typename PFP::MAP& map, Vol v, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, bool CCW, unsigned int thread)
{
	//get all the dart of the volume
	std::vector<Dart> vStore;
	map.foreach_dart_of_orbit(v, [&] (Dart d) { vStore.push_back(d); }, thread);

	bool convex = true;

	DartMarkerStore<typename PFP::MAP> m(map, thread);
	for (std::vector<Dart>::iterator it = vStore.begin() ; it != vStore.end() && convex ; ++it)
	{
		Dart e = *it;
		if (!m.isMarked(e))
		{
			m.markOrbit<EDGE>(e) ;
			convex = isTetrahedronWellOriented<PFP>(map, e, position, CCW) ;
		}
	}

	return convex;
}

// TODO add thread Pameter
template <typename PFP>
bool isPointInVolume(typename PFP::MAP& map, Vol v, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const typename PFP::VEC3& point)
{
	typedef typename PFP::VEC3 VEC3;

	//number of intersection between a ray and the volume must be odd
	int countInter = 0;
	int countInter2 = 0;

	VEC3 dir(0.9f,1.1f,1.3f);

	std::vector<VEC3> interPrec;
	interPrec.reserve(16);

	std::vector<Face> visitedFaces;			// Faces that are traversed
	visitedFaces.reserve(64);
	Face f(v.dart);
	visitedFaces.push_back(f);				// Start with the first face of v

	DartMarkerStore<typename PFP::MAP> mark(map);
	mark.markOrbit(f) ;

	for(unsigned int iface = 0; iface != visitedFaces.size(); ++iface)
	{
		f = visitedFaces[iface];
		VEC3 inter;
		bool interRes = intersectionLineConvexFace<PFP>(map, f, position, point, dir, inter);
		if (interRes)
		{
			// check if already intersect on same point (a vertex certainly)
			bool alreadyfound = false;
			for(typename std::vector<VEC3>::iterator it = interPrec.begin(); !alreadyfound && it != interPrec.end(); ++it)
			{
				if (Geom::arePointsEquals(*it, inter))
					alreadyfound = true;
			}

			if (!alreadyfound)
			{
				float v = dir * (inter - point);
				if (v > 0)
					++countInter;
				if (v < 0)
					++countInter2;
				interPrec.push_back(inter);
			}
		}
		// add all face neighbours to the table
		foreach_adjacent2<EDGE>(map, f, [&] (Face ff)
		{
			if(!mark.isMarked(ff)) // not already marked
			{
				visitedFaces.push_back(ff) ;
				mark.markOrbit(ff) ;
			}
		});
	}

	//if the point is in the volume there is an odd number of intersection with all faces with any direction
	return ((countInter % 2) != 0) && ((countInter2 % 2) != 0); //	return (countInter % 2) == 1;
}

template <typename PFP>
bool isPointInConvexVolume(typename PFP::MAP& map, Vol v, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const typename PFP::VEC3& point, bool CCW)
{
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL;

	std::vector<Face> visitedFaces;			// Faces that are traversed
	visitedFaces.reserve(64);
	Face f(v.dart);
	visitedFaces.push_back(f);				// Start with the first face of v

	DartMarkerStore<typename PFP::MAP> mark(map);		// Lock a marker

	for (std::vector<Face>::iterator face = visitedFaces.begin(); face != visitedFaces.end(); ++face)
	{
		f = *face;
		if (!mark.isMarked(f))
		{
			mark.markOrbit(f);

			Geom::Plane3D<REAL> p = facePlane<PFP>(map, f, position);
			Geom::Orientation3D o3d = p.orient(point);
			if(CCW)
			{
				if(o3d == Geom::OVER)
					return false;
			}
			else if(o3d == Geom::UNDER)
				return false;

			// add all face neighbours to the table
			foreach_adjacent2<EDGE>(map, f, [&] (Face ff)
			{
				if(!mark.isMarked(ff)) // not already marked
					visitedFaces.push_back(ff) ;
			});
		}
	}

	return true;
}

template <typename PFP>
bool isPointInConvexFace(typename PFP::MAP& map, Face f, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const typename PFP::VEC3& point, bool CCW)
{
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL;

	Geom::Plane3D<REAL> pl = Geometry::facePlane<PFP>(map, f, position);
	Geom::Orientation3D o3d = pl.orient(point);
	if(o3d == Geom::ON)
	{
		Traversor2FV<typename PFP::MAP> tfv(map, f) ;
		for(Vertex v = tfv.begin(); v != tfv.end(); v = tfv.next())
		{
			VEC3 N = pl.normal();
			VEC3 v2(position[map.phi1(v.dart)] - position[v]);
			VEC3 norm2 = N ^ v2;
			Geom::Plane3D<REAL> pl2(norm2, position[v]);
			o3d = pl2.orient(point);
			if(CCW)
			{
				if(o3d == Geom::UNDER)
					return false;
			}
			else if(o3d == Geom::OVER)
				return false;
		}
		return true;
	}

	return false;
}

template <typename PFP>
bool isPointInConvexFace2D(typename PFP::MAP& map, Face f, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const typename PFP::VEC3& point, bool CCW )
{
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL;

// 	CGoGNout << "point " << point << "d " << d << "faceDeg " << map.faceDegree(d) << CGoGNendl;

	Geom::Orientation2D o2d;

	Traversor2FV<typename PFP::MAP> tfv(map, f) ;
	for(Vertex v = tfv.begin(); v != tfv.end(); v = tfv.next())
	{
		o2d = Geom::testOrientation2D(point, position[v], position[map.phi1(v.dart)]);
		if(CCW)
		{
			if(o2d == Geom::RIGHT)
				return false;
		}
		else if(o2d == Geom::LEFT)
			return false;
	}

	return true;
}

template <typename PFP>
bool isPointOnEdge(typename PFP::MAP& map, Edge e, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const typename PFP::VEC3& point)
{
// 	typedef typename PFP::REAL REAL;
// 	typedef typename PFP::VEC3 VEC3 ;
// 
// 	VEC3 v1 = vectorOutOfDart<PFP>(map, d, positions);
// 	VEC3 v2(point - positions[d]);
// 
// 	v1.normalize();
// 	v2.normalize();
// 
// 	return fabs(REAL(1) - (v1*v2)) < std::numeric_limits<REAL>::min();

	Dart d = e.dart;

	if(
		( isPointOnHalfEdge<PFP>(map,d,position,point) && isPointOnHalfEdge<PFP>(map,map.phi2(d),position,point) ) ||
		isPointOnVertex<PFP>(map,d,position,point) ||
		isPointOnVertex<PFP>(map,map.phi1(d),position,point)
	)
		return true;
	else
	{
		CGoGNout << " point " << point << CGoGNendl;
		CGoGNout << " d1 " << position[d] << " d2 " << position[map.phi2(d)] << CGoGNendl;
		return false;
	}
}

template <typename PFP>
bool isPointOnHalfEdge(typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const typename PFP::VEC3& point)
{
	typedef typename PFP::REAL REAL;
	typedef typename PFP::VEC3 VEC3;

	VEC3 v1 = vectorOutOfDart<PFP>(map, d, position);
	VEC3 v2(point - position[d]);

	v1.normalize();
	v2.normalize();

	return abs(v1*v2) <= REAL(0.00001);
}

template <typename PFP>
bool isPointOnVertex(typename PFP::MAP& map, Vertex v, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const typename PFP::VEC3& point)
{
	return Geom::arePointsEquals(point, position[v]);
}

template <typename PFP>
bool isConvexFaceInOrIntersectingTetrahedron(typename PFP::MAP& map, Face f, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, const typename PFP::VEC3 points[4], bool CCW)
{
	typedef typename PFP::VEC3 VEC3 ;

	Traversor2FV<typename PFP::MAP> tfv(map, f) ;
	for(Vertex v = tfv.begin(); v != tfv.end(); v = tfv.next())
	{
		if(Geom::isPointInTetrahedron(points, position[v], CCW))
			return true;
	}

	VEC3 inter;
	if( intersectionSegmentConvexFace(map, f, position, points[0], points[1], inter)
	|| 	intersectionSegmentConvexFace(map, f, position, points[1], points[2], inter)
	|| 	intersectionSegmentConvexFace(map, f, position, points[2], points[0], inter)
	|| 	intersectionSegmentConvexFace(map, f, position, points[0], points[3], inter)
	|| 	intersectionSegmentConvexFace(map, f, position, points[1], points[3], inter)
	|| 	intersectionSegmentConvexFace(map, f, position, points[2], points[3], inter)
	)
		return true;

	return false;
}

} // namespace Geometry

} // namespace Surface

} // namspace Algo

} // namespace CGoGN
