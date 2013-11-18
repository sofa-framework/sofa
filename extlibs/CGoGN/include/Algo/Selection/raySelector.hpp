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

#include <algorithm>
#include <set>
#include "Geometry/distances.h"
#include "Geometry/intersection.h"
#include "Algo/Geometry/centroid.h"

namespace CGoGN
{

namespace Algo
{

namespace Selection
{

template <typename R, typename T>
bool distOrdering(const std::pair<R, T>& e1, const std::pair<R, T>& e2)
{
	return (e1.first < e2.first);
}

template <typename PFP>
struct FaceInter
{
	Dart f;
	typename PFP::VEC3 i;
	FaceInter(Dart d, const typename PFP::VEC3& v) : f(d), i(v) {}
	FaceInter() {}
};

/**
 * Function that does the selection of faces, returned darts and intersection points are sorted from closest to farthest
 * @param map the map we want to test
 * @param position the vertex attribute storing positions
 * @param rayA first point of ray (user side)
 * @param rayAB direction of ray (directed to the scene)
 * @param vecFaces (out) vector to store the darts of intersected faces
 * @param iPoints (out) vector to store the intersection points
 */
template<typename PFP>
void facesRaySelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		std::vector<Dart>& vecFaces,
		std::vector<typename PFP::VEC3>& iPoints)
{
	vecFaces.reserve(256);
	iPoints.reserve(256);
	vecFaces.clear();
	iPoints.clear();

	TraversorF<typename PFP::MAP> trav(map);
	for(Dart d = trav.begin(); d!=trav.end(); d = trav.next())
	{
		const typename PFP::VEC3& Ta = position[d];
		Dart dd  = map.phi1(d);
		Dart ddd = map.phi1(dd);
		bool notfound = true;
		do
		{
			// get position of triangle Ta,Tb,Tc
			const typename PFP::VEC3& Tb = position[dd];
			const typename PFP::VEC3& Tc = position[ddd];
			typename PFP::VEC3 I;
			if (Geom::intersectionRayTriangleOpt<typename PFP::VEC3>(rayA, rayAB, Ta, Tb, Tc, I))
			{
				vecFaces.push_back(d);
				iPoints.push_back(I);
				notfound = false;
			}
			// next triangle if we are in polygon
			dd = ddd;
			ddd = map.phi1(dd);
		} while ((ddd != d) && notfound);
	}

	if(vecFaces.size() > 0)
	{
		// compute all distances to observer for each intersected face
		// and put them in a vector for sorting
		typedef std::pair<typename PFP::REAL, FaceInter<PFP> > faceInterDist;
		std::vector<faceInterDist> dist;

		unsigned int nbi = vecFaces.size();
		dist.resize(nbi);
		for (unsigned int i = 0; i < nbi; ++i)
		{
			dist[i].first = (iPoints[i] - rayA).norm2();
			dist[i].second = FaceInter<PFP>(vecFaces[i], iPoints[i]);
		}

		// sort the vector of pair dist/dart
		std::sort(dist.begin(), dist.end(), distOrdering<typename PFP::REAL, FaceInter<PFP> >);

		// store result in returned vectors
		for (unsigned int i = 0; i < nbi; ++i)
		{
			vecFaces[i] = dist[i].second.f;
			iPoints[i] = dist[i].second.i;
		}
	}
}

/**
 * Function that does the selection of faces, returned darts are sorted from closest to farthest
 * @param map the map we want to test
 * @param position the vertex attribute storing positions
 * @param rayA first point of ray (user side)
 * @param rayAB direction of ray (directed to the scene)
 * @param vecFaces (out) vector to store the darts of intersected faces
 */
template<typename PFP>
void facesRaySelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		std::vector<Dart>& vecFaces)
{
	std::vector<typename PFP::VEC3> iPoints;
	facesRaySelection<PFP>(map, position, rayA, rayAB, vecFaces, iPoints);
}

/**
 * Function that does the selection of one face
 * @param map the map we want to test
 * @param rayA first point of  ray (user side)
 * @param rayAB vector of ray (directed ot the scene)
 * @param face (out) dart of intersected edge (set to NIL if no face selected)
 */
template<typename PFP>
void faceRaySelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		Dart& face)
{
	if (map.dimension() > 2)
		CGoGNerr << "faceRaySelection only on map of dimension 2" << CGoGNendl;

	std::vector<Dart> vecFaces;
	std::vector<typename PFP::VEC3> iPoints;

	facesRaySelection<PFP>(map, position, rayA, rayAB, vecFaces, iPoints);

	if(vecFaces.size() > 0)
		face = vecFaces[0];
	else
		face = NIL;
}

/**
 * Function that does the selection of edges, returned darts are sorted from closest to farthest
 * @param map the map we want to test
 * @param rayA first point of  ray (user side)
 * @param rayAB vector of ray (directed ot the scene)
 * @param vecEdges (out) vector to store dart of intersected edges
 * @param dist radius of the cylinder of selection
 */
template<typename PFP>
void edgesRaySelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		std::vector<Dart>& vecEdges,
		float distMax)
{
	typename PFP::REAL dist2 = distMax * distMax;
	typename PFP::REAL AB2 = rayAB * rayAB;

	// recuperation des aretes intersectees
	vecEdges.reserve(256);
	vecEdges.clear();

	TraversorE<typename PFP::MAP> trav(map);
	for(Dart d = trav.begin(); d!=trav.end(); d = trav.next())
	{
		// get back position of segment PQ
		const typename PFP::VEC3& P = position[d];
		const typename PFP::VEC3& Q = position[map.phi1(d)];
		// the three distance to P, Q and (PQ) not used here
		float ld2 = Geom::squaredDistanceLine2Seg(rayA, rayAB, AB2, P, Q);
		if (ld2 < dist2)
			vecEdges.push_back(d);
	}

	if(vecEdges.size() > 0)
	{
		typedef std::pair<typename PFP::REAL, Dart> DartDist;
		std::vector<DartDist> distndart;

		unsigned int nbi = vecEdges.size();
		distndart.resize(nbi);

		// compute all distances to observer for each middle of intersected edge
		// and put them in a vector for sorting
		for (unsigned int i = 0; i < nbi; ++i)
		{
			Dart d = vecEdges[i];
			distndart[i].second = d;
			typename PFP::VEC3 V = (position[d] + position[map.phi1(d)]) / typename PFP::REAL(2);
			V -= rayA;
			distndart[i].first = V.norm2();
		}

		// sort the vector of pair dist/dart
		std::sort(distndart.begin(), distndart.end(), distOrdering<typename PFP::REAL, Dart>);

		// store sorted darts in returned vector
		for (unsigned int i = 0; i < nbi; ++i)
			vecEdges[i] = distndart[i].second;
	}
}

/**
 * Function that does the selection of one edge
 * @param map the map we want to test
 * @param rayA first point of  ray (user side)
 * @param rayAB vector of ray (directed ot the scene)
 * @param edge (out) dart of intersected edge (set to NIL if no edge selected)
 */
template<typename PFP>
void edgeRaySelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		Dart& edge)
{
	if (map.dimension() > 2)
		CGoGNerr << "edgeRaySelection only on map of dimension 2" << CGoGNendl;

	std::vector<Dart> vecFaces;
	std::vector<typename PFP::VEC3> iPoints;

	facesRaySelection<PFP>(map, position, rayA, rayAB, vecFaces, iPoints);

	if(vecFaces.size() > 0)
	{
		// recuperation du point d'intersection sur la face la plus proche
		typename PFP::VEC3 ip = iPoints[0];

		// recuperation de l'arete la plus proche du point d'intersection
		Dart d = vecFaces[0];
		Dart it = d;
		typename PFP::REAL minDist = squaredDistanceLine2Point(position[it], position[map.phi1(it)], ip);
		edge = it;
		it = map.phi1(it);
		while(it != d)
		{
			typename PFP::REAL dist = squaredDistanceLine2Point(position[it], position[map.phi1(it)], ip);
			if(dist < minDist)
			{
				minDist = dist;
				edge = it;
			}
			it = map.phi1(it);
		}
	}
	else
		edge = NIL;
}

/**
 * Function that does the selection of vertices, returned darts are sorted from closest to farthest
 * @param map the map we want to test
 * @param rayA first point of  ray (user side)
 * @param rayAB vector of ray (directed ot the scene)
 * @param vecVertices (out) vector to store dart of intersected vertices
 * @param dist radius of the cylinder of selection
 */
template<typename PFP>
void verticesRaySelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		std::vector<Dart>& vecVertices,
		float dist)
{
	typename PFP::REAL dist2 = dist * dist;
	typename PFP::REAL AB2 = rayAB * rayAB;

	// recuperation des sommets intersectes
	vecVertices.reserve(256);
	vecVertices.clear();

	TraversorV<typename PFP::MAP> trav(map);
	for(Dart d = trav.begin(); d!=trav.end(); d = trav.next())
	{
		const typename PFP::VEC3& P = position[d];
		float ld2 = Geom::squaredDistanceLine2Point(rayA, rayAB, AB2, P);
		if (ld2 < dist2)
			vecVertices.push_back(d);
	}

	if(vecVertices.size() > 0)
	{
		typedef std::pair<typename PFP::REAL, Dart> DartDist;
		std::vector<DartDist> distndart;

		unsigned int nbi = vecVertices.size();
		distndart.resize(nbi);

		// compute all distances to observer for each intersected vertex
		// and put them in a vector for sorting
		for (unsigned int i = 0; i < nbi; ++i)
		{
			Dart d = vecVertices[i];
			distndart[i].second = d;
			typename PFP::VEC3 V = position[d] - rayA;
			distndart[i].first = V.norm2();
		}

		// sort the vector of pair dist/dart
		std::sort(distndart.begin(), distndart.end(), distOrdering<typename PFP::REAL, Dart>);

		// store sorted darts in returned vector
		for (unsigned int i = 0; i < nbi; ++i)
			vecVertices[i] = distndart[i].second;
	}
}

/**
 * Function that does the selection of one vertex
 * @param map the map we want to test
 * @param rayA first point of  ray (user side)
 * @param rayAB vector of ray (directed ot the scene)
 * @param vertex (out) dart of selected vertex (set to NIL if no vertex selected)
 */
template<typename PFP>
void vertexRaySelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		Dart& vertex)
{
	if (map.dimension() > 2)
		CGoGNerr << "vertexRaySelection only on map of dimension 2" << CGoGNendl;

	std::vector<Dart> vecFaces;
	std::vector<typename PFP::VEC3> iPoints;

	facesRaySelection<PFP>(map, position, rayA, rayAB, vecFaces, iPoints);

	if(vecFaces.size() > 0)
	{
		// recuperation du point d'intersection sur la face la plus proche
		typename PFP::VEC3 ip = iPoints[0];

		// recuperation du sommet le plus proche du point d'intersection
		Dart d = vecFaces[0];
		Dart it = d;
		typename PFP::REAL minDist = (ip - position[it]).norm2();
		vertex = it;
		it = map.phi1(it);
		while(it != d)
		{
			typename PFP::REAL dist = (ip - position[it]).norm2();
			if(dist < minDist)
			{
				minDist = dist;
				vertex = it;
			}
			it = map.phi1(it);
		}
	}
	else
		vertex = NIL;
}

template<typename PFP>
void volumesRaySelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		std::vector<Dart>& vecVolumes)
{
	std::vector<Dart> vecFaces;
	std::vector<typename PFP::VEC3> iPoints;

	// get back intersected faces
	vecFaces.reserve(256);
	iPoints.reserve(256);

	TraversorF<typename PFP::MAP> trav(map);
	for(Dart d = trav.begin(); d != trav.end(); d = trav.next())
	{
		const typename PFP::VEC3& Ta = position[d];
		Dart dd  = map.phi1(d);
		Dart ddd = map.phi1(dd);
		bool notfound = true;
		do
		{
			// get back position of triangle Ta,Tb,Tc
			const typename PFP::VEC3& Tb = position[dd];
			const typename PFP::VEC3& Tc = position[ddd];
			typename PFP::VEC3 I;
			if (Geom::intersectionRayTriangleOpt<typename PFP::VEC3>(rayA, rayAB, Ta, Tb, Tc, I))
			{
				vecFaces.push_back(d);
				iPoints.push_back(I);
				notfound = false;
			}
			// next triangle if we are in polygon
			dd = ddd;
			ddd = map.phi1(dd);
		} while ((ddd != d) && notfound);
	}

	// compute all distances to observer for each intersected face
	// and put them in a vector for sorting
	typedef std::pair<typename PFP::REAL, Dart> DartDist;
	std::vector<DartDist> distndart;

	unsigned int nbi = vecFaces.size();
	distndart.resize(nbi);
	for (unsigned int i = 0; i < nbi; ++i)
	{
		distndart[i].second = vecFaces[i];
		typename PFP::VEC3 V = iPoints[i] - rayA;
		distndart[i].first = V.norm2();
	}

	// sort the vector of pair dist/dart
	std::sort(distndart.begin(), distndart.end(), distOrdering<typename PFP::REAL, Dart>);

//TODO
//	think of how sort volumes from faces order
//
}


/**
 * Fonction that does the selection of darts, returned darts are sorted from closest to farthest
 * Dart is here considered as a triangle formed by the 2 end vertices of the edge and the face centroid
 * @param map the map we want to test
 * @param rayA first point of  ray (user side)
 * @param rayAB vector of ray (directed ot the scene)
 * @param vecDarts (out) vector to store dart of intersected darts
 */
//template<typename PFP>
//void dartsRaySelection(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, const typename PFP::VEC3& rayA, const typename PFP::VEC3& rayAB, std::vector<Dart>& vecDarts)
//{
//	// recuperation des brins intersectes
//	vecDarts.clear();
//	Algo::Selection::FuncDartMapD2Inter<PFP> ffi(map, position, vecDarts, rayA, rayAB);
//	map.template foreach_orbit<FACE>(ffi);
//
//	typedef std::pair<typename PFP::REAL, Dart> DartDist;
//	std::vector<DartDist> distndart;
//
//	unsigned int nbi = vecDarts.size();
//	distndart.resize(nbi);
//
//	// compute all distances to observer for each dart of middle of edge
//	// and put them in a vector for sorting
//	for (unsigned int i = 0; i < nbi; ++i)
//	{
//		Dart d = vecDarts[i];
//		distndart[i].second = d;
//		typename PFP::VEC3 V = (position[d] + position[map.phi1(d)]) / typename PFP::REAL(2);
//		V -= rayA;
//		distndart[i].first = V.norm2();
//	}
//
//	// sort the vector of pair dist/dart
//	std::sort(distndart.begin(), distndart.end(), distndartOrdering<PFP>);
//
//	// store sorted darts in returned vector
//	for (unsigned int i=0; i< nbi; ++i)
//		vecDarts[i] = distndart[i].second;
//}

template<typename PFP>
void facesPlanSelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3>& position,
		const typename Geom::Plane3D<typename PFP::VEC3::DATA_TYPE>& plan,
		std::vector<Dart>& vecDarts)
{
	TraversorF<typename PFP::MAP> travF(map);

	for(Dart dit = travF.begin() ; dit != travF.end() ; dit = travF.next() )
	{
		if(Geom::intersectionTrianglePlan<typename PFP::VEC3>(position[dit], position[map.phi1(dit)], position[map.phi_1(dit)],plan.d(), plan.normal()) == Geom::FACE_INTERSECTION)
		{
			vecDarts.push_back(dit);
		}
	}

	std::cout << "nb faces = " << vecDarts.size() << std::endl;
}



/**
 * Function that does the selection of vertices in a cone, returned darts are sorted from closest to farthest
 * @param map the map we want to test
 * @param rayA first point of  ray (user side)
 * @param rayAB vector of ray (directed ot the scene)
 * @param vecVertices (out) vector to store dart of intersected vertices
 * @param angle angle of the code in degree.
 */
template<typename PFP>
void verticesConeSelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		float angle,
		std::vector<Dart>& vecVertices)
{
	typename PFP::REAL AB2 = rayAB * rayAB;

	double sin2 = sin(M_PI/180.0 * angle);
	sin2 = sin2*sin2;

	// recuperation des sommets intersectes
	vecVertices.reserve(256);
	vecVertices.clear();

	TraversorV<typename PFP::MAP> trav(map);
	for(Dart d = trav.begin(); d!=trav.end(); d = trav.next())
	{
		const typename PFP::VEC3& P = position[d];
		float ld2 = Geom::squaredDistanceLine2Point(rayA, rayAB, AB2, P);
		typename PFP::VEC3 V = P - rayA;
		double s2 = double(ld2) / double(V*V);
		if (s2 < sin2)
			vecVertices.push_back(d);
	}

	typedef std::pair<typename PFP::REAL, Dart> DartDist;
	std::vector<DartDist> distndart;

	unsigned int nbi = vecVertices.size();
	distndart.resize(nbi);

	// compute all distances to observer for each intersected vertex
	// and put them in a vector for sorting
	for (unsigned int i = 0; i < nbi; ++i)
	{
		Dart d = vecVertices[i];
		distndart[i].second = d;
		typename PFP::VEC3 V = position[d] - rayA;
		distndart[i].first = V.norm2();
	}

	// sort the vector of pair dist/dart
	std::sort(distndart.begin(), distndart.end(), distOrdering<typename PFP::REAL, Dart>);

	// store sorted darts in returned vector
	for (unsigned int i = 0; i < nbi; ++i)
		vecVertices[i] = distndart[i].second;
}

/**
 * Function that does the selection of edges, returned darts are sorted from closest to farthest
 * @param map the map we want to test
 * @param rayA first point of  ray (user side)
 * @param rayAB vector of ray (directed ot the scene)
 * @param vecEdges (out) vector to store dart of intersected edges
 * @param angle radius of the cylinder of selection
 */
template<typename PFP>
void edgesConeSelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3>& position,
		const typename PFP::VEC3& rayA,
		const typename PFP::VEC3& rayAB,
		float angle,
		std::vector<Dart>& vecEdges)
{
	typename PFP::REAL AB2 = rayAB * rayAB;

	double sin2 = sin(M_PI/180.0 * angle);
	sin2 = sin2*sin2;

	// recuperation des aretes intersectees
	vecEdges.reserve(256);
	vecEdges.clear();

	TraversorE<typename PFP::MAP> trav(map);
	for(Dart d = trav.begin(); d!=trav.end(); d = trav.next())
	{
		// get back position of segment PQ
		const typename PFP::VEC3& P = position[d];
		const typename PFP::VEC3& Q = position[map.phi1(d)];
		// the three distance to P, Q and (PQ) not used here
		float ld2 = Geom::squaredDistanceLine2Seg(rayA, rayAB, AB2, P, Q);
		typename PFP::VEC3 V = (P+Q)/2.0f - rayA;
		double s2 = double(ld2) / double(V*V);
		if (s2 < sin2)
			vecEdges.push_back(d);
	}

	typedef std::pair<typename PFP::REAL, Dart> DartDist;
	std::vector<DartDist> distndart;

	unsigned int nbi = vecEdges.size();
	distndart.resize(nbi);

	// compute all distances to observer for each middle of intersected edge
	// and put them in a vector for sorting
	for (unsigned int i = 0; i < nbi; ++i)
	{
		Dart d = vecEdges[i];
		distndart[i].second = d;
		typename PFP::VEC3 V = (position[d] + position[map.phi1(d)]) / typename PFP::REAL(2);
		V -= rayA;
		distndart[i].first = V.norm2();
	}

	// sort the vector of pair dist/dart
	std::sort(distndart.begin(), distndart.end(), distOrdering<typename PFP::REAL, Dart>);

	// store sorted darts in returned vector
	for (unsigned int i = 0; i < nbi; ++i)
		vecEdges[i] = distndart[i].second;
}



template<typename PFP>
Dart verticesBubbleSelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3>& position,
		const typename PFP::VEC3& cursor,
		typename PFP::REAL radiusMax)
{
	typename PFP::REAL l2max = radiusMax*radiusMax;
	typename PFP::REAL l2min(std::numeric_limits<float>::max());
	Dart d_min=NIL;

	TraversorV<typename PFP::MAP> trav(map);
	for(Dart d = trav.begin(); d!=trav.end(); d = trav.next())
	{
		const typename PFP::VEC3& P = position[d];
		typename PFP::VEC3 V = P - cursor;
		typename PFP::REAL l2 = V*V;
		if ((l2<l2max) && (l2<l2min))
		{
			d_min=d;
		}
	}
	return d_min;
}

template<typename PFP>
Dart edgesBubbleSelection(
		typename PFP::MAP& map,
		const VertexAttribute<typename PFP::VEC3>& position,
		const typename PFP::VEC3& cursor,
		typename PFP::REAL radiusMax)
{
	typename PFP::REAL l2max = radiusMax*radiusMax;
	typename PFP::REAL l2min(std::numeric_limits<float>::max());
	Dart d_min=NIL;

	TraversorE<typename PFP::MAP> trav(map);
	for(Dart d = trav.begin(); d!=trav.end(); d = trav.next())
	{
		const typename PFP::VEC3& A = position[d];
		typename PFP::VEC3 AB =  position[map.phi1(d)] - A;
		typename PFP::REAL l2 = Geom::squaredDistanceSeg2Point(A,AB,AB*AB,cursor);
		if ((l2<l2max) && (l2<l2min))
		{
			d_min=d;
		}
	}
	return d_min;
}





//
//namespace Parallel
//{
//
//template<typename PFP>
//void facesRaySelection(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, const typename PFP::VEC3& rayA, const typename PFP::VEC3& rayAB, std::vector<Dart>& vecFaces, unsigned int nbth=0, unsigned int current_thread=0)
//{
//	if (nbth==0)
////		nbth = Algo::Parallel::optimalNbThreads();
//		nbth =2;	// seems to be optimal ?
//
//	std::vector<FunctorMapThreaded<typename PFP::MAP>*> functs;
//	for (unsigned int i=0; i < nbth; ++i)
//		functs.push_back(new Parallel::FuncFaceInter<PFP>(map,position,rayA, rayAB));
//
//	Algo::Parallel::foreach_cell<typename PFP::MAP,FACE>(map, functs, false, current_thread);
//
//
//	// compute total nb of intersection
//	unsigned int nbtot=0;
//	for (unsigned int i=0; i < nbth; ++i)
//		nbtot += static_cast<Parallel::FuncFaceInter<PFP>*>(functs[i])->getFaceDistances().size();
//
//	std::vector<std::pair<typename PFP::REAL, Dart> > distndart;
//	distndart.reserve(nbtot);
//	for (unsigned int i=0; i < nbth; ++i)
//	{
//		distndart.insert(distndart.end(),static_cast<Parallel::FuncFaceInter<PFP>*>(functs[i])->getFaceDistances().begin(), static_cast<Parallel::FuncFaceInter<PFP>*>(functs[i])->getFaceDistances().end() );
//		delete functs[i];
//	}
//
//	// sort the vector of pair dist/dart
//	std::sort(distndart.begin(), distndart.end(), distndartOrdering<PFP>);
//
//	vecFaces.clear();
//	vecFaces.reserve(nbtot);
//	// store sorted darts in returned vector
//	for (unsigned int i = 0; i < nbtot; ++i)
//		vecFaces.push_back(distndart[i].second);
//}
//
//template<typename PFP>
//void edgesRaySelection(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, const typename PFP::VEC3& rayA, const typename PFP::VEC3& rayAB, std::vector<Dart>& vecEdges, float dist, unsigned int nbth=0, unsigned int current_thread=0)
//{
//	typename PFP::REAL dist2 = dist * dist;
//	typename PFP::REAL AB2 = rayAB * rayAB;
//
//	if (nbth==0)
////		nbth = Algo::Parallel::optimalNbThreads();
//		nbth =2;	// seems to be optimal ?
//
//	std::vector<FunctorMapThreaded<typename PFP::MAP>*> functs;
//	for (unsigned int i=0; i < nbth; ++i)
//		functs.push_back(new Parallel::FuncEdgeInter<PFP>(map,position,rayA, rayAB, AB2, dist2));
//
//	Algo::Parallel::foreach_cell<typename PFP::MAP,EDGE>(map, functs, false, current_thread);
//
//	// compute total nb of intersection
//	unsigned int nbtot=0;
//	for (unsigned int i=0; i < nbth; ++i)
//		nbtot += static_cast<Parallel::FuncEdgeInter<PFP>*>(functs[i])->getEdgeDistances().size();
//
//	std::vector<std::pair<typename PFP::REAL, Dart> > distndart;
//	distndart.reserve(nbtot);
//	for (unsigned int i=0; i < nbth; ++i)
//	{
//		distndart.insert(distndart.end(),static_cast<Parallel::FuncEdgeInter<PFP>*>(functs[i])->getEdgeDistances().begin(), static_cast<Parallel::FuncEdgeInter<PFP>*>(functs[i])->getEdgeDistances().end() );
//		delete functs[i];
//	}
//
//	// sort the vector of pair dist/dart
//	std::sort(distndart.begin(), distndart.end(), distndartOrdering<PFP>);
//
//	// store sorted darts in returned vector
//	vecEdges.clear();
//	vecEdges.reserve(nbtot);
//	for (unsigned int i = 0; i < nbtot; ++i)
//		vecEdges.push_back(distndart[i].second);
//}
//
//
//template<typename PFP>
//void verticesRaySelection(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, const typename PFP::VEC3& rayA, const typename PFP::VEC3& rayAB, std::vector<Dart>& vecVertices, float dist , unsigned int nbth=0, unsigned int current_thread=0)
//{
//	typename PFP::REAL dist2 = dist * dist;
//	typename PFP::REAL AB2 = rayAB * rayAB;
//
//	if (nbth==0)
////		nbth = Algo::Parallel::optimalNbThreads();
//		nbth =2;	// seems to be optimal ?
//
//	std::vector<FunctorMapThreaded<typename PFP::MAP>*> functs;
//	for (unsigned int i=0; i < nbth; ++i)
//		functs.push_back(new Parallel::FuncVertexInter<PFP>(map,position,rayA, rayAB, AB2, dist2));
//
//	Algo::Parallel::foreach_cell<typename PFP::MAP,VERTEX>(map, functs, false, current_thread);
//
//	// compute total nb of intersection
//	unsigned int nbtot=0;
//	for (unsigned int i=0; i < nbth; ++i)
//		nbtot += static_cast<Parallel::FuncVertexInter<PFP>*>(functs[i])->getVertexDistances().size();
//
//	std::vector<std::pair<typename PFP::REAL, Dart> > distndart;
//	distndart.reserve(nbtot);
//	for (unsigned int i=0; i < nbth; ++i)
//	{
//		distndart.insert(distndart.end(),static_cast<Parallel::FuncVertexInter<PFP>*>(functs[i])->getVertexDistances().begin(), static_cast<Parallel::FuncVertexInter<PFP>*>(functs[i])->getVertexDistances().end() );
//		delete functs[i];
//	}
//
//	// sort the vector of pair dist/dart
//	std::sort(distndart.begin(), distndart.end(), distndartOrdering<PFP>);
//
//	// store sorted darts in returned vector
//	vecVertices.clear();
//	vecVertices.reserve(nbtot);
//	for (unsigned int i = 0; i < nbtot; ++i)
//		vecVertices.push_back(distndart[i].second);
//
//
//}
//
//template<typename PFP>
//void vertexRaySelection(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, const typename PFP::VEC3& rayA, const typename PFP::VEC3& rayAB, Dart& vertex, unsigned int nbth=0, unsigned int current_thread=0)
//{
//	std::vector<Dart> vecFaces;
//	vecFaces.reserve(100);
//	Parallel::facesRaySelection<PFP>(map, position, rayA, rayAB, vecFaces, nbth, current_thread);
//
//	if(vecFaces.size() > 0)
//	{
//		// recuperation du sommet le plus proche
//		Dart d = vecFaces.front();
//		Dart it = d;
//		typename PFP::REAL minDist = (rayA - position[it]).norm2();
//		vertex = it;
//		it = map.phi1(it);
//		while(it != d)
//		{
//			typename PFP::REAL dist = (rayA - position[it]).norm2();
//			if(dist < minDist)
//			{
//				minDist = dist;
//				vertex = it;
//			}
//			it = map.phi1(it);
//		}
//	}
//	else
//		vertex = NIL;
//}
//
//}

} //namespace Selection

} //namespace Algo

} //namespace CGoGN

