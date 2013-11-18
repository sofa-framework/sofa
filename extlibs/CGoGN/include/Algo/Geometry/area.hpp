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

#include "Geometry/basic.h"
#include "Algo/Geometry/centroid.h"
#include "Topology/generic/traversorCell.h"
#include "Topology/generic/traversor2.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

template <typename PFP>
typename PFP::REAL triangleArea(typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3>& position)
{
	typename PFP::VEC3 p1 = position[d] ;
	typename PFP::VEC3 p2 = position[map.phi1(d)] ;
	typename PFP::VEC3 p3 = position[map.phi_1(d)] ;

	return Geom::triangleArea(p1, p2, p3) ;
}

template <typename PFP>
typename PFP::REAL convexFaceArea(typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3>& position)
{
	typedef typename PFP::VEC3 VEC3 ;

	if(map.faceDegree(d) == 3)
		return triangleArea<PFP>(map, d, position) ;
	else
	{
		float area = 0.0f ;
		VEC3 centroid = faceCentroid<PFP>(map, d, position) ;
		Traversor2FE<typename PFP::MAP> t(map, d) ;
		for(Dart it = t.begin(); it != t.end(); it = t.next())
		{
			VEC3 p1 = position[it] ;
			VEC3 p2 = position[map.phi1(it)] ;
			area += Geom::triangleArea(p1, p2, centroid) ;
		}
		return area ;
	}
}

template <typename PFP>
typename PFP::REAL totalArea(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, unsigned int thread)
{
	typename PFP::REAL area(0) ;
	TraversorF<typename PFP::MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		area += convexFaceArea<PFP>(map, d, position) ;
	return area ;
}

template <typename PFP>
typename PFP::REAL vertexOneRingArea(typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3>& position)
{
	typename PFP::REAL area(0) ;
	Traversor2VF<typename PFP::MAP> t(map, d) ;
	for(Dart it = t.begin(); it != t.end(); it = t.next())
		area += convexFaceArea<PFP>(map, it, position) ;
	return area ;
}

template <typename PFP>
typename PFP::REAL vertexBarycentricArea(typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3>& position)
{
	typename PFP::REAL area(0) ;
	Traversor2VF<typename PFP::MAP> t(map, d) ;
	for(Dart it = t.begin(); it != t.end(); it = t.next())
		area += convexFaceArea<PFP>(map, it, position) / 3 ;
	return area ;
}

template <typename PFP>
typename PFP::REAL vertexVoronoiArea(typename PFP::MAP& map, Dart d, const VertexAttribute<typename PFP::VEC3>& position)
{
	typename PFP::REAL area(0) ;
	Traversor2VF<typename PFP::MAP> t(map, d) ;
	for(Dart it = t.begin(); it != t.end(); it = t.next())
	{
		const typename PFP::VEC3& p1 = position[it] ;
		const typename PFP::VEC3& p2 = position[map.phi1(it)] ;
		const typename PFP::VEC3& p3 = position[map.phi_1(it)] ;
		if(!Geom::isTriangleObtuse(p1, p2, p3))
		{
			typename PFP::REAL a = Geom::angle(p3 - p2, p1 - p2) ;
			typename PFP::REAL b = Geom::angle(p1 - p3, p2 - p3) ;
			area += ( (p2 - p1).norm2() / tan(b) + (p3 - p1).norm2() / tan(a) ) / 8 ;
		}
		else
		{
			typename PFP::REAL tArea = Geom::triangleArea(p1, p2, p3) ;
			if(Geom::angle(p2 - p1, p3 - p1) > M_PI / 2)
				area += tArea / 2 ;
			else
				area += tArea / 4 ;
		}
	}
	return area ;
}

template <typename PFP>
void computeAreaFaces(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, FaceAttribute<typename PFP::REAL>& face_area)
{
	TraversorF<typename PFP::MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		face_area[d] = convexFaceArea<PFP>(map, d, position) ;
}

template <typename PFP>
void computeOneRingAreaVertices(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, VertexAttribute<typename PFP::REAL>& vertex_area)
{
	TraversorV<typename PFP::MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		vertex_area[d] = vertexOneRingArea<PFP>(map, d, position) ;
}

template <typename PFP>
void computeBarycentricAreaVertices(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, VertexAttribute<typename PFP::REAL>& vertex_area)
{
	TraversorV<typename PFP::MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		vertex_area[d] = vertexBarycentricArea<PFP>(map, d, position) ;
}

template <typename PFP>
void computeVoronoiAreaVertices(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, VertexAttribute<typename PFP::REAL>& vertex_area)
{
	TraversorV<typename PFP::MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
		vertex_area[d] = vertexVoronoiArea<PFP>(map, d, position) ;
}


namespace Parallel
{
template <typename PFP>
class FunctorConvexFaceArea: public FunctorMapThreaded<typename PFP::MAP >
{
	 const VertexAttribute<typename PFP::VEC3>& m_position;
	 FaceAttribute<typename PFP::REAL>& m_area;
public:
	 FunctorConvexFaceArea<PFP>( typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, FaceAttribute<typename PFP::REAL>& area):
	 	 FunctorMapThreaded<typename PFP::MAP>(map), m_position(position), m_area(area)
	 { }

	void run(Dart d, unsigned int threadID)
	{
		m_area[d] = convexFaceArea<PFP>(this->m_map, d, m_position) ;
	}
};

template <typename PFP>
void computeAreaFaces(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, FaceAttribute<typename PFP::REAL>& area, unsigned int nbth)
{
	FunctorConvexFaceArea<PFP> funct(map,position,area);
	Algo::Parallel::foreach_cell<typename PFP::MAP,FACE>(map, funct, nbth, false);
}


template <typename PFP>
class FunctorVertexOneRingArea: public FunctorMapThreaded<typename PFP::MAP >
{
	 const VertexAttribute<typename PFP::VEC3>& m_position;
	 VertexAttribute<typename PFP::REAL>& m_area;
public:
	 FunctorVertexOneRingArea<PFP>( typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, VertexAttribute<typename PFP::REAL>& area):
	 	 FunctorMapThreaded<typename PFP::MAP>(map), m_position(position), m_area(area)
	 { }

	void run(Dart d, unsigned int threadID)
	{
		m_area[d] = vertexOneRingArea<PFP>(this->m_map, d, m_position) ;
	}
};

template <typename PFP>
void computeOneRingAreaVertices(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, VertexAttribute<typename PFP::REAL>& area, unsigned int nbth)
{
	FunctorConvexFaceArea<PFP> funct(map,position,area);
	Algo::Parallel::foreach_cell<typename PFP::MAP,VERTEX>(map, funct, nbth, false);
}



template <typename PFP>
class FunctorVertexVoronoiArea: public FunctorMapThreaded<typename PFP::MAP >
{
	 const VertexAttribute<typename PFP::VEC3>& m_position;
	 VertexAttribute<typename PFP::REAL>& m_area;
public:
	 FunctorVertexVoronoiArea<PFP>( typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, VertexAttribute<typename PFP::REAL>& area):
	 	 FunctorMapThreaded<typename PFP::MAP>(map), m_position(position), m_area(area)
	 { }

	void run(Dart d, unsigned int threadID)
	{
		m_area[d] = vertexVoronoiArea<PFP>(this->m_map, d, m_position) ;
	}
};

template <typename PFP>
void computeVoronoiAreaVertices(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& position, VertexAttribute<typename PFP::REAL>& area, unsigned int nbth)
{
	FunctorConvexFaceArea<PFP> funct(map,position,area);
	Algo::Parallel::foreach_cell<typename PFP::MAP,VERTEX>(map, funct, nbth, false);
}


}







} // namespace Geometry

}

} // namespace Algo

} // namespace CGoGN
