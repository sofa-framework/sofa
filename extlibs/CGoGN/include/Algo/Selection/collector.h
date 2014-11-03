/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009, IGG Team, LSIIT, University of Strasbourg                *
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
* Web site: http://cgogn.unistra.fr/                                  *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#ifndef __COLLECTOR_H__
#define __COLLECTOR_H__

#include "Container/fakeAttribute.h"
#include "Geometry/basic.h"

/*****************************************
 * Class hierarchy :
 * Collector (virtual)
 * - Collector_WithinSphere
 * - Collector_OneRing
 ****************************************/

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Selection
{

/*********************************************************
 * Generic Collector
 *********************************************************/

template <typename PFP>
class Collector
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

protected:
	MAP& map;
	unsigned int m_thread;

	Dart centerDart;

	bool isInsideCollected;

	std::vector<Vertex> insideVertices;
	std::vector<Edge> insideEdges;
	std::vector<Face> insideFaces;
	std::vector<Dart> border;

public:
	Collector(MAP& m, unsigned int thread = 0);

	inline void init(Dart d)
	{
		centerDart = d;
		isInsideCollected = false;
		insideVertices.clear();
		insideEdges.clear();
		insideFaces.clear();
		border.clear();
	}

	virtual void collectAll(Dart d) = 0;
	virtual void collectBorder(Dart d) = 0;

	bool applyOnInsideVertices(FunctorType& f);
	bool applyOnInsideEdges(FunctorType& f);
	bool applyOnInsideFaces(FunctorType& f);
	bool applyOnBorder(FunctorType& f);

	inline void sort()
	{
		std::sort(insideVertices.begin(), insideVertices.end());
		std::sort(insideEdges.begin(), insideEdges.end());
		std::sort(insideFaces.begin(), insideFaces.end());
		std::sort(border.begin(), border.end());
	}

	inline MAP& getMap() { return map; }

	inline Dart getCenterDart() const { return centerDart; }

	inline const std::vector<Vertex>& getInsideVertices() const { assert(isInsideCollected || !"getInsideVertices: inside cells have not been collected.") ; return insideVertices; }
	inline const std::vector<Edge>& getInsideEdges() const { assert(isInsideCollected || !"getInsideEdges: inside cells have not been collected.") ; return insideEdges; }
	inline const std::vector<Face>& getInsideFaces() const { assert(isInsideCollected || !"getInsideFaces: inside cells have not been collected.") ; return insideFaces; }
	inline const std::vector<Dart>& getBorder() const { return border; }

	inline unsigned int getNbInsideVertices() const { assert(isInsideCollected || !"getNbInsideVertices: inside cells have not been collected.") ; return insideVertices.size(); }
	inline unsigned int getNbInsideEdges() const { assert(isInsideCollected || !"getNbInsideEdges: inside cells have not been collected.") ; return insideEdges.size(); }
	inline unsigned int getNbInsideFaces() const { assert(isInsideCollected || !"getNbInsideFaces: inside cells have not been collected.") ; return insideFaces.size(); }
	inline unsigned int getNbBorder() const { return border.size(); }

	template <typename PPFP>
	friend std::ostream& operator<<(std::ostream &out, const Collector<PPFP>& c);

	virtual REAL computeArea (const VertexAttribute<VEC3, MAP>& /*pos*/)
	{
		assert(!"Warning: Collector<PFP>::computeArea() should be overloaded in non-virtual derived classes");
		return 0.0;
	}
	virtual void computeNormalCyclesTensor (const VertexAttribute<VEC3, MAP>& /*pos*/, const EdgeAttribute<REAL, MAP>& /*edgeangle*/, typename PFP::MATRIX33&)
	{
		assert(!"Warning: Collector<PFP>::computeNormalCyclesTensor() should be overloaded in non-virtual derived classes");
	}
	virtual void computeNormalCyclesTensor (const VertexAttribute<VEC3, MAP>& /*pos*/, typename PFP::MATRIX33&)
	{
		assert(!"Warning: Collector<PFP>::computeNormalCyclesTensor() should be overloaded in non-virtual derived classes");
	}
};

/*********************************************************
 * Collector One Ring
 *********************************************************/

/*
 * insideVertices = centerDart
 * insideEdges = star (edges incident to centerDart)
 * insideFaces = triangles incident to centerDart
 * border = vertices of 1-ring -> link (set of adjacent vertices)
 *        = edges of 1-ring
 */
template <typename PFP>
class Collector_OneRing : public Collector<PFP>
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

public:
	Collector_OneRing(MAP& m, unsigned int thread = 0) :
		Collector<PFP>(m, thread)
	{}
	void collectAll(Dart d);
	void collectBorder(Dart d);

	REAL computeArea(const VertexAttribute<VEC3, MAP>& pos);
	void computeNormalCyclesTensor (const VertexAttribute<VEC3, MAP>& pos, const EdgeAttribute<REAL, MAP>& edgeangle, typename PFP::MATRIX33&);
	void computeNormalCyclesTensor (const VertexAttribute<VEC3, MAP>& pos, typename PFP::MATRIX33&);
};

/*********************************************************
 * Collector One Ring around edge
 *********************************************************/

/*
 * insideVertices = 2 ends of center edge (centerDart)
 * insideEdges = center egde + edges adjacent by a vertex
 *             = union of stars of 1-rings of the 2 ends
 * insideFaces = triangles incident to the 2 ends
 * border = vertices of 1-ring -> link (set of adjacent vertices)
 *        = edges of 1-ring
 */

template <typename PFP>
class Collector_OneRing_AroundEdge : public Collector<PFP>
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

public:
	Collector_OneRing_AroundEdge(MAP& m, unsigned int thread = 0) :
		Collector<PFP>(m, thread)
	{}
	void collectAll(Dart d);
	void collectBorder(Dart d);

	REAL computeArea(const VertexAttribute<VEC3, MAP>& pos);
	void computeNormalCyclesTensor (const VertexAttribute<VEC3, MAP>& pos, const EdgeAttribute<REAL, MAP>& edgeangle, typename PFP::MATRIX33&);
	void computeNormalCyclesTensor (const VertexAttribute<VEC3, MAP>& pos, typename PFP::MATRIX33&);

};

/*********************************************************
 * Collector Within Sphere
 *********************************************************/

/*
 * collect all primitives of the connected component containing "centerDart"
 * within the sphere of radius "radius" and center "position[centerDart]"
 * (hopefully) it defines a 2-manifold (if inserting border-vertices along the border-edges)
 * NB : is equivalent to Collector_Vertices with CollectorCriterion_VertexWithinSphere
 */
template <typename PFP>
class Collector_WithinSphere : public Collector<PFP>
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

protected:
	const VertexAttribute<VEC3, MAP>& position;
	REAL radius;

public:
	Collector_WithinSphere(MAP& m, const VertexAttribute<VEC3, MAP>& p, REAL r = 0, unsigned int thread = 0) :
		Collector<PFP>(m, thread),
		position(p),
		radius(r)
	{}
	inline void setRadius(REAL r) { radius = r; }
	inline REAL getRadius() const { return radius; }
	inline const VertexAttribute<VEC3, MAP>& getPosition() const { return position; }

	void collectAll(Dart d);
	void collectBorder(Dart d);

	REAL computeArea(const VertexAttribute<VEC3, MAP>& pos);
	void computeNormalCyclesTensor (const VertexAttribute<VEC3, MAP>& pos, const EdgeAttribute<REAL, MAP>& edgeangle, typename PFP::MATRIX33&);
	void computeNormalCyclesTensor (const VertexAttribute<VEC3, MAP>& pos, typename PFP::MATRIX33&);
};

/*********************************************************
 * Collector Normal Angle (Vertices)
 *********************************************************/

/*
 * collect all primitives of the connected component containing "centerDart"
 * the angle between the included vertices normal vectors and the central normal vector
 * stays under a given threshold
 * NB : is equivalent to Collector_Vertices with CollectorCriterion_VertexNormalAngle
 */
template <typename PFP>
class Collector_NormalAngle : public Collector<PFP>
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

protected:
	const VertexAttribute<VEC3, MAP>& normal ;
	REAL angleThreshold ;

public:
	Collector_NormalAngle(
		MAP& m,
		const VertexAttribute<VEC3, MAP>& n,
		REAL a,
		unsigned int thread = 0
	) :	Collector<PFP>(m, thread), normal(n), angleThreshold(a)
	{}
	inline void setAngleThreshold(REAL a) { angleThreshold = a; }
	inline REAL getAngleThreshold() const { return angleThreshold; }
	inline const VertexAttribute<VEC3, MAP>& getNormal() const { return normal ; }

	void collectAll(Dart d) ;
	void collectBorder(Dart d) ;
};

/*********************************************************
 * Collector Normal Angle (Triangles)
 *********************************************************/

/*
 * collect all primitives of the connected component containing "centerDart"
 * the angle between the included triangles normal vectors and the central normal vector
 * stays under a given threshold
 * NB : is equivalent to Collector_Triangles with CollectorCriterion_TriangleNormalAngle
 */
template <typename PFP>
class Collector_NormalAngle_Triangles : public Collector<PFP>
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

protected:
	const FaceAttribute<VEC3, MAP>& normal ;
	REAL angleThreshold ;

public:
	Collector_NormalAngle_Triangles(
		MAP& m,
		const FaceAttribute<VEC3, MAP>& n,
		REAL a,
		unsigned int thread = 0
	) :	Collector<PFP>(m, thread), normal(n), angleThreshold(a)
	{}
	inline void setAngleThreshold(REAL a) { angleThreshold = a; }
	inline REAL getAngleThreshold() const { return angleThreshold; }
	inline const VertexAttribute<VEC3, MAP>& getNormal() const { return normal ; }

	void collectAll(Dart d) ;
	void collectBorder(Dart d) ;
};

/*********************************************************
 * Collector Criterions
 *********************************************************/

class CollectorCriterion
{
public :
	CollectorCriterion() {}
	virtual ~CollectorCriterion() {}
	virtual void init(Dart center) = 0;
	virtual bool isInside(Dart d) = 0;
};

// tests if the angle between vertex normals is below some threshold
template <typename PFP>
class CollectorCriterion_VertexNormalAngle : public CollectorCriterion
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

private :
	const VertexAttribute<VEC3, MAP>& vertexNormals;
	REAL threshold;
	VEC3 centerNormal;

public :
	CollectorCriterion_VertexNormalAngle(const VertexAttribute<VEC3, MAP>& n, REAL th) :
		vertexNormals(n), threshold(th), centerNormal(0)
	{}

	void init (Dart center) { centerNormal = vertexNormals[center]; }
	bool isInside (Dart d)
	{
		return ( Geom::angle(centerNormal, vertexNormals[d]) < threshold );
	}
};

// tests if the angle between triangle normals is below some threshold
template <typename PFP>
class CollectorCriterion_TriangleNormalAngle : public CollectorCriterion
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

private :
	const FaceAttribute<VEC3, MAP>& faceNormals;
	REAL threshold;
	VEC3 centerNormal;

public :
	CollectorCriterion_TriangleNormalAngle(const FaceAttribute<VEC3, MAP>& n, REAL th) :
		faceNormals(n), threshold(th), centerNormal(0)
	{}

	void init (Dart center) { centerNormal = faceNormals[center]; }
	bool isInside (Dart d)
	{
		return ( Geom::angle(centerNormal, faceNormals[d]) < threshold );
	}
};

// tests if the distance between vertices is below some threshold
template <typename PFP>
class CollectorCriterion_VertexWithinSphere : public CollectorCriterion
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

private :
	const VertexAttribute<VEC3, MAP>& vertexPositions;
	REAL threshold;
	VEC3 centerPosition;

public :
	CollectorCriterion_VertexWithinSphere(const VertexAttribute<VEC3, MAP>& p, REAL th) :
		vertexPositions(p), threshold(th), centerPosition(0)
	{}

	void init (Dart center) { centerPosition = vertexPositions[center]; }
	bool isInside (Dart d)
	{
		return (vertexPositions[d] - centerPosition).norm() < threshold ;
	}
};

/*********************************************************
 * Collector Vertices
 *********************************************************/

/*
 * collect all vertices of the connected component containing "centerDart"
 * that satisfy the CollectorCriterion
 * (hopefully) it defines a 2-manifold (if inserting border-vertices along the border-edges)
 */
template <typename PFP>
class Collector_Vertices : public Collector<PFP>
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

protected:
	CollectorCriterion & crit;

public:
	Collector_Vertices(typename PFP::MAP& m, CollectorCriterion& c, unsigned int thread = 0) :
		Collector<PFP>(m, thread),
		crit(c)
	{}
	void collectAll(Dart d);
	void collectBorder(Dart d);
};

/*********************************************************
 * Collector Triangles
 *********************************************************/

/*
 * collect all triangles of the connected component containing "centerDart"
 * within a distance to centerDart defined by the CollectorCriterion
 */
template <typename PFP>
class Collector_Triangles : public Collector<PFP>
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

protected:
	CollectorCriterion & crit;

public:
	Collector_Triangles(typename PFP::MAP& m, CollectorCriterion& c, unsigned int thread = 0) :
		Collector<PFP>(m,thread), crit(c)
	{}
	void collectAll(Dart d) ;
	void collectBorder(Dart d) ;
};

/*********************************************************
 * Collector Dijkstra_Vertices
 *********************************************************/

/*
 * collect all primitives of the connected component containing "centerDart"
 * within a distance < maxDist (the shortest path follows edges)
 * the edge length is specified in edge_cost attribute
 */

template <typename PFP>
class Collector_Dijkstra_Vertices : public Collector<PFP>
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

protected:
	const EdgeAttribute<REAL, MAP>& edge_cost;
	REAL maxDist;

	typedef struct
	{
		typename std::multimap<float,Dart>::iterator it ;
		bool valid ;
		static std::string CGoGNnameOfType() { return "DijkstraVertexInfo" ; }
	} DijkstraVertexInfo ;
	typedef NoTypeNameAttribute<DijkstraVertexInfo> VertexInfo ;

	VertexAttribute<VertexInfo, MAP> vertexInfo ;

	std::multimap<float, Dart> front ;

public:
	Collector_Dijkstra_Vertices(MAP& m, const EdgeAttribute<REAL, MAP>& c, REAL d = 0, unsigned int thread = 0) :
		Collector<PFP>(m, thread),
		edge_cost(c),
		maxDist(d)
	{
		vertexInfo = m.template addAttribute<VertexInfo, VERTEX>("vertexInfo");
	}
	~Collector_Dijkstra_Vertices()
	{
		this->map.removeAttribute(vertexInfo);
	}
	inline void init (Dart d) { Collector<PFP>::init(d); front.clear(); }
	inline void setMaxDistance(REAL d) { maxDist = d; }
	inline REAL getMaxDist() const { return maxDist; }

	void collectAll(Dart d);
	void collectBorder(Dart d);
};

/*********************************************************
 * Collector Dijkstra
 *********************************************************/

/*
 * collect all primitives of the connected component containing "centerDart"
 * within a distance < maxDist (the shortest path follows edges)
 */
template <typename PFP>
class Collector_Dijkstra : public Collector<PFP>
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

protected:
	const VertexAttribute<VEC3, MAP>& position;
	REAL maxDist;

	typedef struct
	{
		typename std::multimap<float, Dart>::iterator it ;
		bool valid ;
		static std::string CGoGNnameOfType() { return "DijkstraVertexInfo" ; }
	} DijkstraVertexInfo ;
	typedef NoTypeNameAttribute<DijkstraVertexInfo> VertexInfo ;

	VertexAttribute<VertexInfo, MAP> vertexInfo ;

	std::multimap<float, Dart> front ;

public:
	Collector_Dijkstra(MAP& m, const VertexAttribute<VEC3, MAP>& p, REAL d = 0, unsigned int thread=0) :
		Collector<PFP>(m,thread),
		position(p),
		maxDist(d)
	{
		vertexInfo = m.template addAttribute<VertexInfo, VERTEX>("vertexInfo");
	}
	~Collector_Dijkstra()
	{
		this->map.removeAttribute(vertexInfo);
	}
	inline void init (Dart d) { Collector<PFP>::init(d); front.clear(); }
	inline void setMaxDistance(REAL d) { maxDist = d; }
	inline REAL getMaxDist() const { return maxDist; }
	inline const VertexAttribute<VEC3, MAP>& getPosition() const { return position; }

	void collectAll(Dart d);
	void collectBorder(Dart d);

private :
	inline float edgeLength (Dart d);
//	inline Dart oppositeVertex (Dart d);
};

} // namespace Selection

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/Selection/collector.hpp"

#endif
