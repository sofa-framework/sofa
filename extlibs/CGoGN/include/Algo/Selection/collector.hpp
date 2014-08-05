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

#include "Topology/generic/traversor/traversor2.h"
#include "Algo/Geometry/intersection.h"
#include <queue>

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
Collector<PFP>::Collector(MAP& m, unsigned int thread) : map(m), m_thread(thread), isInsideCollected(false)
{}

template <typename PFP>
inline bool Collector<PFP>::applyOnInsideVertices(FunctorType& f)
{
	assert(isInsideCollected || !"applyOnInsideVertices: inside cells have not been collected.") ;
	for(std::vector<Vertex>::iterator iv = insideVertices.begin(); iv != insideVertices.end(); ++iv)
		if(f((*iv).dart))
			return true ;
	return false ;
}

template <typename PFP>
inline bool Collector<PFP>::applyOnInsideEdges(FunctorType& f)
{
	assert(isInsideCollected || !"applyOnInsideEdges: inside cells have not been collected.") ;
	for(std::vector<Edge>::iterator iv = insideEdges.begin(); iv != insideEdges.end(); ++iv)
		if(f((*iv).dart))
			return true ;
	return false ;
}

template <typename PFP>
inline bool Collector<PFP>::applyOnInsideFaces(FunctorType& f)
{
	assert(isInsideCollected || !"applyOnInsideFaces: inside cells have not been collected.") ;
	for(std::vector<Face>::iterator iv = insideFaces.begin(); iv != insideFaces.end(); ++iv)
		if(f((*iv).dart))
			return true ;
	return false ;
}

template <typename PFP>
inline bool Collector<PFP>::applyOnBorder(FunctorType& f)
{
	for(std::vector<Dart>::iterator iv = border.begin(); iv != border.end(); ++iv)
		if(f(*iv))
			return true ;
	return false ;
}

template <typename PPFP>
std::ostream& operator<<(std::ostream &out, const Collector<PPFP>& c)
{
    out << "Collector around " << c.centerDart << std::endl;
    out << "insideVertices (" << c.insideVertices.size() << ") = {";
    for (unsigned int i = 0; i< c.insideVertices.size(); ++i) out << c.insideVertices[i] << " ";
    out << "}" << std::endl ;
    out << "insideEdges (" << c.insideEdges.size() << ") = {";
    for (unsigned int i = 0; i< c.insideEdges.size(); ++i) out << c.insideEdges[i] << " ";
    out << "}" << std::endl ;
    out << "insideFaces (" << c.insideFaces.size() << ") = {";
    for (unsigned int i = 0; i< c.insideFaces.size(); ++i) out << c.insideFaces[i] << " ";
    out << "}" << std::endl ;
    out << "border (" << c.border.size() << ") = {";
    for (unsigned int i = 0; i< c.border.size(); ++i) out << c.border[i] << " ";
    out << "}" << std::endl;
    return out;
}

/*********************************************************
 * Collector One Ring
 *********************************************************/

template <typename PFP>
void Collector_OneRing<PFP>::collectAll(Dart d)
{
	this->init(d);
	this->isInsideCollected = true;
	this->insideEdges.reserve(16);
	this->insideFaces.reserve(16);
	this->border.reserve(16);

	this->insideVertices.push_back(d);

	foreach_incident2<EDGE>(this->map, Vertex(d), [&] (Edge e)
	{
		this->insideEdges.push_back(e);
		this->insideFaces.push_back(e.dart);
		this->border.push_back(this->map.phi1(e.dart));
	});
}

template <typename PFP>
void Collector_OneRing<PFP>::collectBorder(Dart d)
{
	this->init(d);
	this->border.reserve(12);

	foreach_incident2<FACE>(this->map, Vertex(d), [&] (Face f)
	{
		this->border.push_back(this->map.phi1(f.dart));
	});
}

template <typename PFP>
typename PFP::REAL Collector_OneRing<PFP>::computeArea(const VertexAttribute<VEC3, MAP>& pos)
{
	assert(this->isInsideCollected || !"computeArea: inside cells have not been collected.") ;

	REAL area = 0;

	for (std::vector<Face>::const_iterator it = this->insideFaces.begin(); it != this->insideFaces.end(); ++it)
		area += Algo::Surface::Geometry::triangleArea<PFP>(this->map, *it, pos);

	return area;
}

template <typename PFP>
void Collector_OneRing<PFP>::computeNormalCyclesTensor (const VertexAttribute<VEC3, MAP>& pos, const EdgeAttribute<REAL, MAP>& edgeangle, typename PFP::MATRIX33& tensor)
{
	assert(this->isInsideCollected || !"computeNormalCyclesTensor: inside cells have not been collected.") ;

	tensor.zero() ;

	// collect edges inside the neighborhood
	for (std::vector<Edge>::const_iterator it = this->insideEdges.begin(); it != this->insideEdges.end(); ++it)
	{
		const VEC3 e = Algo::Surface::Geometry::vectorOutOfDart<PFP>(this->map, *it, pos) ;
		tensor += Geom::transposed_vectors_mult(e,e) * edgeangle[*it] * (1 / e.norm()) ;
	}

	// collect edges on the border
	// TODO : should be an option ?
	// TODO : not boundary safe
	for (std::vector<Dart>::const_iterator it = this->border.begin(); it != this->border.end(); ++it)
	{
		const VEC3 e = Algo::Surface::Geometry::vectorOutOfDart<PFP>(this->map, *it, pos) ;
		tensor += Geom::transposed_vectors_mult(e,e) * edgeangle[*it] * (1 / e.norm()) ;
	}

	tensor /= computeArea(pos) ;
}

template <typename PFP>
void Collector_OneRing<PFP>::computeNormalCyclesTensor (const VertexAttribute<VEC3, MAP>& pos, typename PFP::MATRIX33& tensor)
{
	assert(this->isInsideCollected || !"computeNormalCyclesTensor: inside cells have not been collected.") ;

	tensor.zero() ;

	// collect edges inside the neighborhood
	for (std::vector<Edge>::const_iterator it = this->insideEdges.begin(); it != this->insideEdges.end(); ++it)
	{
		const VEC3 e = Algo::Surface::Geometry::vectorOutOfDart<PFP>(this->map, *it, pos) ;
		const REAL edgeangle = Algo::Surface::Geometry::computeAngleBetweenNormalsOnEdge<PFP>(this->map, *it, pos) ;
		tensor += Geom::transposed_vectors_mult(e,e) * edgeangle * (1 / e.norm()) ;
	}

	// collect edges on the border
	// TODO : should be an option ?
	// TODO : not boundary safe
	for (std::vector<Dart>::const_iterator it = this->border.begin(); it != this->border.end(); ++it)
	{
		const VEC3 e = Algo::Surface::Geometry::vectorOutOfDart<PFP>(this->map, *it, pos) ;
		const REAL edgeangle = Algo::Surface::Geometry::computeAngleBetweenNormalsOnEdge<PFP>(this->map, *it, pos) ;
		tensor += Geom::transposed_vectors_mult(e,e) * edgeangle * (1 / e.norm()) ;
	}

	tensor /= computeArea(pos) ;
}

/*********************************************************
 * Collector One Ring aroung edge
 *********************************************************/

template <typename PFP>
void Collector_OneRing_AroundEdge<PFP>::collectAll(Dart d)
{
	// TODO : this collector is not boundary safe

	// init
	this->init(d);
	this->isInsideCollected = true;

	const Dart d2 = this->map.phi2(d);

	this->insideEdges.reserve(16);
	this->insideFaces.reserve(16);
	this->border.reserve(16);

	CellMarkerStore<MAP, FACE> fm(this->map, this->m_thread);
	fm.mark(d);
	fm.mark(d2);

	// collect
	this->insideVertices.push_back(d);
	this->insideVertices.push_back(d2);
	this->insideEdges.push_back(d);

	foreach_adjacent2<VERTEX>(this->map, Edge(d), [&] (Edge e)
	{
		this->insideEdges.push_back(e);
		this->insideFaces.push_back(e.dart);
		if (! fm.isMarked(e.dart))
			this->border.push_back(this->map.phi1(e.dart));
	});
}

template <typename PFP>
void Collector_OneRing_AroundEdge<PFP>::collectBorder(Dart d)
{
	// TODO : this collector is not boundary safe

	// init
	this->init(d);
	const Dart d2 = this->map.phi2(d);

	this->border.reserve(16);

	CellMarkerStore<MAP, FACE> fm (this->map, this->m_thread);
	fm.mark(d);
	fm.mark(d2);

	foreach_adjacent2<VERTEX>(this->map, Edge(d), [&] (Edge e)
	{
		if (! fm.isMarked(e.dart))
			this->border.push_back(this->map.phi1(e.dart));
	});
}

template <typename PFP>
typename PFP::REAL Collector_OneRing_AroundEdge<PFP>::computeArea(const VertexAttribute<VEC3, MAP>& pos)
{
	assert(this->isInsideCollected || !"computeArea: inside cells have not been collected.") ;

	REAL area = 0;

	for (std::vector<Face>::const_iterator it = this->insideFaces.begin(); it != this->insideFaces.end(); ++it)
		area += Algo::Surface::Geometry::triangleArea<PFP>(this->map, *it, pos);

	return area;
}

template <typename PFP>
void Collector_OneRing_AroundEdge<PFP>::computeNormalCyclesTensor (const VertexAttribute<VEC3, MAP>& pos, const EdgeAttribute<REAL, MAP>& edgeangle, typename PFP::MATRIX33& tensor)
{
	assert(this->isInsideCollected || !"computeNormalCyclesTensor: inside cells have not been collected.") ;

	tensor.zero() ;

	// collect edges inside the neighborhood
	for (std::vector<Edge>::const_iterator it = this->insideEdges.begin(); it != this->insideEdges.end(); ++it)
	{
		const VEC3 e = Algo::Surface::Geometry::vectorOutOfDart<PFP>(this->map, *it, pos) ;
		tensor += Geom::transposed_vectors_mult(e,e) * edgeangle[*it] * (1 / e.norm()) ;
	}

	// collect edges on the border
	// TODO : should be an option ?
	// TODO : not boundary safe
	for (std::vector<Dart>::const_iterator it = this->border.begin(); it != this->border.end(); ++it)
	{
		const VEC3 e = Algo::Surface::Geometry::vectorOutOfDart<PFP>(this->map, *it, pos) ;
		tensor += Geom::transposed_vectors_mult(e,e) * edgeangle[*it] * (1 / e.norm()) ;
	}

	tensor /= computeArea(pos) ;
}

template <typename PFP>
void Collector_OneRing_AroundEdge<PFP>::computeNormalCyclesTensor (const VertexAttribute<VEC3, MAP>& pos, typename PFP::MATRIX33& tensor)
{
	assert(this->isInsideCollected || !"computeNormalCyclesTensor: inside cells have not been collected.") ;

	tensor.zero() ;

	// collect edges inside the neighborhood
	for (std::vector<Edge>::const_iterator it = this->insideEdges.begin(); it != this->insideEdges.end(); ++it)
	{
		const VEC3 e = Algo::Surface::Geometry::vectorOutOfDart<PFP>(this->map, *it, pos) ;
		const REAL edgeangle = Algo::Surface::Geometry::computeAngleBetweenNormalsOnEdge<PFP>(this->map, *it, pos) ;
		tensor += Geom::transposed_vectors_mult(e,e) * edgeangle * (1 / e.norm()) ;
	}

	// collect edges on the border
	// TODO : should be an option ?
	// TODO : not boundary safe
	for (std::vector<Dart>::const_iterator it = this->border.begin(); it != this->border.end(); ++it)
	{
		const VEC3 e = Algo::Surface::Geometry::vectorOutOfDart<PFP>(this->map, *it, pos) ;
		const REAL edgeangle = Algo::Surface::Geometry::computeAngleBetweenNormalsOnEdge<PFP>(this->map, *it, pos) ;
		tensor += Geom::transposed_vectors_mult(e,e) * edgeangle * (1 / e.norm()) ;
	}

	tensor /= computeArea(pos) ;
}

/*********************************************************
 * Collector Within Sphere
 *********************************************************/

template <typename PFP>
void Collector_WithinSphere<PFP>::collectAll(Dart d)
{
	this->init(d);
	this->isInsideCollected = true;
	this->insideEdges.reserve(32);
	this->insideFaces.reserve(32);
	this->border.reserve(32);

	CellMarkerStore<MAP, VERTEX> vm(this->map, this->m_thread);	// mark the collected inside-vertices
	CellMarkerStore<MAP, EDGE> em(this->map, this->m_thread);	// mark the collected inside-edges + border-edges
	CellMarkerStore<MAP, FACE> fm(this->map, this->m_thread);	// mark the collected inside-faces + border-faces

	this->insideVertices.push_back(d);
	vm.mark(d);

	VEC3 centerPosition = this->position[d];
	unsigned int i = 0;
	while (i < this->insideVertices.size())
	{
		Dart end = this->insideVertices[i];
		Dart e = end;
		do
		{
			if (! em.isMarked(e) || ! fm.isMarked(e)) // are both tests useful ?
			{
				const Dart f = this->map.phi1(e);
				const Dart g = this->map.phi1(f);

				if (! Geom::isPointInSphere(this->position[f], centerPosition, this->radius))
				{
					this->border.push_back(e); // add to border
					em.mark(e);
					fm.mark(e); // is it useful ?
				}
				else
				{
					if (! vm.isMarked(f))
					{
						this->insideVertices.push_back(f);
						vm.mark(f);
					}
					if (! em.isMarked(e))
					{
						this->insideEdges.push_back(e);
						em.mark(e);
					}
					if (! fm.isMarked(e) && Geom::isPointInSphere(this->position[g], centerPosition, this->radius))
					{
						this->insideFaces.push_back(e);
						fm.mark(e);
					}
				}
			}
			e = this->map.phi2_1(e);
		} while (e != end);
		++i;
	}
}

template <typename PFP>
void Collector_WithinSphere<PFP>::collectBorder(Dart d)
{
	this->init(d);
	this->border.reserve(128);
	this->insideVertices.reserve(128);

	CellMarkerStore<MAP, VERTEX> vm(this->map, this->m_thread);	// mark the collected inside-vertices
	CellMarkerStore<MAP, EDGE> em(this->map, this->m_thread);	// mark the collected inside-edges + border-edges

	this->insideVertices.push_back(d);
	vm.mark(d);

	VEC3 centerPosition = this->position[d];
	unsigned int i = 0;
	while (i < this->insideVertices.size())
	{
		Dart end = this->insideVertices[i];
		Dart e = end;
		do
		{
			if ( ! em.isMarked(e) )
			{
				const Dart f = this->map.phi1(e);

				if (! Geom::isPointInSphere(this->position[f], centerPosition, this->radius))
				{
					this->border.push_back(e); // add to border
				}
				else
				{
					if (! vm.isMarked(f))
					{
						this->insideVertices.push_back(f);
						vm.mark(f);
					}
				}
				em.mark(e);
			}
			e = this->map.phi2_1(e);
		} while (e != end);
		++i;
	}
	this->insideVertices.clear();
}


template <typename PFP>
typename PFP::REAL Collector_WithinSphere<PFP>::computeArea(const VertexAttribute<VEC3, MAP>& pos)
{
	assert(this->isInsideCollected || !"computeArea: inside cells have not been collected.") ;

	VEC3 centerPosition = pos[this->centerDart];
	REAL area = 0;

	for (std::vector<Face>::const_iterator it = this->insideFaces.begin(); it != this->insideFaces.end(); ++it)
		area += Geometry::triangleArea<PFP>(this->map, *it, pos);

	for (std::vector<Dart>::const_iterator it = this->border.begin(); it != this->border.end(); ++it)
	{
		const Dart f = this->map.phi1(*it); // we know that f is outside
		const Dart g = this->map.phi1(f);
		if (Geom::isPointInSphere(pos[g], centerPosition, this->radius))
		{ // only f is outside
			REAL alpha, beta;
			Geometry::intersectionSphereEdge<PFP>(this->map, centerPosition, this->radius, *it, pos, alpha);
			Geometry::intersectionSphereEdge<PFP>(this->map, centerPosition, this->radius, this->map.phi2(f), pos, beta);
			area += (alpha+beta - alpha*beta) * Algo::Surface::Geometry::triangleArea<PFP>(this->map, *it, pos);
		}
		else
		{ // f and g are outside
			REAL alpha, beta;
			Geometry::intersectionSphereEdge<PFP>(this->map, centerPosition, this->radius, *it, pos, alpha);
			Geometry::intersectionSphereEdge<PFP>(this->map, centerPosition, this->radius, this->map.phi2(g), pos, beta);
			area += alpha * beta * Algo::Surface::Geometry::triangleArea<PFP>(this->map, *it, pos);
		}
	}
	return area;
}

template <typename PFP>
void Collector_WithinSphere<PFP>::computeNormalCyclesTensor(const VertexAttribute<VEC3, MAP>& pos, const EdgeAttribute<REAL, MAP>& edgeangle, typename PFP::MATRIX33& tensor)
{
	assert(this->isInsideCollected || !"computeNormalCyclesTensor: inside cells have not been collected.") ;

	VEC3 centerPosition = pos[this->centerDart];
	tensor.zero() ;

	// collect edges inside the neighborhood
	for (std::vector<Edge>::const_iterator it = this->insideEdges.begin(); it != this->insideEdges.end(); ++it)
	{
		const VEC3 e = Algo::Surface::Geometry::vectorOutOfDart<PFP>(this->map, *it, pos) ;
		tensor += Geom::transposed_vectors_mult(e,e) * edgeangle[*it] * (1 / e.norm()) ;
	}
	// collect edges crossing the neighborhood's border
	for (std::vector<Dart>::const_iterator it = this->border.begin(); it != this->border.end(); ++it)
	{
		const VEC3 e = Algo::Surface::Geometry::vectorOutOfDart<PFP>(this->map, *it, pos) ;
		REAL alpha ;
		Algo::Surface::Geometry::intersectionSphereEdge<PFP>(this->map, centerPosition, radius, *it, pos, alpha) ;
		tensor += Geom::transposed_vectors_mult(e,e) * edgeangle[*it] * (1 / e.norm()) * alpha ;
	}

	tensor /= computeArea(pos) ;
}

template <typename PFP>
void Collector_WithinSphere<PFP>::computeNormalCyclesTensor(const VertexAttribute<VEC3, MAP>& pos, typename PFP::MATRIX33& tensor)
{
	assert(this->isInsideCollected || !"computeNormalCyclesTensor: inside cells have not been collected.") ;

	VEC3 centerPosition = pos[this->centerDart];
	tensor.zero() ;

	// collect edges inside the neighborhood
	for (std::vector<Edge>::const_iterator it = this->insideEdges.begin(); it != this->insideEdges.end(); ++it)
	{
		const VEC3 e = Algo::Surface::Geometry::vectorOutOfDart<PFP>(this->map, *it, pos) ;
		const REAL edgeangle = Algo::Surface::Geometry::computeAngleBetweenNormalsOnEdge<PFP>(this->map, *it, pos) ;
		tensor += Geom::transposed_vectors_mult(e,e) * edgeangle * (1 / e.norm()) ;
	}
	// collect edges crossing the neighborhood's border
	for (std::vector<Dart>::const_iterator it = this->border.begin(); it != this->border.end(); ++it)
	{
		const VEC3 e = Algo::Surface::Geometry::vectorOutOfDart<PFP>(this->map, *it, pos) ;
		REAL alpha ;
		Algo::Surface::Geometry::intersectionSphereEdge<PFP>(this->map, centerPosition, radius, *it, pos, alpha) ;
		const REAL edgeangle = Algo::Surface::Geometry::computeAngleBetweenNormalsOnEdge<PFP>(this->map, *it, pos) ;
		tensor += Geom::transposed_vectors_mult(e,e) * edgeangle * (1 / e.norm()) * alpha ;
	}

	tensor /= computeArea(pos) ;
}

/*********************************************************
 * Collector Normal Angle (Vertices)
 *********************************************************/

template <typename PFP>
void Collector_NormalAngle<PFP>::collectAll(Dart d)
{
	this->init(d);
	this->isInsideCollected = true;
	this->insideEdges.reserve(32);
	this->insideFaces.reserve(32);
	this->border.reserve(32);

	CellMarkerStore<MAP, VERTEX> vm(this->map, this->m_thread);	// mark the collected inside-vertices
	CellMarkerStore<MAP, EDGE> em(this->map, this->m_thread);	// mark the collected inside-edges + border-edges
	CellMarkerStore<MAP, FACE> fm(this->map, this->m_thread);	// mark the collected inside-faces + border-faces

	this->insideVertices.push_back(this->centerDart);
	vm.mark(this->centerDart);

	VEC3 centerNormal = this->normal[d];
	unsigned int i = 0;
	while (i < this->insideVertices.size())
	{
		Dart end = this->insideVertices[i];
		Dart e = end;
		do
		{
			if (! em.isMarked(e) || ! fm.isMarked(e)) // are both tests useful ?
			{
				const Dart f = this->map.phi1(e);
				const Dart g = this->map.phi1(f);

				REAL a = Geom::angle(centerNormal, this->normal[f]);

				if (a > angleThreshold)
				{
					this->border.push_back(e); // add to border
					em.mark(e);
					fm.mark(e); // is it useful ?
				}
				else
				{
					if (! vm.isMarked(f))
					{
						this->insideVertices.push_back(f);
						vm.mark(f);
					}
					if (! em.isMarked(e))
					{
						this->insideEdges.push_back(e);
						em.mark(e);
					}

					REAL b = Geom::angle(centerNormal, this->normal[g]);
					if (! fm.isMarked(e) && b < angleThreshold)
					{
						this->insideFaces.push_back(e);
						fm.mark(e);
					}
				}
			}
			e = this->map.phi2_1(e);
		} while (e != end);
		++i;
	}
}

template <typename PFP>
void Collector_NormalAngle<PFP>::collectBorder(Dart d)
{
	this->init(d);
	this->border.reserve(128);
	this->insideVertices.reserve(128);

	CellMarkerStore<MAP, VERTEX> vm(this->map, this->m_thread);	// mark the collected inside-vertices
	CellMarkerStore<MAP, EDGE> em(this->map, this->m_thread);	// mark the collected inside-edges + border-edges

	this->insideVertices.push_back(this->centerDart);
	vm.mark(this->centerDart);

	VEC3 centerNormal = this->normal[d];
	unsigned int i = 0;
	while (i < this->insideVertices.size())
	{
		Dart end = this->insideVertices[i];
		Dart e = end;
		do
		{
			if ( ! em.isMarked(e) )
			{
				const Dart f = this->map.phi1(e);

				REAL a = Geom::angle(centerNormal, this->normal[f]);

				if (a > angleThreshold)
				{
					this->border.push_back(e); // add to border
				}
				else
				{
					if (! vm.isMarked(f))
					{
						this->insideVertices.push_back(f);
						vm.mark(f);
					}
				}
				em.mark(e);
			}
			e = this->map.phi2_1(e);
		} while (e != end);
		++i;
	}
	this->insideVertices.clear();
}

/*********************************************************
 * Collector Normal Angle (Triangles)
 *********************************************************/

template <typename PFP>
void Collector_NormalAngle_Triangles<PFP>::collectAll(Dart d)
{
	this->init(d);
	this->isInsideCollected = true;
	this->insideVertices.reserve(32);
	this->insideEdges.reserve(32);
	this->insideFaces.reserve(32);
	this->border.reserve(32);

	CellMarkerStore<MAP, FACE> fm(this->map, this->m_thread);	// mark the collected inside-faces + front-faces
	CellMarkerStore<MAP, FACE> fminside(this->map, this->m_thread);	// mark the collected inside-faces

	std::queue<Dart> front;
	front.push(this->centerDart);
	fm.mark(this->centerDart);
	VEC3 centerNormal = this->normal[this->centerDart];

	while ( !front.empty() ) // collect inside faces
	{
		Dart f = front.front();
		front.pop();
		REAL a = Geom::angle(centerNormal, this->normal[f]);


		if (a < angleThreshold )
		{ // collect this face and add adjacent faces to the front
			this->insideFaces.push_back(f);
			fminside.mark(f);
			Traversor2FFaE<MAP> t (this->map, f) ;
			for (Dart it = t.begin(); it != t.end(); it=t.next())
			{
				if (!fm.isMarked(it))
				{
					front.push(it);
					fm.mark(it);
				}
			}
		}
	}

	CellMarkerStore<MAP, VERTEX> vm(this->map, this->m_thread);	// mark inside-vertices and border-vertices
	CellMarkerStore<MAP, EDGE> em(this->map, this->m_thread);	// mark inside-edges and border-edges
	std::vector<Dart>::iterator f_it;
	for (f_it = this->insideFaces.begin(); f_it != this->insideFaces.end(); f_it++)
	{ // collect insideVertices, insideEdges, and border
		Traversor2FE<MAP> te (this->map, *f_it) ;
		for (Dart it = te.begin(); it != te.end(); it=te.next())
		{ // collect insideEdges and border
			if (!em.isMarked(it))
			{
				em.mark(it);
				if (this->map.isBoundaryEdge(it))
					this->border.push_back(it);
				else if ( fminside.isMarked(it) && fminside.isMarked(this->map.phi2(it)) )
					this->insideEdges.push_back(it);
				else
					this->border.push_back(it);
			}
		}

		Traversor2FV<MAP> tv (this->map, *f_it) ;
		for (Dart it = tv.begin(); it != tv.end(); it=tv.next())
		{ // collect insideVertices
			if (!vm.isMarked(it))
			{
				vm.mark(it);
				this->insideVertices.push_back(it);
//				if ( !this->map.isBoundaryVertex(it))
//				{
//					Traversor2VF<typename PFP::MAP> tf (this->map, it);
//					Dart it2 = tf.begin();
//					while ( (it2 != tf.end()) && fminside.isMarked(it2))
//						it2=tf.next();
//					if (it2 == tf.end())
//						this->insideVertices.push_back(it);
//				}
			}
		}
	}

}

template <typename PFP>
void Collector_NormalAngle_Triangles<PFP>::collectBorder(Dart d)
{
	this->init(d);
	this->insideFaces.reserve(32);
	this->border.reserve(32);

	CellMarkerStore<MAP, FACE> fm(this->map, this->m_thread);	// mark the collected inside-faces + front-faces
	CellMarkerStore<MAP, FACE> fminside(this->map, this->m_thread);	// mark the collected inside-faces

	std::queue<Dart> front;
	front.push(this->centerDart);
	fm.mark(this->centerDart);
	VEC3 centerNormal = this->normal[this->centerDart];

	while ( !front.empty() ) // collect inside faces
	{
		Dart f = front.front();
		front.pop();
		REAL a = Geom::angle(centerNormal, this->normal[f]);

		if (a < angleThreshold )
		{ // collect this face and add adjacent faces to the front
			this->insideFaces.push_back(f);
			fminside.mark(f);
			Traversor2FFaE<MAP> t (this->map, f) ;
			for (Dart it = t.begin(); it != t.end(); it=t.next())
			{
				if (!fm.isMarked(it))
				{
					front.push(it);
					fm.mark(it);
				}
			}
		}
	}

	CellMarkerStore<MAP, EDGE> em(this->map, this->m_thread);	// mark inside-edges and border-edges
	std::vector<Dart>::iterator f_it;
	for (f_it = this->insideFaces.begin(); f_it != this->insideFaces.end(); f_it++)
	{ // collect border (edges)
		Traversor2FE<MAP> te (this->map, *f_it) ;
		for (Dart it = te.begin(); it != te.end(); it=te.next())
		{
			if (!em.isMarked(it))
			{
				em.mark(it);
				if (this->map.isBoundaryEdge(it))
					this->border.push_back(it);
				else if ( !fminside.isMarked(it) || !fminside.isMarked(this->map.phi2(it)) )
					this->border.push_back(it);
			}
		}
	}
	this->insideFaces.clear();
}

/*********************************************************
 * Collector Vertices
 *********************************************************/

template <typename PFP>
void Collector_Vertices<PFP>::collectAll(Dart d)
{
	crit.init(d);
	this->init(d);
	this->isInsideCollected = true;
	this->insideEdges.reserve(32);
	this->insideFaces.reserve(32);
	this->border.reserve(32);

	CellMarkerStore<MAP, VERTEX> vm(this->map, this->m_thread);	// mark the collected inside-vertices
	CellMarkerStore<MAP, EDGE> em(this->map, this->m_thread);	// mark the collected inside-edges + border-edges
	CellMarkerStore<MAP, FACE> fm(this->map, this->m_thread);	// mark the collected inside-faces + border-faces

	this->insideVertices.push_back(this->centerDart);
	vm.mark(this->centerDart);

	unsigned int i = 0;
	while (i < this->insideVertices.size())
	{
		Dart end = this->insideVertices[i];
		Dart e = end;
		do
		{
			if (! em.isMarked(e) || ! fm.isMarked(e)) // are both tests useful ?
			{
				const Dart f = this->map.phi1(e);
				const Dart g = this->map.phi1(f);

				if (! crit.isInside(f))
				{
					this->border.push_back(e); // add to border
					em.mark(e);
					fm.mark(e); // is it useful ?
				}
				else
				{
					if (! vm.isMarked(f))
					{
						this->insideVertices.push_back(f);
						vm.mark(f);
					}
					if (! em.isMarked(e))
					{
						this->insideEdges.push_back(e);
						em.mark(e);
					}
					if (! fm.isMarked(e) && crit.isInside(g))
					{
						this->insideFaces.push_back(e);
						fm.mark(e);
					}
				}
			}
			e = this->map.phi2_1(e);
		} while (e != end);
		++i;
	}
}

template <typename PFP>
void Collector_Vertices<PFP>::collectBorder(Dart d)
{
	crit.init(d);
	this->init(d);
	this->border.reserve(128);
	this->insideVertices.reserve(128);

	CellMarkerStore<MAP, VERTEX> vm(this->map, this->m_thread);	// mark the collected inside-vertices
	CellMarkerStore<MAP, EDGE> em(this->map, this->m_thread);	// mark the collected inside-edges + border-edges

	this->insideVertices.push_back(this->centerDart);
	vm.mark(this->centerDart);

	unsigned int i = 0;
	while (i < this->insideVertices.size())
	{
		Dart end = this->insideVertices[i];
		Dart e = end;
		do
		{
			if ( ! em.isMarked(e) )
			{
				const Dart f = this->map.phi1(e);

				if (! crit.isInside(f))
				{
					this->border.push_back(e); // add to border
				}
				else
				{
					if (! vm.isMarked(f))
					{
						this->insideVertices.push_back(f);
						vm.mark(f);
					}
				}
				em.mark(e);
			}
			e = this->map.phi2_1(e);
		} while (e != end);
		++i;
	}
	this->insideVertices.clear();
}

/*********************************************************
 * Collector Triangles
 *********************************************************/

template <typename PFP>
void Collector_Triangles<PFP>::collectAll(Dart d)
{
	crit.init(d);
	this->init(d);
	this->isInsideCollected = true;
	this->insideVertices.reserve(32);
	this->insideEdges.reserve(32);
	this->insideFaces.reserve(32);
	this->border.reserve(32);

	CellMarkerStore<MAP, FACE> fm(this->map, this->m_thread);	// mark the collected inside-faces + front-faces
	CellMarkerStore<MAP, FACE> fminside(this->map, this->m_thread);	// mark the collected inside-faces

	std::queue<Dart> front;
	front.push(this->centerDart);
	fm.mark(this->centerDart);

	while ( !front.empty() ) // collect inside faces
	{
		Dart f = front.front();
		front.pop();

		if (crit.isInside(f))
		{ // collect this face and add adjacent faces to the front
			this->insideFaces.push_back(f);
			fminside.mark(f);
			Traversor2FFaE<MAP> t (this->map, f) ;
			for (Dart it = t.begin(); it != t.end(); it=t.next())
			{
				if (!fm.isMarked(it))
				{
					front.push(it);
					fm.mark(it);
				}
			}
		}
	}

	CellMarkerStore<MAP, VERTEX> vm(this->map, this->m_thread);	// mark inside-vertices and border-vertices
	CellMarkerStore<MAP, EDGE> em(this->map, this->m_thread);	// mark inside-edges and border-edges
	std::vector<Dart>::iterator f_it;
	for (f_it = this->insideFaces.begin(); f_it != this->insideFaces.end(); f_it++)
	{ // collect insideVertices, insideEdges, and border
		Traversor2FE<MAP> te (this->map, *f_it) ;
		for (Dart it = te.begin(); it != te.end(); it=te.next())
		{ // collect insideEdges and border
			if (!em.isMarked(it))
			{
				em.mark(it);
				if (this->map.isBoundaryEdge(it))
					this->border.push_back(it);
				else if ( fminside.isMarked(it) && fminside.isMarked(this->map.phi2(it)) )
					this->insideEdges.push_back(it);
				else
					this->border.push_back(it);
			}
		}

		Traversor2FV<MAP> tv (this->map, *f_it) ;
		for (Dart it = tv.begin(); it != tv.end(); it=tv.next())
		{ // collect insideVertices
			if (!vm.isMarked(it))
			{
				vm.mark(it);
				this->insideVertices.push_back(it);
			}
		}
	}
}

template <typename PFP>
void Collector_Triangles<PFP>::collectBorder(Dart d)
{
	crit.init(d);
	this->init(d);
	this->insideFaces.reserve(32);
	this->border.reserve(32);

	CellMarkerStore<MAP, FACE> fm(this->map, this->m_thread);	// mark the collected inside-faces + front-faces
	CellMarkerStore<MAP, FACE> fminside(this->map, this->m_thread);	// mark the collected inside-faces

	std::queue<Dart> front;
	front.push(this->centerDart);
	fm.mark(this->centerDart);

	while ( !front.empty() ) // collect inside faces
	{
		Dart f = front.front();
		front.pop();

		if (crit.isInside(f) )
		{ // collect this face and add adjacent faces to the front
			this->insideFaces.push_back(f);
			fminside.mark(f);
			Traversor2FFaE<MAP> t (this->map, f) ;
			for (Dart it = t.begin(); it != t.end(); it=t.next())
			{
				if (!fm.isMarked(it))
				{
					front.push(it);
					fm.mark(it);
				}
			}
		}
	}

	CellMarkerStore<MAP, EDGE> em(this->map, this->m_thread);	// mark inside-edges and border-edges
	std::vector<Dart>::iterator f_it;
	for (f_it = this->insideFaces.begin(); f_it != this->insideFaces.end(); f_it++)
	{ // collect border (edges)
		Traversor2FE<MAP> te (this->map, *f_it) ;
		for (Dart it = te.begin(); it != te.end(); it=te.next())
		{
			if (!em.isMarked(it))
			{
				em.mark(it);
				if (this->map.isBoundaryEdge(it))
					this->border.push_back(it);
				else if ( !fminside.isMarked(it) || !fminside.isMarked(this->map.phi2(it)) )
					this->border.push_back(it);
			}
		}
	}
	this->insideFaces.clear();
}

/*********************************************************
 * Collector Dijkstra_Vertices
 *********************************************************/

template <typename PFP>
void Collector_Dijkstra_Vertices<PFP>::collectAll(Dart dinit)
{
	init(dinit);
	this->isInsideCollected = true;

	CellMarkerStore<MAP, VERTEX> vmReached (this->map, this->m_thread);
	vertexInfo[this->centerDart].it = front.insert(std::pair<float,Dart>(0.0, this->centerDart));
	vertexInfo[this->centerDart].valid = true;
	vmReached.mark(this->centerDart);

	while ( !front.empty() && front.begin()->first < this->maxDist)
	{
		Dart e = front.begin()->second;
		float d = front.begin()->first;
		front.erase(vertexInfo[e].it);
		vertexInfo[e].valid=false;
		this->insideVertices.push_back(e);

		Traversor2VVaE<MAP> tv (this->map, e);
		for (Dart f = tv.begin(); f != tv.end(); f=tv.next())
		{
			VertexInfo& vi (vertexInfo[f]);
			if (vmReached.isMarked(f))
			{
				if (vi.valid) // probably useless (because of distance test) but faster
				{
					float dist = d + edge_cost[f];
					if (dist < vi.it->first)
					{
						front.erase(vi.it);
						vi.it = front.insert(std::pair<float,Dart>(dist, f));
					}
				}
			}
			else
			{
				vi.it = front.insert(std::pair<float,Dart>(d + edge_cost[f], f));
				vi.valid=true;
				vmReached.mark(f);
			}

		}

	}

	while ( !front.empty())
	{
		vmReached.unmark(front.begin()->second);
		front.erase(front.begin());
	}

	CellMarkerStore<MAP, EDGE> em (this->map, this->m_thread);
	CellMarkerStore<MAP, FACE> fm (this->map, this->m_thread);
	for (std::vector<Dart>::iterator e_it = this->insideVertices.begin(); e_it != this->insideVertices.end() ; e_it++)
	{
		// collect insideEdges
		Traversor2VE<MAP> te (this->map, *e_it);
		for (Dart e = te.begin(); e != te.end(); e=te.next())
		{
			if ( !em.isMarked(e) && vmReached.isMarked(this->map.phi2(e)) )
			{ // opposite vertex is inside -> inside edge
				this->insideEdges.push_back(e);
				em.mark(e);
			}
		}

		// collect insideFaces and border
		Traversor2VF<MAP> tf (this->map, *e_it);
		for (Dart f = tf.begin(); f != tf.end(); f=tf.next())
		{
			if ( !fm.isMarked(f) )
			{
				fm.mark(f);
				Traversor2FV<MAP> tv (this->map, f);
				Dart v = tv.begin();
				while ( v != tv.end() && vmReached.isMarked(v) ) {v=tv.next();}
				if ( v == tv.end() )
					this->insideFaces.push_back(f);
				else
					this->border.push_back(f);
			}
		}
	}
}

template <typename PFP>
void Collector_Dijkstra_Vertices<PFP>::collectBorder(Dart dinit)
{
	init(dinit);

	CellMarkerStore<MAP, VERTEX> vmReached (this->map, this->m_thread);
	vertexInfo[this->centerDart].it = front.insert(std::pair<float,Dart>(0.0, this->centerDart));
	vertexInfo[this->centerDart].valid = true;
	vmReached.mark(this->centerDart);

	while ( !front.empty() && front.begin()->first < this->maxDist)
	{
		Dart e = front.begin()->second;
		float d = front.begin()->first;
		front.erase(vertexInfo[e].it);
		vertexInfo[e].valid=false;
		this->insideVertices.push_back(e);

		Traversor2VVaE<MAP> tv (this->map, e);
		for (Dart f = tv.begin(); f != tv.end(); f=tv.next())
		{
			VertexInfo& vi (vertexInfo[f]);
			if (vmReached.isMarked(f))
			{
				if (vi.valid) // probably useless (because of distance test) but faster
				{
					float dist = d + edge_cost[f];
					if (dist < vi.it->first)
					{
						front.erase(vi.it);
						vi.it = front.insert(std::pair<float,Dart>(dist, f));
					}
				}
			}
			else
			{
				vi.it = front.insert(std::pair<float,Dart>(d + edge_cost[f], f));
				vi.valid = true;
				vmReached.mark(f);
			}
		}
	}

	while ( !front.empty())
	{
		vmReached.unmark(front.begin()->second);
		front.erase(front.begin());
	}

	CellMarkerStore<MAP, FACE> fm (this->map, this->m_thread);
	for (std::vector<Dart>::iterator e_it = this->insideVertices.begin(); e_it != this->insideVertices.end() ; e_it++)
	{
		// collect border
		Traversor2VF<MAP> tf (this->map, *e_it);
		for (Dart f = tf.begin(); f != tf.end(); f=tf.next())
		{
			if ( !fm.isMarked(f) )
			{
				fm.mark(f);
				Traversor2FV<MAP> tv (this->map, f);
				Dart v = tv.begin();
				while ( v != tv.end() && vmReached.isMarked(v) ) {v=tv.next();}
				if ( v != tv.end() )
					this->border.push_back(f);
			}
		}
	}
	this->insideVertices.clear();
}

/*********************************************************
 * Collector Dijkstra
 *********************************************************/

template <typename PFP>
void Collector_Dijkstra<PFP>::collectAll(Dart dinit)
{
	init(dinit);
	this->isInsideCollected = true;

	CellMarkerStore<MAP, VERTEX> vmReached (this->map, this->m_thread);
	vertexInfo[this->centerDart].it = front.insert(std::pair<float,Dart>(0.0, this->centerDart));
	vertexInfo[this->centerDart].valid = true;
	vmReached.mark(this->centerDart);

	while ( !front.empty() && front.begin()->first < this->maxDist)
	{
		Dart e = front.begin()->second;
		float d = front.begin()->first;
		front.erase(vertexInfo[e].it);
		vertexInfo[e].valid=false;
		this->insideVertices.push_back(e);

		Traversor2VVaE<MAP> tv (this->map, e);
		for (Dart f = tv.begin(); f != tv.end(); f=tv.next())
		{
			VertexInfo& vi (vertexInfo[f]);
			if (vmReached.isMarked(f))
			{
				if (vi.valid) // probably useless (because of distance test) but faster
				{
					float dist = d + edgeLength(f);
					if (dist < vi.it->first)
					{
						front.erase(vi.it);
						vi.it = front.insert(std::pair<float,Dart>(dist, f));
					}
				}
			}
			else
			{
				vi.it = front.insert(std::pair<float,Dart>(d + edgeLength(f), f));
				vi.valid = true;
				vmReached.mark(f);
			}
		}
	}

	while ( !front.empty())
	{
		vmReached.unmark(front.begin()->second);
		front.erase(front.begin());
	}

	CellMarkerStore<MAP, EDGE> em (this->map, this->m_thread);
	CellMarkerStore<MAP, FACE> fm (this->map, this->m_thread);
	for (std::vector<Dart>::iterator e_it = this->insideVertices.begin(); e_it != this->insideVertices.end() ; e_it++)
	{
		// collect insideEdges
		Traversor2VE<MAP> te (this->map, *e_it);
		for (Dart e = te.begin(); e != te.end(); e=te.next())
		{
			if ( !em.isMarked(e) && vmReached.isMarked(this->map.phi2(e)) )
			{ // opposite vertex is inside -> inside edge
				this->insideEdges.push_back(e);
				em.mark(e);
			}
		}

		// collect insideFaces and border
		Traversor2VF<MAP> tf (this->map, *e_it);
		for (Dart f = tf.begin(); f != tf.end(); f=tf.next())
		{
			if ( !fm.isMarked(f) )
			{
				fm.mark(f);
				Traversor2FV<MAP> tv (this->map, f);
				Dart v = tv.begin();
				while ( v != tv.end() && vmReached.isMarked(v) ) {v=tv.next();}
				if ( v == tv.end() )
					this->insideFaces.push_back(f);
				else
					this->border.push_back(f);
			}
		}
	}
}

template <typename PFP>
void Collector_Dijkstra<PFP>::collectBorder(Dart dinit)
{
	init(dinit);

	CellMarkerStore<MAP, VERTEX> vmReached (this->map, this->m_thread);
	vertexInfo[this->centerDart].it = front.insert(std::pair<float,Dart>(0.0, this->centerDart));
	vertexInfo[this->centerDart].valid = true;
	vmReached.mark(this->centerDart);

	while ( !front.empty() && front.begin()->first < this->maxDist)
	{
		Dart e = front.begin()->second;
		float d = front.begin()->first;
		front.erase(vertexInfo[e].it);
		vertexInfo[e].valid=false;
		this->insideVertices.push_back(e);

		Traversor2VVaE<MAP> tv (this->map, e);
		for (Dart f = tv.begin(); f != tv.end(); f=tv.next())
		{
			VertexInfo& vi (vertexInfo[f]);
			if (vmReached.isMarked(f))
			{
				if (vi.valid) // probably useless (because of distance test) but faster
				{
					float dist = d + edgeLength(f);
					if (dist < vi.it->first)
					{
						front.erase(vi.it);
						vi.it = front.insert(std::pair<float,Dart>(dist, f));
					}
				}
			}
			else
			{
				vi.it = front.insert(std::pair<float,Dart>(d + edgeLength(f), f));
				vi.valid = true;
				vmReached.mark(f);
			}
		}
	}

	while ( !front.empty())
	{
		vmReached.unmark(front.begin()->second);
		front.erase(front.begin());
	}

	CellMarkerStore<MAP, FACE> fm (this->map, this->m_thread);
	for (std::vector<Dart>::iterator e_it = this->insideVertices.begin(); e_it != this->insideVertices.end() ; e_it++)
	{
		// collect border
		Traversor2VF<MAP> tf (this->map, *e_it);
		for (Dart f = tf.begin(); f != tf.end(); f=tf.next())
		{
			if ( !fm.isMarked(f) )
			{
				fm.mark(f);
				Traversor2FV<MAP> tv (this->map, f);
				Dart v = tv.begin();
				while ( v != tv.end() && vmReached.isMarked(v) ) {v=tv.next();}
				if ( v != tv.end() )
					this->border.push_back(f);
			}
		}
	}
	this->insideVertices.clear();
}

template <typename PFP>
inline float Collector_Dijkstra<PFP>::edgeLength (Dart d)
{
	typename PFP::VEC3 v = Geometry::vectorOutOfDart<PFP>(this->map, d, this->position);
	return v.norm();
}

} // namespace Selection

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
