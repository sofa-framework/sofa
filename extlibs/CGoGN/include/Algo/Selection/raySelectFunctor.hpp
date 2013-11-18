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

#ifndef RAYSELECTFUNCTOR_H_
#define RAYSELECTFUNCTOR_H_

#include "Geometry/distances.h"
#include "Geometry/intersection.h"

#include "Algo/Geometry/centroid.h"

namespace CGoGN
{

namespace Algo
{

namespace Selection
{
//
//template <typename PFP>
//class FuncFaceInter: public FunctorMap<typename PFP::MAP>
//{
//	typedef typename PFP::MAP MAP;
//
//protected:
//	std::vector<Dart>& m_faces;
//	std::vector<typename PFP::VEC3>& m_Ipoints ;
//	const typename PFP::VEC3& m_A;
//	const typename PFP::VEC3& m_AB;
//	const VertexAttribute<typename PFP::VEC3>& m_positions;
//
//public:
//	/**
// 	* @param map the map
//	* @param f vector of selected darts
//	* @param A first point of ray
//	* @param AB direction of ray
//	*/
//	FuncFaceInter(MAP& map, const VertexAttribute<typename PFP::VEC3>& position, std::vector<Dart>& f, std::vector<typename PFP::VEC3>& ip, const typename PFP::VEC3& A, const typename PFP::VEC3& AB):
//		FunctorMap<typename PFP::MAP>(map), m_faces(f), m_Ipoints(ip), m_A(A), m_AB(AB), m_positions(position)
//	{}
//
//	bool operator()(Dart d)
//	{
//		const typename PFP::VEC3& Ta = m_positions[d];
//
//		Dart dd  = this->m_map.phi1(d);
//		Dart ddd = this->m_map.phi1(dd);
//		bool notfound = true;
//		do
//		{
//			// get back position of triangle Ta,Tb,Tc
//			const typename PFP::VEC3& Tb = m_positions[dd];
//			const typename PFP::VEC3& Tc = m_positions[ddd];
//			typename PFP::VEC3 I;
////			if (Geom::intersectionLineTriangle<typename PFP::VEC3>(m_A, m_AB, Ta, Tb, Tc, I))
//			if (Geom::intersectionRayTriangleOpt<typename PFP::VEC3>(m_A, m_AB, Ta, Tb, Tc, I))
//			{
//				m_faces.push_back(d);
//				m_Ipoints.push_back(I);
//				notfound = false;
//			}
//			// next triangle if we are in polygon
//			dd = ddd;
//			ddd = this->m_map.phi1(dd);
//		} while ((ddd != d) && notfound);
//		return false;
//	}
//};
//
//
//template <typename PFP>
//class FuncEdgeInter: public FunctorMap<typename PFP::MAP>
//{
//	typedef typename PFP::MAP MAP;
//
//protected:
//	std::vector<Dart>& m_edges;
//	const typename PFP::VEC3& m_A;
//	const typename PFP::VEC3& m_AB;
//	float m_AB2;
//	float m_distMax;
//	const VertexAttribute<typename PFP::VEC3>& m_positions;
//
//public:
//	/**
// 	* @param map the map
//	* @param e vector of selected darts
//	* @param A first point of ray
//	* @param AB direction of ray
//	* @param AB2 squared length of direction
//	* @param dm2 max distance from ray squared
//	*/
//	FuncEdgeInter(MAP& map, const VertexAttribute<typename PFP::VEC3>& position, std::vector<Dart>& e, const typename PFP::VEC3& A, const typename PFP::VEC3& AB, typename PFP::REAL AB2, typename PFP::REAL dm2):
//		FunctorMap<typename PFP::MAP>(map), m_edges(e), m_A(A), m_AB(AB), m_AB2(AB2), m_distMax(dm2), m_positions(position)
//	{}
//
//	bool operator()(Dart d)
//	{
//		// get back position of segment PQ
//		const typename PFP::VEC3& P = m_positions[d];
//		Dart dd = this->m_map.phi1(d);
//		const typename PFP::VEC3& Q = m_positions[dd];
//		// the three distance to P, Q and (PQ) not used here
//		float dist = Geom::squaredDistanceLine2Seg(m_A, m_AB, m_AB2, P, Q);
//
//		if (dist < m_distMax)
//		{
//			m_edges.push_back(d);
//		}
//		return false;
//	}
//};
//
//
//template <typename PFP>
//class FuncVertexInter: public FunctorMap<typename PFP::MAP>
//{
//	typedef typename PFP::MAP MAP;
//
//protected:
//	std::vector<Dart>& m_vertices;
//	const typename PFP::VEC3& m_A;
//	const typename PFP::VEC3& m_AB;
//	float m_AB2;
//	float m_distMax;
//	const VertexAttribute<typename PFP::VEC3>& m_positions;
//public:
//	/**
// 	* @param map the map
//	* @param v vector of selected darts
//	* @param A first point of ray
//	* @param AB direction of ray
//	* @param AB2 squared length of direction
//	* @param dm2 max distance from ray squared
//	*/
//	FuncVertexInter(MAP& map, const VertexAttribute<typename PFP::VEC3>& position, std::vector<Dart>& v, const typename PFP::VEC3& A, const typename PFP::VEC3& AB, typename PFP::REAL AB2, typename PFP::REAL dm2):
//		FunctorMap<typename PFP::MAP>(map), m_vertices(v), m_A(A), m_AB(AB), m_AB2(AB2), m_distMax(dm2), m_positions(position)
//	{}
//
//	bool operator()(Dart d)
//	{
//		const typename PFP::VEC3& P = m_positions[d];
//		float dist = Geom::squaredDistanceLine2Point(m_A, m_AB, m_AB2, P);
//		if (dist < m_distMax)
//		{
//			m_vertices.push_back(d);
//		}
//		return false;
//	}
//};

/**
 * Functor which store the dart that correspond to the subpart of face
 * that is intersected
 * Must be called in foreachface
 */
template <typename PFP>
class FuncDartMapD2Inter: public FunctorMap<typename PFP::MAP>
{
	typedef typename PFP::MAP MAP ;

protected:
	std::vector<Dart>& m_darts ;
	const typename PFP::VEC3& m_A ;
	const typename PFP::VEC3& m_AB ;
	const VertexAttribute<typename PFP::VEC3>& m_positions;
public:
	/**
 	* @param map the map
	* @param f vector of selected darts
	* @param A first point of ray
	* @param AB direction of ray
	*/
	FuncDartMapD2Inter(MAP& map, const VertexAttribute<typename PFP::VEC3>& position, std::vector<Dart>& f, const typename PFP::VEC3& A, const typename PFP::VEC3& AB):
		FunctorMap<typename PFP::MAP>(map), m_darts(f), m_A(A), m_AB(AB), m_positions(position)
	{}

	bool operator()(Dart d)
	{
		typename PFP::VEC3 center = Surface::Geometry::faceCentroid<PFP>(this->m_map, d, m_positions) ;
		bool notfound = true ;
		Dart face = d ;
		do
		{
			// get back position of triangle
			const typename PFP::VEC3& Tb = m_positions[face]; //this->m_map.getVertexEmb(face)->getPosition() ;
			const typename PFP::VEC3& Tc = m_positions[this->m_map.phi1(face)]; //this->m_map.getVertexEmb(this->m_map.phi1(face))->getPosition() ;
//			typename PFP::VEC3 I;
//			if (Geom::intersectionLineTriangle(m_A, m_AB, center, Tb, Tc, I))
			if (Geom::intersectionRayTriangleOpt(m_A, m_AB, center, Tb, Tc))
			{
				m_darts.push_back(face) ;
				notfound = false ;
			}
			face = this->m_map.phi1(face) ;
		} while((face != d) && notfound) ;
		return false;
	}
};




namespace Parallel
{

template <typename PFP>
class FuncVertexInter: public FunctorMapThreaded<typename PFP::MAP>
{
	typedef typename PFP::MAP MAP;

protected:
	const VertexAttribute<typename PFP::VEC3>& m_positions;
	const typename PFP::VEC3& m_A;
	const typename PFP::VEC3& m_AB;
	float m_AB2;
	float m_distMax;
	std::vector<std::pair<typename PFP::REAL, Dart> > m_vd;

public:
	FuncVertexInter(MAP& map, const VertexAttribute<typename PFP::VEC3>& position, const typename PFP::VEC3& A, const typename PFP::VEC3& AB, typename PFP::REAL AB2, typename PFP::REAL dm2):
		FunctorMapThreaded<typename PFP::MAP>(map),  m_positions(position), m_A(A), m_AB(AB), m_AB2(AB2), m_distMax(dm2)
	{}

	void run(Dart d, unsigned int thread)
	{
		const typename PFP::VEC3& P = m_positions[d];
		float dist = Geom::squaredDistanceLine2Point(m_A, m_AB, m_AB2, P);
		if (dist < m_distMax)
		{
			typename PFP::REAL distA = (P-m_A).norm2();
			m_vd.push_back(std::pair<typename PFP::REAL, Dart>(distA,d));
		}

	}

	const std::vector<std::pair<typename PFP::REAL, Dart> >& getVertexDistances() { return m_vd;}
};


template <typename PFP>
class FuncEdgeInter: public FunctorMapThreaded<typename PFP::MAP>
{
	typedef typename PFP::MAP MAP;

protected:
	const VertexAttribute<typename PFP::VEC3>& m_positions;
	const typename PFP::VEC3& m_A;
	const typename PFP::VEC3& m_AB;
	float m_AB2;
	float m_distMax;
	std::vector<std::pair<typename PFP::REAL, Dart> > m_ed;

public:
	FuncEdgeInter(MAP& map, const VertexAttribute<typename PFP::VEC3>& position, const typename PFP::VEC3& A, const typename PFP::VEC3& AB, typename PFP::REAL AB2, typename PFP::REAL dm2):
		FunctorMapThreaded<typename PFP::MAP>(map), m_positions(position), m_A(A), m_AB(AB), m_AB2(AB2), m_distMax(dm2)
	{}

	void run(Dart d, unsigned int thread)
	{
		// get back position of segment PQ
		const typename PFP::VEC3& P = m_positions[d];
		Dart dd = this->m_map.phi1(d);
		const typename PFP::VEC3& Q = m_positions[dd];
		// the three distance to P, Q and (PQ) not used here
		float dist = Geom::squaredDistanceLine2Seg(m_A, m_AB, m_AB2, P, Q);
		if (dist < m_distMax)
		{
			typename PFP::VEC3 M = (P+Q)/2.0;
			typename PFP::REAL distA = (M-m_A).norm2();
			m_ed.push_back(std::pair<typename PFP::REAL, Dart>(distA,d));
		}
	}

	const std::vector<std::pair<typename PFP::REAL, Dart> >& getEdgeDistances() { return m_ed;}
};



template <typename PFP>
class FuncFaceInter: public FunctorMapThreaded<typename PFP::MAP>
{
	typedef typename PFP::MAP MAP;

protected:
	const VertexAttribute<typename PFP::VEC3>& m_positions;
	const typename PFP::VEC3& m_A;
	const typename PFP::VEC3& m_AB;
	std::vector<std::pair<typename PFP::REAL, Dart> > m_fd;

public:
	FuncFaceInter(MAP& map, const VertexAttribute<typename PFP::VEC3>& position, const typename PFP::VEC3& A, const typename PFP::VEC3& AB):
		FunctorMapThreaded<typename PFP::MAP>(map), m_positions(position), m_A(A), m_AB(AB)
	{}

	void run(Dart d, unsigned int thread)
	{
		const typename PFP::VEC3& Ta = m_positions[d];

		Dart dd  = this->m_map.phi1(d);
		Dart ddd = this->m_map.phi1(dd);
		bool notfound = true;
		do
		{
			// get back position of triangle Ta,Tb,Tc
			const typename PFP::VEC3& Tb = m_positions[dd];
			const typename PFP::VEC3& Tc = m_positions[ddd];
			typename PFP::VEC3 I;
			if (Geom::intersectionRayTriangleOpt<typename PFP::VEC3>(m_A, m_AB, Ta, Tb, Tc, I))
			{
				typename PFP::REAL dist = (I-m_A).norm2();
				m_fd.push_back(std::pair<typename PFP::REAL, Dart>(dist,d));
				notfound = false;
			}
			// next triangle if we are in polygon
			dd = ddd;
			ddd = this->m_map.phi1(dd);
		} while ((ddd != d) && notfound);
	}

	const std::vector<std::pair<typename PFP::REAL, Dart> >& getFaceDistances() { return m_fd;}
};


}

} //namespace Selection

} //namespace Algo

} //namespace CGoGN

#endif
