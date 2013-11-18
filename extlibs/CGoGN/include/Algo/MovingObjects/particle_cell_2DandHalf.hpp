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

//#define DEBUG

#include "Geometry/frame.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MovingObjects
{

template <typename PFP>
void ParticleCell2DAndHalf<PFP>::display()
{
// 	CGoGNout << "pos : " << this->m_position << CGoGNendl;
// 	CGoGNout << "d : " << this->d << CGoGNendl;
// 	CGoGNout << "state : " << this->state << CGoGNendl;
}

template <typename PFP>
typename PFP::VEC3 ParticleCell2DAndHalf<PFP>::pointInFace(Dart d)
{
	const VEC3& p1(m_positions[d]) ;
	Dart dd = m.phi1(d) ;
	const VEC3& p2(m_positions[dd]) ;
	dd = m.phi1(dd) ;
	VEC3& p3(m_positions[dd]) ;

	VEC3 v1(p2 - p1) ;

	while ((v1 ^ VEC3(p3 - p1)).norm2() == 0.0f)
	{
		dd = m.phi1(dd) ;
		p3 = m_positions[dd] ;
	}

	CGoGNout << "pointInFace " << (p1 + p3) * 0.5f << CGoGNendl ;

	return (p1 + p3) * 0.5f ;
}

template <typename PFP>
Geom::Orientation3D ParticleCell2DAndHalf<PFP>::getOrientationEdge(const VEC3& point, Dart d)
{
	const VEC3& endPoint = m_positions[m.phi1(d)];
	const VEC3& vertexPoint = m_positions[d];

	const VEC3& n1 = Geometry::faceNormal<PFP>(m, d, m_positions);

	//orientation relative to the plane orthogonal to the face going through the edge
	return Geom::testOrientation3D(point, vertexPoint, endPoint, vertexPoint+n1);
}

template <typename PFP>
typename PFP::VEC3 ParticleCell2DAndHalf<PFP>::intersectLineEdge(const VEC3& pA, const VEC3& pB, Dart d)
{
	const VEC3& q1 = m_positions[d];
	const VEC3& q2 = m_positions[m.phi1(d)];
	VEC3 Inter;

	VEC3 n1 = Geometry::faceNormal<PFP>(m, d, m_positions);
	VEC3 n = (q2 - q1) ^ n1 ;

	Geom::intersectionLinePlane(pA, pB - pA, q1, n, Inter) ;

	Geom::Plane3D<float> pl = Geometry::facePlane<PFP>(m, d, m_positions);
	pl.project(Inter);

	return Inter;
}

template <typename PFP>
Geom::Orientation3D ParticleCell2DAndHalf<PFP>::getOrientationFace(VEC3 point, VEC3 sourcePoint, Dart d)
{
	const VEC3& dPoint = m_positions[d];

	VEC3 n1 = Geometry::faceNormal<PFP>(m, d, m_positions);

	return Geom::testOrientation3D(point, sourcePoint, dPoint+n1, dPoint);
}

template <typename PFP>
void ParticleCell2DAndHalf<PFP>::vertexState(VEC3 goal)
{
	#ifdef DEBUG
	CGoGNout << "vertexState" << d << CGoGNendl;
	#endif
	assert(goal.isFinite()) ;

	crossCell = CROSS_OTHER;

	if(Geometry::isPointOnVertex<PFP>(m,d,m_positions,goal))
	{
		state = VERTEX;
		distance += (goal - this->getPosition()).norm();
		this->Algo::MovingObjects::ParticleBase<PFP>::move(goal) ;
		return;
	}
	else
	{
		//orientation step
		if(m_positions[d][0] == m_positions[m.phi1(d)][0] && m_positions[d][1] == m_positions[m.phi1(d)][1])
			d = m.phi2_1(d);
		if(getOrientationEdge(this->getPosition(),m.phi2_1(d)) != Geom::UNDER)
		{
			Dart dd_vert = d;
			do
			{
				d = m.phi2_1(d);
				if(m_positions[d][0] == m_positions[m.phi1(d)][0] && m_positions[d][1] == m_positions[m.phi1(d)][1])
					d = m.phi2_1(d);
//			} while(getOrientationEdge(current, m.phi2_1(d)) != Geom::UNDER && dd_vert != d);
			} while(getOrientationEdge(goal, m.phi2_1(d)) != Geom::UNDER && dd_vert != d);

			if(dd_vert == d)
			{
				//orbit with 2 edges : point on one edge
				if(m.phi2_1(m.phi2_1(d)) == d)
				{
					if(!Geometry::isPointOnHalfEdge<PFP>(m,d,m_positions,goal))
//					if(!Geometry::isPointOnHalfEdge<PFP>(m,d,m_positions,current))
						d = m.phi2_1(d);
				}
				else
				{
					distance += (goal - this->getPosition()).norm();
					this->Algo::MovingObjects::ParticleBase<PFP>::move(goal);
					state = VERTEX;
					return;
				}
			}
		}
		else
		{
			Dart dd_vert = m.phi2_1(d);
			while(getOrientationEdge(goal, d) == Geom::OVER && dd_vert != d)
//			while(getOrientationEdge(current, d) == Geom::OVER && dd_vert != d)
			{
				d = m.phi12(d);
				if(m_positions[d][0] == m_positions[m.phi1(d)][0] && m_positions[d][1] == m_positions[m.phi1(d)][1])
					d = m.phi12(d);
			}
		}

		//displacement step
		if(getOrientationEdge(goal, d) == Geom::ON && Geometry::isPointOnHalfEdge<PFP>(m, d, m_positions, goal))
			edgeState(goal);
		else
		{
			d = m.phi1(d);
			faceState(goal);
		}
	}
}

template <typename PFP>
void ParticleCell2DAndHalf<PFP>::edgeState(VEC3 goal, Geom::Orientation3D sideOfEdge)
{
	#ifdef DEBUG
	CGoGNout << "edgeState" <<  d << CGoGNendl;
	#endif

	assert(goal.isFinite()) ;
// 	assert(Geometry::isPointOnEdge<PFP>(m,d,m_positions,m_position));

	if(crossCell == NO_CROSS)
	{
		crossCell = CROSS_EDGE;
		lastCrossed = d;
	}
	else
		crossCell = CROSS_OTHER;

	if(sideOfEdge == Geom::ON)
		sideOfEdge = getOrientationEdge(goal, d);

	switch(sideOfEdge)
	{
		case Geom::UNDER :
		{
			d = m.phi1(d);
			faceState(goal);
			return;
		}

		case Geom::OVER:
		{
			//transform the displacement into the new entered face
			VEC3 displ = goal - this->getPosition();

			VEC3 n1 = Geometry::faceNormal<PFP>(m, d, m_positions);
			VEC3 n2 = Geometry::faceNormal<PFP>(m, m.phi2(d), m_positions);
			VEC3 axis = n1 ^ n2 ;

			float angle = Geom::angle(n1, n2) ;

			displ = Geom::rotate(axis, angle, displ) ;
			goal = this->getPosition() + displ;

			d = m.phi1(m.phi2(d));
			faceState(goal);
			return;
		}

		default :
			state = EDGE;
	}

	if(!Geometry::isPointOnHalfEdge<PFP>(m, d, m_positions, goal))
	{
		distance += (goal - this->getPosition()).norm();
		this->Algo::MovingObjects::ParticleBase<PFP>::move(m_positions[d]) ;
		vertexState(goal);
		return;
	}
	else if(!Geometry::isPointOnHalfEdge<PFP>(m, m.phi2(d), m_positions, goal))
	{
		d = m.phi2(d);
		distance += (m_positions[d] - this->getPosition()).norm();
		this->Algo::MovingObjects::ParticleBase<PFP>::move(m_positions[d]) ;
		vertexState(goal);
		return;
	}
	distance += (goal - this->getPosition()).norm();
	this->Algo::MovingObjects::ParticleBase<PFP>::move(goal);
}

template <typename PFP>
void ParticleCell2DAndHalf<PFP>::faceState(VEC3 goal)
{
	#ifdef DEBUG
	CGoGNout << "faceState" <<  d << CGoGNendl;
	#endif

	assert(goal.isFinite()) ;
	assert(this->getPosition().isFinite()) ;

	//project goal within face plane
	VEC3 n1 = Geometry::faceNormal<PFP>(m,d,m_positions);
	VEC3 n2 = goal - this->getPosition();
//	n1.normalize();
	VEC3 n3 = n1 ^ n2;

	VEC3 n4 = n3 ^ n1;
	goal = this->getPosition() + n4;

	//track new position within map
	Dart dd = d;
	float wsoe = getOrientationFace(goal, this->getPosition(), m.phi1(d));

	// orientation step
	if(wsoe != Geom::UNDER)
	{
		d = m.phi1(d);
		wsoe = getOrientationFace(goal, this->getPosition(), m.phi1(d));
		while(wsoe != Geom::UNDER && dd != d)
		{
			d = m.phi1(d);
			wsoe = getOrientationFace(goal, this->getPosition(), m.phi1(d));
		}

 		// source and position to reach are the same : verify if no edge is crossed due to numerical approximation
		if(dd == d)
		{
			do
			{
				switch (getOrientationEdge(goal, d))
				{
				case Geom::UNDER: 	d = m.phi1(d);
									break;
//				case Geom::ON:		this->Algo::MovingObjects::ParticleBase<PFP>::move(current) ;
				case Geom::ON:		distance += (goal - this->getPosition()).norm();
									this->Algo::MovingObjects::ParticleBase<PFP>::move(goal) ;
									edgeState(goal);
									return;
				case Geom::OVER:
//									CGoGNout << "smthg went bad " << m_position << " " << goal << CGoGNendl;
//									CGoGNout << "d1 " << m_positions[d] << " d2 " << m_positions[m.phi1(d)] << CGoGNendl;
//									this->Algo::MovingObjects::ParticleBase<PFP>::move(intersectLineEdge(current, this->getPosition(), d));
									VEC3 inter = intersectLineEdge(goal, this->getPosition(), d);
									distance += (inter - this->getPosition()).norm();
									this->Algo::MovingObjects::ParticleBase<PFP>::move(inter);
//									CGoGNout << " " << m_position << CGoGNendl;

									edgeState(goal,Geom::OVER);
									return;
				}
			} while(d != dd);
			distance += (goal - this->getPosition()).norm();
			this->Algo::MovingObjects::ParticleBase<PFP>::move(goal);
			state = FACE;

// 			m_position = Geometry::faceCentroid<PFP>(m,d,m_positions);
// 			d = m.phi1(d);
// 			m_position = pointInFace(d);
// 			faceState(goal);

// 			m_position = m_positions[d];
// 			vertexState(current);
			return;
		}
		// take the orientation with d1 : in case we are going through a vertex
//		wsoe = getOrientationFace(current, this->getPosition(), d);
		wsoe = getOrientationFace(goal, this->getPosition(), d);
	}
	else
	{
		wsoe = getOrientationFace(goal,this->getPosition(),d);
		while(wsoe == Geom::UNDER && m.phi_1(d) != dd)
		{
			d = m.phi_1(d);
			wsoe = getOrientationFace(goal, this->getPosition(), d);
		}

		// in case of numerical incoherence
		if(m.phi_1(d) == dd && wsoe == Geom::UNDER)
		{
			d = m.phi_1(d);
			do
			{
				switch (getOrientationEdge(goal, d))
				{
				case Geom::UNDER :
					d = m.phi1(d);
					break;
				case Geom::ON :
// 					CGoGNout << "pic" << CGoGNendl;
					distance += (goal - this->getPosition()).norm();
					this->Algo::MovingObjects::ParticleBase<PFP>::move(goal) ;
					edgeState(goal);
					return;
				case Geom::OVER:
//					CGoGNout << "smthg went bad(2) " << m_position << CGoGNendl;
					VEC3 inter = intersectLineEdge(goal, this->getPosition(), d);
					distance += (inter - this->getPosition()).norm();
					this->Algo::MovingObjects::ParticleBase<PFP>::move(inter) ;
// 					CGoGNout << " " << m_position << CGoGNendl;
//					edgeState(current, Geom::OVER);
					edgeState(goal, Geom::OVER);
					return;
				}
			} while(d != dd);

			distance += (goal - this->getPosition()).norm();
			this->Algo::MovingObjects::ParticleBase<PFP>::move(goal);
			state = FACE;
			return;
		}
	}

	//displacement step
	switch (getOrientationEdge(goal, d))
	{
	case Geom::UNDER :
		distance += (goal - this->getPosition()).norm();
		this->Algo::MovingObjects::ParticleBase<PFP>::move(goal);
		state = FACE;
		break;
	default :
		if(wsoe == Geom::ON)
		{
//			std::cout << __FILE__ << " to uncomment and check" << std::endl;
//			d = m.phi1(d); //to check
//			m_position = m_positions[d];
//
//			vertexState(goal);
		}
		else
		{
// 			CGoGNout << "wsoe : " << wsoe << CGoGNendl;
// 			CGoGNout << "current " << current << " " << m_position << CGoGNendl;
// 			CGoGNout << "d " << d << "d1 " << m_positions[d] << " d2 " << m_positions[m.phi2(d)] << CGoGNendl;
//			VEC3 isect = intersectLineEdge(current, this->getPosition(), d);
			VEC3 isect = intersectLineEdge(goal, this->getPosition(), d);
			distance += (isect - this->getPosition()).norm();
			this->Algo::MovingObjects::ParticleBase<PFP>::move(isect);
// 			CGoGNout << " inter : " << m_position << CGoGNendl;
			edgeState(goal, Geom::OVER);
		}
	}
}

} // namespace MovingObjects

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
