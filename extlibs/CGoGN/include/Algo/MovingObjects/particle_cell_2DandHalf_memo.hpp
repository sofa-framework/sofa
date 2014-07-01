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
std::vector<Dart> ParticleCell2DAndHalfMemo<PFP>::move(const VEC3& goal)
{
	this->distance = 0;
	this->crossCell = NO_CROSS ;
	if (!Geom::arePointsEquals(goal, this->getPosition()))
	{
		CellMarkerMemo<FACE> memo_cross(this->m);

		switch (this->getState())
		{
			case VERTEX :
				vertexState(goal,memo_cross) ;
				break ;
			case EDGE :
				edgeState(goal,memo_cross) ;
				break ;
			case FACE :
				faceState(goal,memo_cross) ;
				break ;
		}
		return memo_cross.get_markedCells();
	}
	else
		this->Algo::MovingObjects::ParticleBase<PFP>::move(goal) ;

	std::vector<Dart> res;
	res.push_back(this->d);
	return res;
}

template <typename PFP>
std::vector<Dart> ParticleCell2DAndHalfMemo<PFP>::move(const VEC3& goal, CellMarkerMemo<MAP, FACE>& memo_cross)
{
	this->distance = 0;
	this->crossCell = NO_CROSS ;
	if (!Geom::arePointsEquals(goal, this->getPosition()))
	{
		switch (this->getState())
		{
			case VERTEX :
				this->vertexState(goal,memo_cross) ;
				break ;
			case EDGE :
				this->edgeState(goal,memo_cross) ;
				break ;
			case FACE :
				this->faceState(goal,memo_cross) ;
				break ;
		}

		return memo_cross.get_markedCells();
	}
	else
		this->Algo::MovingObjects::ParticleBase<PFP>::move(goal) ;

	std::vector<Dart> res;
	res.push_back(this->d);
	return res;
}

template <typename PFP>
void ParticleCell2DAndHalfMemo<PFP>::vertexState(VEC3 current, CellMarkerMemo<MAP, FACE>& memo_cross)
{
	#ifdef DEBUG
	CGoGNout << "vertexState" << d << CGoGNendl;
	#endif
	assert(current.isFinite());

	this->crossCell = CROSS_OTHER;

	if(Geometry::isPointOnVertex<PFP>(this->m,this->d,this->m_positions,current))
	{
		this->state = VERTEX;
		this->Algo::MovingObjects::ParticleBase<PFP>::move(current) ;
		return;
	}
	else
	{
		//orientation step
		if(this->m_positions[this->d][0] == this->m_positions[this->m.phi1(this->d)][0] && this->m_positions[this->d][1] == this->m_positions[this->m.phi1(this->d)][1])
			this->d = this->m.phi2_1(this->d);
		if(this->getOrientationEdge(current,this->m.phi2_1(this->d)) != Geom::UNDER)
		{
			Dart dd_vert = this->d;
			do
			{
				this->d = this->m.phi2_1(this->d);
				if(this->m_positions[this->d][0] == this->m_positions[this->m.phi1(this->d)][0] && this->m_positions[this->d][1] == this->m_positions[this->m.phi1(this->d)][1])
					this->d = this->m.phi2_1(this->d);
			} while(this->getOrientationEdge(current, this->m.phi2_1(this->d)) != Geom::UNDER && dd_vert != this->d);

			if(dd_vert == this->d)
			{
				//orbit with 2 edges : point on one edge
				if(this->m.phi2_1(this->m.phi2_1(this->d)) == this->d)
				{
					if(!Geometry::isPointOnHalfEdge<PFP>(this->m,this->d,this->m_positions,current))
						this->d = this->m.phi2_1(this->d);
				}
				else
				{
					this->Algo::MovingObjects::ParticleBase<PFP>::move(current) ;
					this->state = VERTEX;
					return;
				}
			}
		}
		else
		{
			Dart dd_vert = this->m.phi2_1(this->d);
			while(this->getOrientationEdge(current, this->d) == Geom::OVER && dd_vert != this->d)
			{
				this->d = this->m.phi12(this->d);
				if(this->m_positions[this->d][0] == this->m_positions[this->m.phi1(this->d)][0] && this->m_positions[this->d][1] == this->m_positions[this->m.phi1(this->d)][1])
					this->d = this->m.phi12(this->d);
			}
		}

		//displacement step

		if(this->getOrientationEdge(current, this->d) == Geom::ON && Geometry::isPointOnHalfEdge<PFP>(this->m, this->d, this->m_positions, current))
			edgeState(current, memo_cross);
		else
		{
			this->d = this->m.phi1(this->d);
			faceState(current, memo_cross);
		}
	}
}

template <typename PFP>
void ParticleCell2DAndHalfMemo<PFP>::edgeState(VEC3 current, CellMarkerMemo<MAP, FACE>& memo_cross, Geom::Orientation3D sideOfEdge)
{
	#ifdef DEBUG
	CGoGNout << "edgeState" <<  d << CGoGNendl;
	#endif

	assert(current.isFinite());
// 	assert(Geometry::isPointOnEdge<PFP>(m,d,m_positions,m_position));

	if(this->crossCell == NO_CROSS)
	{
		this->crossCell = CROSS_EDGE;
		this->lastCrossed = this->d;
	}
	else
		this->crossCell = CROSS_OTHER;

	if(sideOfEdge == Geom::ON)
		sideOfEdge = this->getOrientationEdge(current,this-> d);

	switch(sideOfEdge)
	{
		case Geom::UNDER :
		{
			this->d = this->m.phi1(this->d);
			faceState(current, memo_cross);
			return;
		}

		case Geom::OVER:
		{
			//transform the displacement into the new entered face
			VEC3 displ = current - this->getPosition();

			VEC3 n1 = Geometry::faceNormal<PFP>(this->m, this->d, this->m_positions);
			VEC3 n2 = Geometry::faceNormal<PFP>(this->m, this->m.phi2(this->d), this->m_positions);
			VEC3 axis = n1 ^ n2 ;

			float angle = Geom::angle(n1, n2) ;

			displ = Geom::rotate(axis, angle, displ) ;
			current = this->getPosition() + displ;

			this->d = this->m.phi1(this->m.phi2(this->d));
			faceState(current, memo_cross);
			return;
		}

		default :
			this->state = EDGE;break;
	}

	if(!Geometry::isPointOnHalfEdge<PFP>(this->m, this->d, this->m_positions, current))
	{
		this->Algo::MovingObjects::ParticleBase<PFP>::move(this->m_positions[this->d]) ;
		vertexState(current, memo_cross);
		return;
	}
	else if(!Geometry::isPointOnHalfEdge<PFP>(this->m, this->m.phi2(this->d), this->m_positions, current))
	{
		this->d = this->m.phi2(this->d);
		this->Algo::MovingObjects::ParticleBase<PFP>::move(this->m_positions[this->d]) ;
		vertexState(current, memo_cross);
		return;
	}

	this->Algo::MovingObjects::ParticleBase<PFP>::move(current) ;
}

template <typename PFP>
void ParticleCell2DAndHalfMemo<PFP>::faceState(VEC3 current, CellMarkerMemo<MAP, FACE>& memo_cross)
{
	#ifdef DEBUG
	CGoGNout << "faceState" <<  this->d << CGoGNendl;
	#endif
	if(memo_cross.isMarked(this->d)) return ;
	memo_cross.mark(this->d);

 	assert(this->getPosition().isFinite());
 	assert(current.isFinite());
// 	assert(Geometry::isPointInConvexFace2D<PFP>(m,d,m_positions,m_position,true));

	//project current within face plane
	VEC3 n1 = Geometry::faceNormal<PFP>(this->m,this->d,this->m_positions);
	VEC3 n2 = current - this->getPosition();
//	n1.normalize();
	VEC3 n3 = n1 ^ n2;

	VEC3 n4 = n3 ^ n1;
	current = this->getPosition() + n4;

	//track new position within map
	Dart dd = this->d;
	float wsoe = this->getOrientationFace(current, this->getPosition(), this->m.phi1(this->d));

	// orientation step
	if(wsoe != Geom::UNDER)
	{
		this->d = this->m.phi1(this->d);
		wsoe = this->getOrientationFace(current, this->getPosition(), this->m.phi1(this->d));
		while(wsoe != Geom::UNDER && dd != this->d)
		{
			this->d = this->m.phi1(this->d);
			wsoe = this->getOrientationFace(current, this->getPosition(), this->m.phi1(this->d));
		}

 		// source and position to reach are the same : verify if no edge is crossed due to numerical approximation
		if(dd == this->d)
		{
			do
			{
				switch (this->getOrientationEdge(current, this->d))
				{
				case Geom::UNDER: 	this->d = this->m.phi1(this->d);
									break;
				case Geom::ON:			this->Algo::MovingObjects::ParticleBase<PFP>::move(current);
									edgeState(current, memo_cross);
									return;
				case Geom::OVER:
//									CGoGNout << "smthg went bad " << m_position << " " << current << CGoGNendl;
//									CGoGNout << "d1 " << m_positions[d] << " d2 " << m_positions[m.phi1(d)] << CGoGNendl;
									this->Algo::MovingObjects::ParticleBase<PFP>::move(this->intersectLineEdge(current, this->getPosition(), this->d));
//									CGoGNout << " " << this->m_position << CGoGNendl;

									edgeState(current, memo_cross,Geom::OVER);
									return;
				}
			} while(this->d != dd);

			this->Algo::MovingObjects::ParticleBase<PFP>::move(current);
			this->state = FACE;

// 			m_position = Geometry::faceCentroid<PFP>(m,d,m_positions);
// 			d = m.phi1(d);
// 			m_position = pointInFace(d);
// 			faceState(current);

// 			m_position = m_positions[d];
// 			vertexState(current);
			return;
		}
		// take the orientation with d1 : in case we are going through a vertex
		wsoe = this->getOrientationFace(current, this->getPosition(), this->d);
	}
	else
	{
		wsoe = this->getOrientationFace(current,this->getPosition(),this->d);
		while(wsoe == Geom::UNDER && this->m.phi_1(this->d) != dd)
		{
			this->d = this->m.phi_1(this->d);
			wsoe = this->getOrientationFace(current, this->getPosition(), this->d);
		}

		// in case of numerical incoherence
		if(this->m.phi_1(this->d) == dd && wsoe == Geom::UNDER)
		{
			this->d = this->m.phi_1(this->d);
			do
			{
				switch (this->getOrientationEdge(current, this->d))
				{
				case Geom::UNDER :
					this->d = this->m.phi1(this->d);
					break;
				case Geom::ON :
// 					CGoGNout << "pic" << CGoGNendl;
					this->Algo::MovingObjects::ParticleBase<PFP>::move(current);
					edgeState(current, memo_cross);
					return;
				case Geom::OVER:
//					CGoGNout << "smthg went bad(2) " << m_position << CGoGNendl;
					this->Algo::MovingObjects::ParticleBase<PFP>::move(this->intersectLineEdge(current, this->getPosition(), this->d));
// 					CGoGNout << " " << m_position << CGoGNendl;
					edgeState(current, memo_cross, Geom::OVER);
					return;
				}
			} while(this->d != dd);

			this->Algo::MovingObjects::ParticleBase<PFP>::move(current);
			this->state = FACE;
			return;
		}
	}

	//displacement step
	switch (this->getOrientationEdge(current, this->d))
	{
	case Geom::UNDER :
		this->distance += (current - this->getPosition()).norm();
		this->Algo::MovingObjects::ParticleBase<PFP>::move(current);
		this->state = FACE;
		break;
	default :
		if(wsoe == Geom::ON)
		{
			std::cout << __FILE__ << " to uncomment and check" << std::endl;
//			this->d = this->m.phi1(this->d); //to check
//			this->Algo::MovingObjects::ParticleBase<PFP>::move(this->m_positions[this->d]);
//			vertexState(current, memo_cross);
		}
		else
		{
// 			CGoGNout << "wsoe : " << wsoe << CGoGNendl;
// 			CGoGNout << "current " << current << " " << m_position << CGoGNendl;
// 			CGoGNout << "d " << d << "d1 " << m_positions[d] << " d2 " << m_positions[m.phi2(d)] << CGoGNendl;
			VEC3 isect = this->intersectLineEdge(current, this->getPosition(), this->d);
			this->distance += (isect - this->getPosition()).norm();
			this->Algo::MovingObjects::ParticleBase<PFP>::move(isect);
// 			CGoGNout << " inter : " << m_position << CGoGNendl;
			edgeState(current, memo_cross, Geom::OVER);
		};break;
	}
}

} // namespace MovingObjects

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
