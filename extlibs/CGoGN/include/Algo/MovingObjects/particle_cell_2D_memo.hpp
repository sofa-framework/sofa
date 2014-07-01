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

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MovingObjects
{

template <typename PFP>
std::vector<Dart> ParticleCell2DMemo<PFP>::move(const VEC3& goal)
{
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
std::vector<Dart> ParticleCell2DMemo<PFP>::move(const VEC3& goal, CellMarkerMemo<MAP, FACE>& memo_cross)
{
	this->crossCell = NO_CROSS ;
	if (!Geom::arePointsEquals(goal, this->getPosition()))
	{
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
void ParticleCell2DMemo<PFP>::vertexState(const VEC3& current, CellMarkerMemo<MAP, FACE>& memo_cross)
{
#ifdef DEBUG
	CGoGNout << "vertexState" << this->d << CGoGNendl ;
#endif
	assert(std::isfinite(current[0]) && std::isfinite(current[1]) && std::isfinite(current[2])) ;
	memo_cross.mark(this->d);
	this->crossCell = CROSS_OTHER ;

	if (Geometry::isPointOnVertex < PFP > (this->m, this->d, this->positionAttribut, current))
	{
		this->setState(VERTEX) ;
		this->Algo::MovingObjects::ParticleBase<PFP>::move(current) ;
		return ;
	}
	else
	{
		//orientation step
		if (this->positionAttribut[this->d][0] == this->positionAttribut[this->m.phi1(this->d)][0] && this->positionAttribut[this->d][1] == this->positionAttribut[this->m.phi1(this->d)][1])
			this->d = this->m.phi2_1(this->d) ;
		if (this->getOrientationEdge(current, this->m.phi2_1(this->d)) != Geom::RIGHT)
		{
			Dart dd_vert = this->d ;
			do
			{
				this->d = this->m.phi2_1(this->d) ;
				if (this->positionAttribut[this->d][0] == this->positionAttribut[this->m.phi1(this->d)][0]
					&& this->positionAttribut[this->d][1]== this->positionAttribut[this->m.phi1(this->d)][1])
					this->d = this->m.phi2_1(this->d) ;
			} while (this->getOrientationEdge(current, this->m.phi2_1(this->d)) != Geom::RIGHT && dd_vert != this->d) ;

			if (dd_vert == this->d)
			{
				//orbit with 2 edges : point on one edge
				if (this->m.phi2_1(this->m.phi2_1(this->d)) == this->d)
				{
					if (!Geometry::isPointOnHalfEdge<PFP>(this->m, this->d, this->positionAttribut, current))
						this->d = this->m.phi2_1(this->d) ;
				}
				else
				{
					//checking : case with 3 orthogonal darts and point on an edge
					do
					{
						if(Geometry::isPointOnHalfEdge<PFP>(this->m,this->d,this->positionAttribut,current)
								&& Geometry::isPointOnHalfEdge<PFP>(this->m,this->m.phi2(this->d),this->positionAttribut,current))
//								&& this->getOrientationEdge(current, this->d) == Geom::ALIGNED)
						{

							this->edgeState(current,memo_cross) ;
							return;
						}
						this->d = this->m.phi2_1(this->d) ;
					} while (this->getOrientationEdge(current, this->m.phi2_1(this->d)) != Geom::RIGHT && dd_vert != this->d) ;

					this->Algo::MovingObjects::ParticleBase<PFP>::move(current) ;
					this->setState(VERTEX) ;
					return ;
				}
			}
		}
		else
		{
			Dart dd_vert = this->m.phi2_1(this->d) ;
			while (this->getOrientationEdge(current, this->d) == Geom::RIGHT && dd_vert != this->d)
			{
				this->d = this->m.phi12(this->d) ;
				if (this->positionAttribut[this->d][0] == this->positionAttribut[this->m.phi1(this->d)][0]
				    && this->positionAttribut[this->d][1] == this->positionAttribut[this->m.phi1(this->d)][1])
					this->d = this->m.phi12(this->d) ;
			}
		}

		//displacement step

		if (this->getOrientationEdge(current, this->d) == Geom::ALIGNED
				&& Geometry::isPointOnHalfEdge<PFP>(this->m, this->d, this->positionAttribut, current))
			edgeState(current,memo_cross) ;
		else
		{
			this->d = this->m.phi1(this->d) ;
			faceState(current,memo_cross) ;
		}
	}
}

template <typename PFP>
void ParticleCell2DMemo<PFP>::edgeState(const VEC3& current, CellMarkerMemo<MAP, FACE>& memo_cross, Geom::Orientation2D sideOfEdge)
{
#ifdef DEBUG
	CGoGNout << "edgeState" << this->d << CGoGNendl ;
#endif

	assert(std::isfinite(current[0]) && std::isfinite(current[1]) && std::isfinite(current[2])) ;
// 	assert(Geometry::isPointOnEdge<PFP>(m,d,m_positions,m_position));
	memo_cross.mark(this->d);
	if (this->crossCell == NO_CROSS)
	{
		this->crossCell = CROSS_EDGE ;
		this->lastCrossed = this->d ;
	}
	else
		this->crossCell = CROSS_OTHER ;

	if (sideOfEdge == Geom::ALIGNED) sideOfEdge = this->getOrientationEdge(current, this->d) ;

	switch (sideOfEdge)
	{
		case Geom::LEFT :
			this->d = this->m.phi1(this->d) ;
			faceState(current,memo_cross) ;
			return ;
		case Geom::RIGHT :
			this->d = this->m.phi1(this->m.phi2(this->d)) ;
			faceState(current,memo_cross) ;
			return ;
		default :
			this->setState(EDGE) ;
			break ;
	}

	if (!Geometry::isPointOnHalfEdge < PFP
	    > (this->m, this->d, this->positionAttribut, current))
	{
		this->Algo::MovingObjects::ParticleBase<PFP>::move(this->positionAttribut[this->d]) ;
		vertexState(current,memo_cross) ;
		return ;
	}
	else if (!Geometry::isPointOnHalfEdge < PFP
	    > (this->m, this->m.phi2(this->d), this->positionAttribut, current))
	{
		this->d = this->m.phi2(this->d) ;
		this->Algo::MovingObjects::ParticleBase<PFP>::move(this->positionAttribut[this->d]) ;
		vertexState(current,memo_cross) ;
		return ;
	}

	this->Algo::MovingObjects::ParticleBase<PFP>::move(current) ;
}

template <typename PFP>
void ParticleCell2DMemo<PFP>::faceState(const VEC3& current, CellMarkerMemo<MAP, FACE>& memo_cross)
{
#ifdef DEBUG
	CGoGNout << "faceState" << this->d << CGoGNendl ;
#endif

	assert(
	    std::isfinite(this->getPosition()[0]) && std::isfinite(this->getPosition()[1])
	        && std::isfinite(this->getPosition()[2])) ;
	assert(std::isfinite(current[0]) && std::isfinite(current[1]) && std::isfinite(current[2])) ;
// 	assert(Geometry::isPointInConvexFace2D<PFP>(m,d,m_positions,m_position,true));
	memo_cross.mark(this->d);
	Dart dd = this->d ;
	float wsoe = this->getOrientationFace(current, this->m.phi1(this->d)) ;

// orientation step
	if (wsoe != Geom::RIGHT)
	{
		this->d = this->m.phi1(this->d) ;
		wsoe = this->getOrientationFace(current, this->m.phi1(this->d)) ;
		while (wsoe != Geom::RIGHT && dd != this->d)
		{
			this->d = this->m.phi1(this->d) ;
			wsoe = this->getOrientationFace(current, this->m.phi1(this->d)) ;
		}

		// source and position to reach are the same : verify if no edge is crossed due to numerical approximation
		if (dd == this->d)
		{
			do
			{
				switch (this->getOrientationEdge(current, this->d))
				{
					case Geom::LEFT :
						this->d = this->m.phi1(this->d) ;
						break ;
					case Geom::ALIGNED :
						this->Algo::MovingObjects::ParticleBase<PFP>::move(current) ;
						edgeState(current,memo_cross) ;
						return ;
					case Geom::RIGHT :
//									CGoGNout << "smthg went bad " << m_position << " " << current << CGoGNendl;
//									CGoGNout << "d1 " << m_positions[d] << " d2 " << m_positions[m.phi1(d)] << CGoGNendl;
					this->Algo::MovingObjects::ParticleBase<PFP>::move(this->intersectLineEdge(current, this->getPosition(), this->d)) ;
//									CGoGNout << " " << m_position << CGoGNendl;

					edgeState(current,memo_cross, Geom::RIGHT) ;
					return ;
				}
			} while (this->d != dd) ;
			this->Algo::MovingObjects::ParticleBase<PFP>::move(current);
			this->setState(FACE) ;

// 			m_position = Geometry::faceCentroid<PFP>(m,d,m_positions);
// 			d = m.phi1(d);
// 			m_position = pointInFace(d);
// 			faceState(current);

// 			m_position = m_positions[d];
// 			vertexState(current);
			return ;
		}
		// take the orientation with d1 : in case we are going through a vertex
		wsoe = this->getOrientationFace(current, this->d) ;
	}
	else
	{
		wsoe = this->getOrientationFace(current, this->d) ;
		while (wsoe == Geom::RIGHT && this->m.phi_1(this->d) != dd)
		{
			this->d = this->m.phi_1(this->d) ;
			wsoe = this->getOrientationFace(current, this->d) ;
		}

		// in case of numerical incoherence
		if (this->m.phi_1(this->d) == dd && wsoe == Geom::RIGHT)
		{
			this->d = this->m.phi_1(this->d) ;
			do
			{
				switch (this->getOrientationEdge(current, this->d))
				{
					case Geom::LEFT :
						this->d = this->m.phi1(this->d) ;
						break ;
					case Geom::ALIGNED :
// 					CGoGNout << "pic" << CGoGNendl;
						this->Algo::MovingObjects::ParticleBase<PFP>::move(current) ;
						edgeState(current,memo_cross) ;
						return ;
					case Geom::RIGHT :
//					CGoGNout << "smthg went bad(2) " << m_position << CGoGNendl;
						this->Algo::MovingObjects::ParticleBase<PFP>::move(this->intersectLineEdge(current, this->getPosition(), this->d)) ;
// 					CGoGNout << " " << m_position << CGoGNendl;
						edgeState(current,memo_cross ,Geom::RIGHT) ;
						return ;
				}
			} while (this->d != dd) ;

			this->Algo::MovingObjects::ParticleBase<PFP>::move(current) ;
			this->setState(FACE) ;
			return ;
		}
	}

//displacement step
	switch (this->getOrientationEdge(current, this->d))
	{
		case Geom::LEFT :
			this->Algo::MovingObjects::ParticleBase<PFP>::move(current) ;
			this->setState(FACE) ;
			;
			break ;
// 	case Geom::ALIGNED :
//		if(wsoe==Geom::ALIGNED) {
// 			m_position = m_positions[d];
// 			vertexState(current);
// 		}
// 		else {
// 			CGoGNout << "poc" << CGoGNendl;
// 			m_position = current;
// 			state = EDGE;
// 		}
// 		break;
		default :
			if (wsoe == Geom::ALIGNED)
			{

				this->d = this->m.phi1(this->d) ; //to check
				this->Algo::MovingObjects::ParticleBase<PFP>::move(this->positionAttribut[this->d]) ;
				vertexState(current,memo_cross) ;
			}
			else
			{
				this->Algo::MovingObjects::ParticleBase<PFP>::move(this->intersectLineEdge(current, this->getPosition(), this->d)) ;
				edgeState(current,memo_cross, Geom::RIGHT) ;
			}
	}

}

} // namespace MovingObjects

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
