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
std::vector<Dart> ParticleCell2DSecured<PFP>::move(const VEC3& goal)
{
	this->crossCell = NO_CROSS ;
	if (!Geom::arePointsEquals(goal, this->getPosition()))
	{
		CellMarkerMemo<FACE> memo_cross(this->m);
//		memo_cross.mark(this->d);

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
void ParticleCell2DSecured<PFP>::vertexState(const VEC3& current, CellMarkerMemo<MAP, FACE>& memo_cross)
{
#ifdef DEBUG
	CGoGNout << "vertexState" << this->d << CGoGNendl ;
#endif

//	std::vector<Dart> mc = memo_cross.get_markedCells();
//	if(std::find(mc.begin(),mc.end(),this->d)!=mc.end())
//	{
//		std::cout << "Error : particle outside map (vertex) " << std::endl;
//		return;
//	}
//	else
	{
		assert(std::isfinite(current[0]) && std::isfinite(current[1]) && std::isfinite(current[2])) ;
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
									&& Geometry::isPointOnHalfEdge<PFP>(this->m,this->m.phi2(this->d),this->positionAttribut,current)
									&& this->getOrientationEdge(current, this->d) == Geom::ALIGNED)
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
}

template <typename PFP>
void ParticleCell2DSecured<PFP>::edgeState(const VEC3& current, CellMarkerMemo<MAP, FACE>& memo_cross, Geom::Orientation2D sideOfEdge)
{
#ifdef DEBUG
	CGoGNout << "edgeState" << this->d << CGoGNendl ;
#endif

//	std::vector<Dart> mc = memo_cross.get_markedCells();
//	if(std::find(mc.begin(),mc.end(),this->d)!=mc.end())
//	{
//		std::cout << "Error : particle outside map (edge)" << std::endl;
//		return;
//	}
//	else
	{
			assert(std::isfinite(current[0]) && std::isfinite(current[1]) && std::isfinite(current[2])) ;
		// 	assert(Geometry::isPointOnEdge<PFP>(m,d,m_positions,m_position));
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

}

template <typename PFP>
void ParticleCell2DSecured<PFP>::faceState(const VEC3& current, CellMarkerMemo<MAP, FACE>& memo_cross)
{
#ifdef DEBUG
	CGoGNout << "faceState" << this->d << CGoGNendl ;
#endif

	std::vector<Dart> mc = memo_cross.get_markedCells();
	if(std::find(mc.begin(),mc.end(),this->d)!=mc.end())
	{
		std::cout << "Error : particle outside map (face)" << std::endl;
		return;
	}
	else
	{
		ParticleCell2DMemo<PFP>::faceState(current,memo_cross);
	}

}

} // namespace MovingObjects

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
