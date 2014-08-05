#ifndef PARTCELL_H
#define PARTCELL_H

#include "Algo/MovingObjects/particle_base.h"
#include "Algo/Geometry/inclusion.h"
#include "Geometry/intersection.h"
#include "Algo/Geometry/orientation.h"

/* A particle cell is a particle base within a map, within a precise cell,
 * the displacement function should indicate after each displacement
 * wherein lies the new position of the particle
 */

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MovingObjects
{

#ifndef PARTCELL25D_H
enum
{
	NO_CROSS, CROSS_EDGE, CROSS_OTHER
} ;
#endif

template <typename PFP>
class ParticleCell2D : public Algo::MovingObjects::ParticleBase<PFP>
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef VertexAttribute<VEC3, MAP> TAB_POS ;

	MAP& m ;

	const TAB_POS& positionAttribut ;

	Dart d ;
	Dart lastCrossed ;

	unsigned int crossCell ;

	ParticleCell2D(MAP& map, Dart belonging_cell, const VEC3& pos, const TAB_POS& tabPos) :
		Algo::MovingObjects::ParticleBase<PFP>(pos),
		m(map),
		positionAttribut(tabPos),
		d(belonging_cell),
		lastCrossed(belonging_cell),
		crossCell(NO_CROSS)
	{
	}

	~ParticleCell2D()
	{
	}

	Dart getCell()
	{
		return d ;
	}

	Geom::Orientation2D getOrientationEdge(const VEC3& point, Dart d) ;

	void display() ;

// 	template <unsigned int DD, typename TT>
// 	friend std::istream& operator>> (std::istream& in, Vector<DD,TT>& v) ;

	VEC3 pointInFace(Dart d) ;

	VEC3 intersectLineEdge(const VEC3& pA, const VEC3& pB, Dart d) ;

	Geom::Orientation2D getOrientationFace(VEC3 sourcePoint, Dart d) ;

	virtual void vertexState(const VEC3& current) ;

	virtual void edgeState(const VEC3& current, Geom::Orientation2D sideOfEdge = Geom::ALIGNED) ;

	//just an orientation test : check which dart is aimed to leave the current face to reach an other position
	Dart faceOrientationState(const VEC3& toward) ;

	virtual void faceState(const VEC3& current) ;

	void move(const VEC3& goal)
	{
		crossCell = NO_CROSS ;
		if (!Geom::arePointsEquals(goal, this->getPosition()))
		{
			switch (this->getState())
			{
				case VERTEX :
					vertexState(goal) ;
					break ;
				case EDGE :
					edgeState(goal) ;
					break ;
				case FACE :
					faceState(goal) ;
					break ;
			}

			display() ;
		}
		else
		{
			// TODO Des petits pas répétés peuvent faire sortir de la cellule actuelle
			this->Algo::MovingObjects::ParticleBase<PFP>::move(goal) ;
		}
	}
} ;

} // namespace MovingObjects

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/MovingObjects/particle_cell_2D.hpp"

#endif
