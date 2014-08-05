#ifndef PARTCELL25D_H
#define PARTCELL25D_H

#include "Algo/MovingObjects/particle_base.h"

#include "Algo/Geometry/inclusion.h"
#include "Algo/Geometry/plane.h"
#include "Geometry/intersection.h"
#include "Geometry/orientation.h"

#include <iostream>

/* A particle cell is a particle base within a map, within a precise cell, the displacement function should indicate
   after each displacement wherein lies the new position of the particle */

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MovingObjects
{

#ifndef PARTCELL_H
enum
{
	NO_CROSS,
	CROSS_EDGE,
	CROSS_OTHER
};
#endif

template <typename PFP>
class ParticleCell2DAndHalf : public Algo::MovingObjects::ParticleBase<PFP>
{
public :
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;
	typedef VertexAttribute<VEC3, MAP> TAB_POS;

	MAP& m;

	const TAB_POS& m_positions;

	Dart d;
	Dart lastCrossed;

	unsigned int state;

	unsigned int crossCell ;

	float distance;

	ParticleCell2DAndHalf(MAP& map) : m(map)
	{}

	ParticleCell2DAndHalf(MAP& map, Dart belonging_cell, VEC3 pos, const TAB_POS& tabPos) :
		Algo::MovingObjects::ParticleBase<PFP>(pos),
		m(map),
		m_positions(tabPos),
		d(belonging_cell),
		lastCrossed(belonging_cell),
		state(FACE),
		crossCell(NO_CROSS),
		distance(0)
	{}

	Dart getCell() { return d; }

	float getDistance() { return distance; }

	Geom::Orientation3D getOrientationEdge(const VEC3& point, Dart d);

	void display();

	VEC3 pointInFace(Dart d);

	VEC3 intersectLineEdge(const VEC3& pA, const VEC3& pB, Dart d);

	Geom::Orientation3D getOrientationFace(VEC3 sourcePoint, VEC3 point, Dart d);

	void vertexState(VEC3 current);

	void edgeState(VEC3 current, Geom::Orientation3D sideOfEdge = Geom::ON);

	void faceState(VEC3 current);

	virtual unsigned int getState()
	{
		return state;
	}

	void move(const VEC3& newCurrent)
	{
		distance = 0 ;
		crossCell = NO_CROSS ;
		if(!Geom::arePointsEquals(newCurrent, this->getPosition()))
		{
			switch(state) {
				case VERTEX : 	vertexState(newCurrent); break;
				case EDGE : 	edgeState(newCurrent);   break;
				case FACE : 	faceState(newCurrent);   break;
			}

//			display();
		}
		else
			this->Algo::MovingObjects::ParticleBase<PFP>::move(newCurrent);
	}
};

} // namespace MovingObjects

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/MovingObjects/particle_cell_2DandHalf.hpp"

#endif
