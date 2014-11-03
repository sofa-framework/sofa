#ifndef PARTCELL_3D_H
#define PARTCELL_3D_H

#include "Algo/MovingObjects/particle_base.h"

#include "Algo/Geometry/inclusion.h"
#include "Geometry/intersection.h"
#include "Geometry/orientation.h"
#include "Geometry/plane_3d.h"

#include <iostream>

/* A particle cell is a particle base within a map, within a precise cell, the displacement function should indicate
   after each displacement wherein lies the new position of the particle */

namespace CGoGN
{

namespace Algo
{

namespace Volume
{

namespace MovingObjects
{

enum {
	NO_CROSS,
	CROSS_FACE,
	CROSS_OTHER
};

template <typename PFP>
class ParticleCell3D : public Algo::MovingObjects::ParticleBase<PFP>
{
public :
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;
	typedef VertexAttribute<VEC3, MAP> TAB_POS;

	MAP& m;

	const TAB_POS& position;

	Dart d;
	Dart lastCrossed;

	VEC3 m_positionFace;

	unsigned int state;

	unsigned int crossCell ;

	ParticleCell3D(MAP& map) : m(map)
	{}

	ParticleCell3D(MAP& map, Dart belonging_cell, VEC3 pos, const TAB_POS& tabPos) :
		Algo::MovingObjects::ParticleBase<PFP>(pos),
		m(map),
		position(tabPos),
		d(belonging_cell),
		state(3)
	{
		m_positionFace = pointInFace(d);
	}

	void display();

	Dart getCell() { return d; }

	VEC3 pointInFace(Dart d);

	Geom::Orientation3D isLeftENextVertex(VEC3 c, Dart d, VEC3 base);

	bool isRightVertex(VEC3 c, Dart d, VEC3 base);

	Geom::Orientation3D whichSideOfFace(VEC3 c, Dart d);

	Geom::Orientation3D isLeftL1DVol(VEC3 c, Dart d, VEC3 base, VEC3 top);

	Geom::Orientation3D isRightDVol(VEC3 c, Dart d, VEC3 base, VEC3 top);

	Geom::Orientation3D isAbove(VEC3 c, Dart d, VEC3 top);

	int isLeftL1DFace(VEC3 c, Dart d, VEC3 base, VEC3 normal);

	bool isRightDFace(VEC3 c, Dart d, VEC3 base, VEC3 normal);

	Dart nextDartOfVertexNotMarked(Dart d, CellMarkerStore<MAP, FACE>& mark);

	Dart nextNonPlanar(Dart d);

	Dart nextFaceNotMarked(Dart d, CellMarkerStore<MAP, FACE>& mark);

	Geom::Orientation3D whichSideOfEdge(VEC3 c, Dart d);

	bool isOnHalfEdge(VEC3 c, Dart d);

	void vertexState(const VEC3& current);

	void edgeState(const VEC3& current);

	void faceState(const VEC3& current, Geom::Orientation3D sideOfFace = Geom::ON);

	void volumeState(const VEC3& current);

	void volumeSpecialCase(const VEC3& current);

	void move(const VEC3& newCurrent)
	{
		crossCell = NO_CROSS ;

		if(!Geom::arePointsEquals(newCurrent, this->getPosition()))
		{
			switch(state) {
			case VERTEX : vertexState(newCurrent); break;
			case EDGE : 	edgeState(newCurrent);   break;
			case FACE : 	faceState(newCurrent);   break;
			case VOLUME : volumeState(newCurrent);   break;
			}

			display();
		}
	}
};

} // namespace MovingObjects

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/MovingObjects/particle_cell_3D.hpp"

#endif
