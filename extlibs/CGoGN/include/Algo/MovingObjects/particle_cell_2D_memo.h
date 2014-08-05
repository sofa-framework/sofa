#ifndef PARTCELL2DMEMO_H
#define PARTCELL2DMEMO_H

//#define DEBUG

#include "Algo/MovingObjects/particle_cell_2D.h"

#include "Algo/Geometry/inclusion.h"
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

template <typename PFP>
class ParticleCell2DMemo : public ParticleCell2D<PFP>
{
public:
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3;
	typedef VertexAttribute<VEC3, MAP> TAB_POS ;

private:
	ParticleCell2DMemo()
	{
	}

public:

	ParticleCell2DMemo(MAP& map, Dart belonging_cell, VEC3 pos, const TAB_POS& tabPos) :
	ParticleCell2D<PFP>(map, belonging_cell, pos, tabPos)
	{
	}

	virtual ~ParticleCell2DMemo()
	{
	}

	virtual void vertexState(const VEC3& current, CellMarkerMemo<MAP, FACE>& memo_cross) ;

	virtual void edgeState(const VEC3& current, CellMarkerMemo<MAP, FACE>& memo_cross, Geom::Orientation2D sideOfEdge = Geom::ALIGNED) ;

	virtual void faceState(const VEC3& current, CellMarkerMemo<MAP, FACE>& memo_cross) ;

	std::vector<Dart> move(const VEC3& goal, CellMarkerMemo<MAP, FACE>& memo_cross) ;

	std::vector<Dart> move(const VEC3& goal);
} ;

} // namespace MovingObjects

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/MovingObjects/particle_cell_2D_memo.hpp"

#endif
