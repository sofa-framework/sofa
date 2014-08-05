#ifndef PARTCELL2DANDHALFMEMO_H
#define PARTCELL2DANDHALFMEMO_H

#include "Algo/MovingObjects/particle_cell_2DandHalf.h"

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
class ParticleCell2DAndHalfMemo : public ParticleCell2DAndHalf<PFP>
{
public :
	typedef typename PFP::MAP MAP;
	typedef typename PFP::VEC3 VEC3;
	typedef VertexAttribute<VEC3, MAP> TAB_POS;

//	bool detect_vertex;
//	bool detect_edge;
//	bool detect_face;
	ParticleCell2DAndHalfMemo() {};

	ParticleCell2DAndHalfMemo(MAP& map, Dart belonging_cell, VEC3 pos, const TAB_POS& tabPos) :
		ParticleCell2DAndHalf<PFP>(map,belonging_cell,pos,tabPos)
//		detect_vertex(false),detect_edge(false),detect_face(true)
	{
//	 memo_cross.push_back(this->d);
	};

	void vertexState(VEC3 current, CellMarkerMemo<MAP, FACE>& memo_cross);

	void edgeState(VEC3 current, CellMarkerMemo<MAP, FACE>& memo_cross, Geom::Orientation3D sideOfEdge=Geom::ON);

	void faceState(VEC3 current, CellMarkerMemo<MAP, FACE>& memo_cross);

	std::vector<Dart> move(const VEC3& newCurrent, CellMarkerMemo<MAP, FACE>& memo_cross);

	std::vector<Dart> move(const VEC3& newCurrent);
};

} // namespace MovingObjects

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/MovingObjects/particle_cell_2DandHalf_memo.hpp"

#endif
