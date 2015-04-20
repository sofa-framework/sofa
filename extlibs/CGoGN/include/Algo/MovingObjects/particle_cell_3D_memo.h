#ifndef PARTCELL_3D_MEMO_H
#define PARTCELL_3D_MEMO_H

#include "Algo/MovingObjects/particle_cell_3D.h"

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

template <typename PFP>
class ParticleCell3DMemo : public Algo::Volume::MovingObjects::ParticleCell3D<PFP>
{
public :
    typedef typename PFP::MAP MAP;
    typedef typename PFP::VEC3 VEC3;
    typedef VertexAttribute<VEC3, MAP> TAB_POS;
    typedef FaceAttribute<VEC3,MAP> TAB_FACE;
    typedef VolumeAttribute<VEC3,MAP> TAB_VOL;

    ParticleCell3DMemo(MAP& map, Dart belonging_cell, VEC3 pos, const TAB_POS& tabPos,const TAB_FACE * fa_center = NULL,
                       const TAB_VOL * vol_center = NULL) :
    ParticleCell3D<PFP>(map, belonging_cell, pos, tabPos,fa_center,vol_center)
	{

	}

	virtual ~ParticleCell3DMemo()
	{

	}

	void vertexState(const VEC3& current, CellMarkerMemo<MAP, VOLUME>& memo_cross);

	void edgeState(const VEC3& current, CellMarkerMemo<MAP, VOLUME>& memo_cross);

	void faceState(const VEC3& current, CellMarkerMemo<MAP, VOLUME>& memo_cross, Geom::Orientation3D sideOfFace = Geom::ON);

	void volumeState(const VEC3& current, CellMarkerMemo<MAP, VOLUME>& memo_cross);

	std::vector<Dart> move(const VEC3& newCurrent, CellMarkerMemo<MAP, VOLUME>& memo_cross);

	std::vector<Dart> move(const VEC3& newCurrent);


};

} // namespace MovingObjects

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/MovingObjects/particle_cell_3D_memo.hpp"

#endif
