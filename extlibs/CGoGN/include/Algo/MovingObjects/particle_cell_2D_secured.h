#ifndef PARTCELL2DMEMOSECURED_H
#define PARTCELL2DMEMOSECURED_H

//#define DEBUG

#include "Algo/MovingObjects/particle_cell_2D_memo.h"

#include "Algo/Geometry/inclusion.h"
#include "Geometry/intersection.h"
#include "Geometry/orientation.h"

#include <iostream>

/* A particle cell secured : version of particle cell-memo where particles might go outside the navigation map
 * this class should be used for debug mode only */

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace MovingObjects
{

template <typename PFP>
class ParticleCell2DSecured : public ParticleCell2DMemo<PFP>
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3;
	typedef VertexAttribute<VEC3, MAP> TAB_POS ;

private:
	ParticleCell2DSecured()
	{
		std::cout << "Particle Secured : for debugging (unoptimized)" << std::endl;
	}

public:

	ParticleCell2DSecured(MAP& map, Dart belonging_cell, VEC3 pos, const TAB_POS& tabPos) :
		ParticleCell2DMemo<PFP>(map, belonging_cell, pos, tabPos)
	{
//		std::cout << "Particle Memo : for debugging (unoptimized)" << std::endl;
	}

	~ParticleCell2DSecured()
	{
	}

	virtual void vertexState(const VEC3& current, CellMarkerMemo<MAP, FACE>& memo_cross) ;

	virtual void edgeState(const VEC3& current, CellMarkerMemo<MAP, FACE>& memo_cross, Geom::Orientation2D sideOfEdge = Geom::ALIGNED) ;

	virtual void faceState(const VEC3& current, CellMarkerMemo<MAP, FACE>& memo_cross) ;

	std::vector<Dart> move(const VEC3& goal) ;

	std::vector<Dart> move(const VEC3& goal, CellMarkerMemo<MAP, FACE>& memo_cross) ;
} ;

} // namespace MovingObjects

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#include "Algo/MovingObjects/particle_cell_2D_secured.hpp"

#endif
