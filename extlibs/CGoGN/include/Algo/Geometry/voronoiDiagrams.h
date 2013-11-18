#ifndef __VORONOI_DIAGRAMS_H__
#define __VORONOI_DIAGRAMS_H__

#include <vector>
#include <map>
#include <set>

//#include "Topology/map/map2.h"
#include "Topology/generic/traversor2.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{


namespace Geometry
{

template <typename PFP>
class VoronoiDiagram {
protected :
	typedef typename PFP::REAL REAL;

	typedef struct
	{
		typename std::multimap<float,Dart>::iterator it ;
		bool valid ;
//		unsigned int region;
		Dart pathOrigin;
		static std::string CGoGNnameOfType() { return "VoronoiVertexInfo" ; }
	} VoronoiVertexInfo ;
	typedef NoTypeNameAttribute<VoronoiVertexInfo> VertexInfo ;

	typename PFP::MAP& map;
	const EdgeAttribute<REAL>& edgeCost; // weights on the graph edges
	VertexAttribute<unsigned int>& regions; // region labels
	std::vector<Dart> border;
	std::vector<Dart> seeds;

	VertexAttribute<VertexInfo> vertexInfo;
	std::multimap<float,Dart> front ;
	CellMarker<VERTEX> vmReached;

public :
	VoronoiDiagram (typename PFP::MAP& m, const EdgeAttribute<REAL>& c, VertexAttribute<unsigned int>& r);
	~VoronoiDiagram ();

	const std::vector<Dart>& getSeeds (){return seeds;}
	virtual void setSeeds_fromVector (const std::vector<Dart>&);
	virtual void setSeeds_random (unsigned int nbseeds);
	const std::vector<Dart>& getBorder (){return border;}
	void setCost (const EdgeAttribute<REAL>& c);

	Dart computeDiagram ();
	virtual void computeDiagram_incremental (unsigned int nbseeds);
	void computeDistancesWithinRegion (Dart seed);

protected :
	virtual void clear ();
	void initFrontWithSeeds();
	virtual void collectVertexFromFront(Dart e);
	void addVertexToFront(Dart f, float d);
	void updateVertexInFront(Dart f, float d);
};


template <typename PFP>
class CentroidalVoronoiDiagram : public VoronoiDiagram<PFP> {
private :
	typedef typename PFP::REAL REAL;
	typedef typename PFP::VEC3 VEC3;

	double globalEnergy;
	std::vector<VEC3> energyGrad; // gradient of the region energy at seed

	VertexAttribute<REAL>& distances; // distances from the seed
	VertexAttribute<Dart>& pathOrigins; // previous vertex on the shortest path from origin
	VertexAttribute<REAL>& areaElts; // area element attached to each vertex

public :
	CentroidalVoronoiDiagram (typename PFP::MAP& m,
			const EdgeAttribute<REAL>& c,
			VertexAttribute<unsigned int>& r,
			VertexAttribute<REAL>& d,
			VertexAttribute<Dart>& o,
			VertexAttribute<REAL>& a);
	~CentroidalVoronoiDiagram ();

	void setSeeds_fromVector (const std::vector<Dart>&);
	void setSeeds_random (unsigned int nbseeds);
	void computeDiagram_incremental (unsigned int nbseeds);
	void cumulateEnergy();
	void cumulateEnergyAndGradients();
	unsigned int moveSeedsOneEdgeNoCheck(); // returns the number of seeds that did move
	// move each seed along one edge according to the energy gradient
	unsigned int moveSeedsOneEdgeCheck(); // returns the number of seeds that did move
	// move each seed along one edge according to the energy gradient + check that the energy decreases
	unsigned int moveSeedsToMedioid(); // returns the number of seeds that did move
	// move each seed to the medioid of its region
	REAL getGlobalEnergy() {return globalEnergy;}

protected :
	void clear();
	void collectVertexFromFront(Dart e);
	REAL cumulateEnergyFromRoot(Dart e);
	void cumulateEnergyAndGradientFromSeed(unsigned int numSeed);
	Dart selectBestNeighborFromSeed(unsigned int numSeed);
//	unsigned int moveSeed(unsigned int numSeed);
};


}// end namespace Geometry
}// end namespace Surface
}// end namespace Algo
}// end namespace CGoGN

#include "voronoiDiagrams.hpp"

#endif
