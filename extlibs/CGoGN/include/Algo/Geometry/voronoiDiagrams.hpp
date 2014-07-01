namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Geometry
{

/***********************************************************
 * class VoronoiDiagram
 ***********************************************************/

template <typename PFP>
VoronoiDiagram<PFP>::VoronoiDiagram (MAP& m, const EdgeAttribute<REAL, MAP>& p, VertexAttribute<unsigned int, MAP>& r) :
	map(m),
	edgeCost (p),
	regions (r),
	vmReached(m)
{
	vertexInfo = map.template addAttribute<VertexInfo, VERTEX>("vertexInfo");
}

template <typename PFP>
VoronoiDiagram<PFP>::~VoronoiDiagram ()
{
	map.removeAttribute(vertexInfo);
}

template <typename PFP>
void VoronoiDiagram<PFP>::clear ()
{
	regions.setAllValues(0);
	border.clear();
	front.clear();
	vmReached.unmarkAll();
}

template <typename PFP>
void VoronoiDiagram<PFP>::setSeeds_fromVector (const std::vector<Dart>& s)
{
	seeds.clear();
	seeds = s;
}

template <typename PFP>
void VoronoiDiagram<PFP>::setSeeds_random (unsigned int nseeds)
{
	seeds.clear();
	srand ( time(NULL) );
	const unsigned int nbv = map.getNbCells(VERTEX);

	std::set<unsigned int> myVertices ;
	while (myVertices.size() < nseeds)
	{
		myVertices.insert(rand() % nbv);
	}

	std::set<unsigned int>::iterator it = myVertices.begin();
	unsigned int n = 0;
	TraversorV<MAP> tv (map);
	Dart dit = tv.begin();

	while (it != myVertices.end())
	{
		while(n<*it)
		{
			dit = tv.next();
			++n;
		}
		seeds.push_back(dit);
		it++;
	}

	// random permutation = un-sort the seeds
	for (unsigned int i = 0; i < nseeds; i++)
	{
		unsigned int j = i + rand() % (nseeds - i);
		Dart d = seeds[i];
		seeds[i] = seeds[j];
		seeds[j] = d;
	}
}

template <typename PFP>
void VoronoiDiagram<PFP>::initFrontWithSeeds ()
{
//	vmReached.unmarkAll();
	clear();
	for (unsigned int i = 0; i < seeds.size(); i++)
	{
		Dart d = seeds[i];
		vmReached.mark(d);
		vertexInfo[d].it = front.insert(std::pair<float,Dart>(0.0, d));
		vertexInfo[d].valid = true;
		regions[d] = i;
		vertexInfo[d].pathOrigin = d;
	}
}

template <typename PFP>
void VoronoiDiagram<PFP>::setCost (const EdgeAttribute<typename PFP::REAL>& c)
{
	edgeCost = c;
}

template <typename PFP>
void VoronoiDiagram<PFP>::collectVertexFromFront(Dart e)
{
	regions[e] = regions[vertexInfo[e].pathOrigin];
	front.erase(vertexInfo[e].it);
	vertexInfo[e].valid=false;
}

template <typename PFP>
void VoronoiDiagram<PFP>::addVertexToFront(Dart f, float d)
{
	VertexInfo& vi (vertexInfo[f]);
	vi.it = front.insert(std::pair<float,Dart>(d + edgeCost[f], f));
	vi.valid = true;
	vi.pathOrigin = map.phi2(f);
	vmReached.mark(f);
}

template <typename PFP>
void VoronoiDiagram<PFP>::updateVertexInFront(Dart f, float d)
{
	VertexInfo& vi (vertexInfo[f]);
	float dist = d + edgeCost[f];
	if (dist < vi.it->first)
	{
		front.erase(vi.it);
		vi.it = front.insert(std::pair<float,Dart>(dist, f));
		vi.pathOrigin = map.phi2(f);
	}
}

template <typename PFP>
Dart VoronoiDiagram<PFP>::computeDiagram ()
{
	initFrontWithSeeds();

	Dart e;
	while ( !front.empty() )
	{
		e = front.begin()->second;
		float d = front.begin()->first;

		collectVertexFromFront(e);

		Traversor2VVaE<MAP> tv (map, e);
		for (Dart f = tv.begin(); f != tv.end(); f=tv.next())
		{
			if (vmReached.isMarked(f))
			{ // f has been reached
				if (vertexInfo[f].valid) // f is in the front : update
					updateVertexInFront(f,d);
				else // f is not in the front any more (already collected) : detect a border edge
				{
					if ( regions[f] != regions[e] )
						border.push_back(f);
				}
			}
			else
			{ // f has not been reached : add it to the front
				addVertexToFront(f,d);
			}
		}
	}
	return e;
}

template <typename PFP>
void VoronoiDiagram<PFP>::computeDiagram_incremental (unsigned int nseeds)
{
	seeds.clear();

	// first seed
	srand ( time(NULL) );
	unsigned int s = rand() % map.getNbCells(VERTEX);
	unsigned int n = 0;
	TraversorV<MAP> tv (map);
	Dart dit = tv.begin();
	while(n < s)
	{
		dit = tv.next();
		++n;
	}
	seeds.push_back(dit);

	// add other seeds one by one
	Dart e = computeDiagram();

	for(unsigned int i = 1; i < nseeds ; i++)
	{
		seeds.push_back(e);
		e = computeDiagram();
	}
}

template <typename PFP>
void VoronoiDiagram<PFP>::computeDistancesWithinRegion (Dart seed)
{
	// init
	front.clear();
	vmReached.unmarkAll();

	vmReached.mark(seed);
	vertexInfo[seed].it = front.insert(std::pair<float,Dart>(0.0, seed));
	vertexInfo[seed].valid = true;
	vertexInfo[seed].pathOrigin = seed;

	//compute
	while ( !front.empty() )
	{
		Dart e = front.begin()->second;
		float d = front.begin()->first;

		collectVertexFromFront(e);

		Traversor2VVaE<MAP> tv (map, e);
		for (Dart f = tv.begin(); f != tv.end(); f=tv.next())
		{
			if (vmReached.isMarked(f))
			{ // f has been reached
				if (vertexInfo[f].valid) updateVertexInFront(f,d); // f is in the front : update
			}
			else
			{ // f has not been reached : add it to the front
				if ( regions[f] == regions[e] ) addVertexToFront(f,d);
			}
		}
	}
}

/***********************************************************
 * class CentroidalVoronoiDiagram
 ***********************************************************/

template <typename PFP>
CentroidalVoronoiDiagram<PFP>::CentroidalVoronoiDiagram (
	MAP& m,
	const EdgeAttribute<REAL, MAP>& c,
	VertexAttribute<unsigned int, MAP>& r,
	VertexAttribute<REAL, MAP>& d,
	VertexAttribute<Dart, MAP>& o,
	VertexAttribute<REAL, MAP>& a) :
		VoronoiDiagram<PFP>(m,c,r), distances(d), pathOrigins(o), areaElts(a)
{
}

template <typename PFP>
CentroidalVoronoiDiagram<PFP>::~CentroidalVoronoiDiagram ()
{
}

template <typename PFP>
void CentroidalVoronoiDiagram<PFP>::clear ()
{
	VoronoiDiagram<PFP>::clear();
	distances.setAllValues(0.0);
}

template <typename PFP>
void CentroidalVoronoiDiagram<PFP>::collectVertexFromFront(Dart e)
{
	distances[e] = this->vertexInfo[e].it->first;
	pathOrigins[e] = this->vertexInfo[e].pathOrigin;

	VoronoiDiagram<PFP>::collectVertexFromFront(e);
}


template <typename PFP>
void CentroidalVoronoiDiagram<PFP>::setSeeds_fromVector (const std::vector<Dart>& s)
{
	VoronoiDiagram<PFP>::setSeeds_fromVector (s);
	energyGrad.resize(this->seeds.size());
}

template <typename PFP>
void CentroidalVoronoiDiagram<PFP>::setSeeds_random (unsigned int nseeds)
{
	VoronoiDiagram<PFP>::setSeeds_random (nseeds);
	energyGrad.resize(this->seeds.size());
}

template <typename PFP>
void CentroidalVoronoiDiagram<PFP>::computeDiagram_incremental (unsigned int nseeds)
{
	VoronoiDiagram<PFP>::computeDiagram_incremental (nseeds);
	energyGrad.resize(this->seeds.size());
}

template <typename PFP>
void CentroidalVoronoiDiagram<PFP>::cumulateEnergy()
{
	globalEnergy = 0.0;
	for (unsigned int i = 0; i < this->seeds.size(); i++)
	{
		cumulateEnergyFromRoot(this->seeds[i]);
		globalEnergy += distances[this->seeds[i]];
	}
}

template <typename PFP>
void CentroidalVoronoiDiagram<PFP>::cumulateEnergyAndGradients()
{
	globalEnergy = 0.0;
	for (unsigned int i = 0; i < this->seeds.size(); i++)
	{
		cumulateEnergyAndGradientFromSeed(i);
		globalEnergy += distances[this->seeds[i]];
	}
}

template <typename PFP>
unsigned int CentroidalVoronoiDiagram<PFP>::moveSeedsOneEdgeNoCheck()
{
	unsigned int m = 0;
	for (unsigned int i = 0; i < this->seeds.size(); i++)
	{
		Dart oldSeed = this->seeds[i];
		Dart newSeed = selectBestNeighborFromSeed(i);

		// move the seed
		if (newSeed != oldSeed)
		{
			this->seeds[i] = newSeed;
			m++;
		}
	}
	return m;
}

template <typename PFP>
unsigned int CentroidalVoronoiDiagram<PFP>::moveSeedsOneEdgeCheck()
{
	unsigned int m = 0;
	for (unsigned int i = 0; i < this->seeds.size(); i++)
	{
		Dart oldSeed = this->seeds[i];
		Dart newSeed = selectBestNeighborFromSeed(i);

		// move the seed
		if (newSeed != oldSeed)
		{
			REAL regionEnergy = distances[oldSeed];
			this->seeds[i] = newSeed;
			this->computeDistancesWithinRegion(newSeed);
			cumulateEnergyAndGradientFromSeed(i);
			if (distances[newSeed] < regionEnergy)
				m++;
			else
				this->seeds[i] = oldSeed;
		}
	}
	return m;
}

template <typename PFP>
unsigned int CentroidalVoronoiDiagram<PFP>::moveSeedsToMedioid()
{
	unsigned int m = 0;
	for (unsigned int i = 0; i < this->seeds.size(); i++)
	{
		Dart oldSeed, newSeed;
		unsigned int seedMoved = 0;
		REAL regionEnergy;

		do
		{
			oldSeed = this->seeds[i];
			regionEnergy = distances[oldSeed];

			newSeed = selectBestNeighborFromSeed(i);
			this->seeds[i] = newSeed;
			this->computeDistancesWithinRegion(newSeed);
			cumulateEnergyAndGradientFromSeed(i);
			if (distances[newSeed] < regionEnergy)
				seedMoved = 1;
			else
			{
				this->seeds[i] = oldSeed;
				newSeed = oldSeed;
			}

		} while (newSeed != oldSeed);

		m += seedMoved;
	}
	return m;
}

template <typename PFP>
typename PFP::REAL CentroidalVoronoiDiagram<PFP>::cumulateEnergyFromRoot(Dart e)
{
	REAL distArea = areaElts[e] * distances[e];
	distances[e] = areaElts[e] * distances[e] * distances[e];

	Traversor2VVaE<MAP> tv (this->map, e);
	for (Dart f = tv.begin(); f != tv.end(); f=tv.next())
	{
		if ( pathOrigins[f] == this->map.phi2(f))
		{
			distArea += cumulateEnergyFromRoot(f);
			distances[e] += distances[f];
		}
	}
	return distArea;
}

template <typename PFP>
void CentroidalVoronoiDiagram<PFP>::cumulateEnergyAndGradientFromSeed(unsigned int numSeed)
{
	// precondition : energyGrad.size() > numSeed
	Dart e = this->seeds[numSeed];

	std::vector<Dart> v;
	v.reserve(8);

	std::vector<float> da;
	da.reserve(8);

	distances[e] = 0.0;

	Traversor2VVaE<MAP> tv (this->map, e);
	for (Dart f = tv.begin(); f != tv.end(); f=tv.next())
	{
		if ( pathOrigins[f] == this->map.phi2(f))
		{
			float distArea = cumulateEnergyFromRoot(f);
			da.push_back(distArea);
			distances[e] += distances[f];
			v.push_back(f);
		}
	}

	// compute the gradient
	// TODO : check if the computation of grad and proj is still valid for other edgeCost than geodesic distances
	VEC3 grad (0.0);
	const VertexAttribute<VEC3, MAP>& pos = this->map.template getAttribute<VEC3, VERTEX>("position");

	for (unsigned int j = 0; j < v.size(); ++j)
	{
		Dart f = v[j];
		VEC3 edgeV = pos[f] - pos[this->map.phi2(f)];
		edgeV.normalize();
		grad += da[j] * edgeV;
	}
	grad /= 2.0;
	energyGrad[numSeed] = grad;
}

template <typename PFP>
Dart CentroidalVoronoiDiagram<PFP>::selectBestNeighborFromSeed(unsigned int numSeed)
{
	Dart e = this->seeds[numSeed];
	Dart newSeed = e;
	const VertexAttribute<VEC3, MAP>& pos = this->map.template getAttribute<VEC3,VERTEX>("position");

	// TODO : check if the computation of grad and proj is still valid for other edgeCost than geodesic distances
	float maxProj = 0.0;
	Traversor2VVaE<MAP> tv (this->map, e);
	for (Dart f = tv.begin(); f != tv.end(); f=tv.next())
	{
		if ( pathOrigins[f] == this->map.phi2(f))
		{
			VEC3 edgeV = pos[f] - pos[this->map.phi2(f)];
	//		edgeV.normalize();
			float proj = edgeV * energyGrad[numSeed];
			if (proj > maxProj)
			{
				maxProj = proj;
				newSeed = f;
			}
		}
	}
	return newSeed;
}

/*
template <typename PFP>
unsigned int CentroidalVoronoiDiagram<PFP>::moveSeed(unsigned int numSeed){

	// collect energy and compute the gradient
//	cumulateEnergyAndGradientFromSeed(numSeed);

	// select the new seed
	Dart newSeed = selectBestNeighborFromSeed(numSeed);

	// move the seed
	Dart oldSeed = this->seeds[numSeed];
	unsigned int seedMoved = 0;
	if (newSeed != oldSeed)
	{
		this->seeds[numSeed] = newSeed;
		seedMoved = 1;
	}

	return seedMoved;
}
*/

/*
template <typename PFP>
unsigned int CentroidalVoronoiDiagram<PFP>::moveSeed(unsigned int numSeed){
	Dart e = this->seeds[numSeed];
	unsigned int seedMoved = 0;

	std::vector<Dart> v;
	v.reserve(8);

	std::vector<float> da;
	da.reserve(8);

	distances[e] = 0.0;

	Traversor2VVaE<typename PFP::MAP> tv (this->map, e);
	for (Dart f = tv.begin(); f != tv.end(); f=tv.next())
	{
		if ( pathOrigins[f] == this->map.phi2(f))
		{
			float distArea = cumulateEnergyFromRoot(f);
			da.push_back(distArea);
			distances[e] += distances[f];
			v.push_back(f);
		}
	}

	// TODO : check if the computation of grad and proj is still valid for other edgeCost than geodesic distances
	VEC3 grad (0.0);
	const VertexAttribute<VEC3>& pos = this->map.template getAttribute<VEC3,VERTEX>("position");

	// compute the gradient
	for (unsigned int j = 0; j<v.size(); ++j)
	{
		Dart f = v[j];
		VEC3 edgeV = pos[f] - pos[this->map.phi2(f)];
		edgeV.normalize();
		grad += da[j] * edgeV;
	}
	grad /= 2.0;

	float maxProj = 0.0;
//	float memoForTest = 0.0;
	for (unsigned int j = 0; j<v.size(); ++j)
	{
		Dart f = v[j];
		VEC3 edgeV = pos[f] - pos[this->map.phi2(f)];
//		edgeV.normalize();
		float proj = edgeV * grad;
//		proj -= areaElts[e] * this->edgeCost[f] * this->edgeCost[f];
		if (proj > maxProj)
		{
//			if (numSeed==1) memoForTest = (edgeV * grad) / (areaElts[e] * this->edgeCost[f] * this->edgeCost[f]);
//				CGoGNout << (edgeV * grad) / (areaElts[e] * this->edgeCost[f] * this->edgeCost[f]) * this->seeds.size() << CGoGNendl;
//				CGoGNout << edgeV * grad << "\t - \t" << areaElts[e] * this->edgeCost[f] * this->edgeCost[f] << CGoGNendl;
			maxProj = proj;
			seedMoved = 1;
			this->seeds[numSeed] = v[j];
		}
	}

	return seedMoved;
}
*/

} // namespace Geometry

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
