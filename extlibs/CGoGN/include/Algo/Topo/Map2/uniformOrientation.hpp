namespace CGoGN
{

namespace Algo
{

namespace Topo
{

// private function
template <typename MAP>
void reverse2MapFaceKeepPhi2(MAP& map, Dart d)
{
	unsigned int first = map.template getEmbedding<VERTEX>(d);

	Dart e=d;
	do
	{
		Dart f=map.phi1(e);
		unsigned int emb = map.template getEmbedding<VERTEX>(f);
		map.template setDartEmbedding<VERTEX>(e,emb);
		e =f;
	}while (e!=d);
	map.template setDartEmbedding<VERTEX>(map.phi_1(d),first);

	map.reverseCycle(d);

}

// private function
inline Dart findOtherInCouplesOfDarts(const std::vector<Dart>& couples, Dart d)
{
	unsigned int nb = couples.size();
	for (unsigned int i=0; i<nb; ++i)
	{
		if (couples[i] == d)
		{
			if (i%2 == 0)
				return couples[i+1];
			//else
				return couples[i-1];
		}
	}
	return NIL;
}



template <typename MAP>
void uniformOrientationCC(MAP& map, Dart faceSeed)
{
	// first bufferize boundary edges
	std::vector<Dart> boundEdges;
	TraversorE<MAP> travEdge(map);
	for (Dart d=travEdge.begin(); d!= travEdge.end(); d = travEdge.next())
	{
		if (map.isBoundaryEdge(d))
			boundEdges.push_back(d);
	}

	// store couple of boundary edges the have same embedding
	std::vector<Dart> couples;
	int nb = boundEdges.size();
	int nbm = nb-1;
	for (int i=0; i< nbm; ++i)
	{
		if (boundEdges[i] != NIL)
		{
			for (int j=i+1; j< nb; ++j)
			{
				if (boundEdges[j] != NIL)
				{
					Dart d = boundEdges[i];
					Dart d1 = map.phi1(d);
					Dart e = boundEdges[j];
					Dart e1 = map.phi1(e);

					if ((map.template getEmbedding<VERTEX>(d) == map.template getEmbedding<VERTEX>(e)) && (map.template getEmbedding<VERTEX>(d1) == map.template getEmbedding<VERTEX>(e1)))
					{
						couples.push_back(d);
						couples.push_back(e);
						boundEdges[j] = NIL; // not using the dart again
						j=nb; // out of the loop
					}
				}
			}
		}
	}


	// vector of front propagation for good orientation
	std::vector<Dart> propag;
	boundEdges.clear();
	propag.swap(boundEdges);// reused memory of boundEdges


	// vector of front propagation for wrong orientation
	std::vector<Dart> propag_inv;

	//vector of faces to invert
	std::vector<Dart> face2invert;

	// optimize memory reserve
	propag_inv.reserve(1024);
	face2invert.reserve(1024);

	DartMarker<MAP> cmf(map);

	cmf.markOrbit<FACE>(faceSeed);
	propag.push_back(faceSeed);

	while (!propag.empty() || !propag_inv.empty())
	{
		if (!propag.empty())
		{
			Dart f = propag.back();
			propag.pop_back();

			Dart d = f;
			do
			{
				Dart e = map.phi2(d);
				if (map.isBoundaryMarked2(e))
				{
					e = findOtherInCouplesOfDarts(couples,d);
					if (e!=NIL)
					{
						if (!cmf.isMarked(e))
						{
							propag_inv.push_back(e);
							face2invert.push_back(e);
							cmf.markOrbit<FACE>(e);
						}
						cmf.mark(map.phi2(e));// use cmf also to mark boudary cycle to invert
					}

				}
				else
				{
					if (!cmf.isMarked(e))
					{
						propag.push_back(e);
						cmf.markOrbit<FACE>(e);
					}
				}
				d= map.phi1(d);

			} while (d!=f);
		}

		if (!propag_inv.empty())
		{
			Dart f = propag_inv.back();
			propag_inv.pop_back();

			Dart d = f;
			do
			{
				Dart e = map.phi2(d);
				if (map.isBoundaryMarked2(e))
				{
					e = findOtherInCouplesOfDarts(couples,d);
					if (e!=NIL)
					{
						if (!cmf.isMarked(e))
						{
							propag.push_back(e);
							cmf.markOrbit<FACE>(e);
						}
						cmf.mark(map.phi2(d));// use cmf also to mark boudary cycle to invert
					}
				}
				else
				{
					if (!cmf.isMarked(e))
					{
						propag_inv.push_back(e);
						face2invert.push_back(e);
						cmf.markOrbit<FACE>(e);
					}
				}
				d= map.phi1(d); // traverse all edges of face
			} while (d!=f);
		}
	}

	// reverse all faces of the wrong orientation
	for (std::vector<Dart>::iterator id=face2invert.begin(); id!=face2invert.end(); ++id)
		reverse2MapFaceKeepPhi2<MAP>(map,*id);

	// reverse the boundary cycles that bound reverse faces
	for (std::vector<Dart>::iterator id=couples.begin(); id!=couples.end(); ++id)
	{
		Dart e =  map.phi2(*id);
		if (cmf.isMarked(e))	// check cmf for wrong orientation
		{
			reverse2MapFaceKeepPhi2<MAP>(map,e);
			cmf.unmarkOrbit<FACE>(e);
		}
	}

	// sew the faces that correspond to couples of boundary edges
	for (std::vector<Dart>::iterator id=couples.begin(); id!=couples.end(); ++id)// second ++ inside !
	{
		Dart d = *id++;
		Dart e = *id;
		if (cmf.isMarked(d) || cmf.isMarked(e))
			map.sewFaces(d,e);
	}

}


} // namespace Topo

} // namespace Algo

} // namespace CGoGN
