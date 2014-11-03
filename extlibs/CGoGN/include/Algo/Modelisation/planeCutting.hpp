namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Modelisation
{

/*
template <typename PFP>
void planeCut(typename PFP::MAP& map, VertexAttribute<typename PFP::VEC3>& position, const Geom::Plane3D<typename PFP::REAL>& plane,
			  CellMarker<FACE>& cmf_over, bool keepTriangles, bool with_unsew)
{
	typedef typename PFP::REAL REAL;

	//marker for vertices on the plane
	CellMarker<VERTEX> cmv(map);

	// marker for vertices over the plane
	CellMarker<VERTEX> cmv_over(map);


	TraversorV<typename PFP::MAP> traVert(map);
	for (Dart d=traVert.begin(); d!=traVert.end();d=traVert.next())
	{
		Geom::Orientation3D or1 = plane.orient(position[d]);
		if (or1 == Geom::ON)
			cmv.mark(d);
		if (or1 == Geom::OVER)
			cmv_over.mark(d);
	}



	TraversorE<typename PFP::MAP> traEdg(map);
	for (Dart d=traEdg.begin(); d!=traEdg.end();d=traEdg.next())
	{
		Dart dd = map.phi1(d);


		if (!cmv.isMarked(d) && !cmv.isMarked(dd) && (cmv_over.isMarked(d) != cmv_over.isMarked(dd)))
		{
			Dart x = map.cutEdge(d);
			REAL dist1 = plane.distance(position[d]);
			REAL dist2 = plane.distance(position[dd]);

			if (dist1<0.0)
				dist1 = -dist1;
			if (dist2<0.0)			// no abs() to avoid type problem with REAL template
				dist2 = -dist2;

			position[x] = (position[d]*dist2 + position[dd]*dist1)/(dist1+dist2);

			traEdg.skip(x);
			traEdg.skip(map.phi_1(x));
			cmv.mark(x);
		}
	}

	Algo::Surface::Modelisation::EarTriangulation<PFP>* triangulator=NULL;
	if (keepTriangles)	// triangule faces if needed
	{
		triangulator = new Algo::Surface::Modelisation::EarTriangulation<PFP>(map);
	}

	TraversorF<typename PFP::MAP> traFac(map);
	for (Dart d=traFac.begin(); d!=traFac.end();d=traFac.next())
	{
		// turn in the face to search if there are 2 vertices marked as on the plane
		Traversor2FV<typename PFP::MAP> traV(map,d);
		Dart e=traV.begin();
		while ((e!=traV.end())&&(!cmv.isMarked(e)))
			e=traV.next();

		Dart V1=NIL;
		if (e!=traV.end())
			V1 = e;

		e=traV.next();
		while ((e!=traV.end())&&(!cmv.isMarked(e)))
			e=traV.next();

		Dart V2=NIL;
		if (e!=traV.end())
			V2 = e;

		// is there 2 vertices in the plane (but not consecutive)
		if ((V1!=NIL) && (V2!=NIL) && (V2!=map.phi1(V1)) && (V1!=map.phi1(V2)))
		{
			map.splitFace(V1,V2);
			if (with_unsew)
				map.unsewFaces(map.phi_1(V1)); // ne marche pas !

			// ensure to not scan this two new faces
			traFac.skip(V1);
			traFac.skip(V2);
			// mark face of V1 if it is over
			if (cmv_over.isMarked(map.phi1(V1)))
				cmf_over.mark(V1);
			// mark face of V2 if it is over
			if (cmv_over.isMarked(map.phi1(V2)))
				cmf_over.mark(V2);

			if (keepTriangles)	// triangule faces if needed
			{
				triangulator->trianguleFace(V1);
				triangulator->trianguleFace(V2);
			}
		}
		else
		{
			// find the first vertex not on the plane
			Dart e = d;
			while (cmv.isMarked(e))
				e = map.phi1(e);
			// face is all on same side than vertex
			if (cmv_over.isMarked(e))
				cmf_over.mark(e);
		}
	}

	if (triangulator != NULL)
		delete triangulator;
}
*/

template <typename PFP>
void planeCut(
	typename PFP::MAP& map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const Geom::Plane3D<typename PFP::REAL>& plane,
	CellMarker<typename PFP::MAP, FACE>& cmf_over,
	bool keepTriangles,
	bool with_unsew)
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::REAL REAL;

	//marker for vertices on the plane
	CellMarker<MAP, VERTEX> cmv(map);
	// marker for vertices over the plane
	CellMarker<MAP, VERTEX> cmv_over(map);

	TraversorE<MAP> traEdg(map);
	for (Dart d = traEdg.begin(); d != traEdg.end(); d = traEdg.next())
	{
		Dart dd = map.phi1(d);

		REAL dist1;
		REAL dist2;

		Geom::Orientation3D or1 = plane.orient(position[d],dist1);
		Geom::Orientation3D or2 = plane.orient(position[dd],dist2);

		if (or1 == Geom::ON)
			cmv.mark(d);

		if (or2 == Geom::ON)
			cmv.mark(dd);

		if ((or1!=Geom::ON) && (or2!=Geom::ON) && (or1 != or2))
		{
			Dart x = map.cutEdge(d);

			if (dist1<0.0)
				dist1 = -dist1;
			if (dist2<0.0)			// no abs() to avoid type problem with REAL template
				dist2 = -dist2;

			position[x] = (position[d]*dist2 + position[dd]*dist1)/(dist1+dist2);

			traEdg.skip(x);
			traEdg.skip(map.phi_1(x));
			cmv.mark(x);

			if (or1 == Geom::OVER)
				cmv_over.mark(d);
			else
				cmv_over.mark(dd);
		}
		else
		{
			if (or1 == Geom::OVER)
			{
				cmv_over.mark(d);
				cmv_over.mark(dd);
			}
		}
	}

	Algo::Surface::Modelisation::EarTriangulation<PFP>* triangulator=NULL;
	if (keepTriangles)	// triangule faces if needed
	{
		triangulator = new Algo::Surface::Modelisation::EarTriangulation<PFP>(map);
	}

	TraversorF<MAP> traFac(map);
	for (Dart d = traFac.begin(); d != traFac.end(); d = traFac.next())
	{
		// turn in the face to search if there are 2 vertices marked as on the plane
		Traversor2FV<MAP> traV(map,d);
		Dart e=traV.begin();
		while ((e != traV.end()) && (!cmv.isMarked(e)))
			e = traV.next();

		Dart V1 = NIL;
		if (e != traV.end())
			V1 = e;

		e = traV.next();
		while ((e != traV.end()) && (!cmv.isMarked(e)))
			e = traV.next();

		Dart V2 = NIL;
		if (e != traV.end())
			V2 = e;

		// is there 2 vertices in the plane (but not consecutive)
		if ((V1 != NIL) && (V2 != NIL) && (V2 != map.phi1(V1)) && (V1 != map.phi1(V2)))
		{
			map.splitFace(V1,V2);
			if (with_unsew)
				map.unsewFaces(map.phi_1(V1)); // ne marche pas !

			// ensure to not scan this two new faces
			traFac.skip(V1);
			traFac.skip(V2);
			// mark face of V1 if it is over
			if (cmv_over.isMarked(map.phi1(V1)))
				cmf_over.mark(V1);
			// mark face of V2 if it is over
			if (cmv_over.isMarked(map.phi1(V2)))
				cmf_over.mark(V2);

			if (keepTriangles)	// triangule faces if needed
			{
				triangulator->trianguleFace(V1);
				triangulator->trianguleFace(V2);
			}
		}
		else
		{
			// find the first vertex not on the plane
			Dart e = d;
			while (cmv.isMarked(e))
				e = map.phi1(e);
			// face is all on same side than vertex
			if (cmv_over.isMarked(e))
				cmf_over.mark(e);
		}
	}

	if (triangulator != NULL)
		delete triangulator;
}

template <typename PFP>
void planeCut2(
	typename PFP::MAP& map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const Geom::Plane3D<typename PFP::REAL>& plane,
	CellMarker<typename PFP::MAP, FACE>& cmf_over,
	bool with_unsew)
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::REAL REAL;

	//marker for vertices on the plane
	CellMarker<MAP, VERTEX> cmv(map);
	// marker for vertices over the plane
	CellMarker<MAP, VERTEX> cmv_over(map);

	EdgeAutoAttribute<VEC3, MAP> positionEdge(map);

	CellMarker<MAP, EDGE> cme(map);

	TraversorE<MAP> traEdg(map);
	for (Dart d = traEdg.begin(); d != traEdg.end(); d = traEdg.next())
	{
		Dart dd = map.phi1(d);

		REAL dist1;
		REAL dist2;

		Geom::Orientation3D or1 = plane.orient(position[d],dist1);
		Geom::Orientation3D or2 = plane.orient(position[dd],dist2);

		if (or1 == Geom::ON)
			cmv.mark(d);

		if (or2 == Geom::ON)
			cmv.mark(dd);

		if ((or1!=Geom::ON) && (or2!=Geom::ON) && (or1 != or2))
		{
			if (dist1<0.0)
				dist1 = -dist1;
			if (dist2<0.0)			// no abs() to avoid type problem with REAL template
				dist2 = -dist2;

			positionEdge[d] = (position[d]*dist2 + position[dd]*dist1)/(dist1+dist2);
			cme.mark(d);

			if (or1 == Geom::OVER)
				cmv_over.mark(d);
			else
				cmv_over.mark(dd);
		}
		else
		{
			if (or1 == Geom::OVER)
			{
				cmv_over.mark(d);
				cmv_over.mark(dd);
			}
		}
	}

	TraversorF<MAP> traFac(map);
	for (Dart d = traFac.begin(); d != traFac.end(); d = traFac.next())
	{
		// turn in the face to search if there are 2 edges marked as intersecting the plane
		Traversor2FE<MAP> traFE(map,d);
		Dart e=traFE.begin();
		while ((e!=traFE.end())&&(!cme.isMarked(e)))
			e=traFE.next();

		Dart E1=NIL;
		if (e!=traFE.end())
			E1 = e;

		e=traFE.next();
		while ((e!=traFE.end())&&(!cme.isMarked(e)))
			e=traFE.next();

		Dart E2=NIL;
		if (e!=traFE.end())
			E2 = e;

		// is there 2 edges intersecting the plane
		if ((E1!=NIL) && (E2!=NIL))// && (V2!=map.phi1(V1)) && (V1!=map.phi1(V2)))
		{
			Dart x = Algo::Surface::Modelisation::trianguleFace<PFP>(map,E1);
			position[x] = (positionEdge[E1] + positionEdge[E2] ) * 0.5;

			//ensure to not scan this three new faces
			traFac.skip(x);
			traFac.skip(map.phi2(x));
			traFac.skip(map.phi2(map.phi_1(x)));

			if (cmv_over.isMarked(E1))
				cmf_over.mark(map.phi2((map.phi_1(E1))));

			if (cmv_over.isMarked(E2))
				cmf_over.mark(map.phi2((map.phi_1(E2))));
		}
		else
		{
			// find the first vertex not on the plane
			Dart e = d;
			while (cmv.isMarked(e))
				e = map.phi1(e);
			// face is all on same side than vertex
			if (cmv_over.isMarked(e))
				cmf_over.mark(e);
		}
	}

	for (Dart d = traEdg.begin(); d != traEdg.end(); d = traEdg.next())
	{
		if(cme.isMarked(d))
		{
			map.flipBackEdge(d);
			if (with_unsew)
			{
				Dart d2 = map.phi2(d);
				map.unsewFaces(d); // ne marche pas !

				if(cmv_over.isMarked(map.phi_1(d)))
					cmf_over.mark(d);

				if(cmv_over.isMarked(map.phi_1(d2)))
					cmf_over.mark(d2);
			}
		}
	}

}

} // namespace Modelisation

} // namespace Surface

namespace Volume
{

namespace Modelisation
{

template <typename PFP>
void planeCut(
	typename PFP::MAP& map,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	const Geom::Plane3D<typename PFP::REAL>& plane,
	CellMarker<typename PFP::MAP, FACE>& cmf_over,
	bool keepTetrahedra,
	bool with_unsew)
{
	typedef typename PFP::MAP MAP;
	typedef typename PFP::MAP MAP;
	typedef typename PFP::REAL REAL;

	//marker for vertices on the plane
	CellMarker<MAP, VERTEX> cmv(map);
	// marker for vertices over the plane
	CellMarker<MAP, VERTEX> cmv_over(map);

	TraversorE<MAP> traEdg(map);
	for (Dart d = traEdg.begin(); d != traEdg.end(); d = traEdg.next())
	{
		Dart dd = map.phi1(d);

		REAL dist1;
		REAL dist2;

		Geom::Orientation3D or1 = plane.orient(position[d],dist1);
		Geom::Orientation3D or2 = plane.orient(position[dd],dist2);

		if (or1 == Geom::ON)
			cmv.mark(d);

		if (or2 == Geom::ON)
			cmv.mark(dd);

		if ((or1!=Geom::ON) && (or2!=Geom::ON) && (or1 != or2))
		{
			Dart x = map.cutEdge(d);

			if (dist1<0.0)
				dist1 = -dist1;
			if (dist2<0.0)			// no abs() to avoid type problem with REAL template
				dist2 = -dist2;

			position[x] = (position[d]*dist2 + position[dd]*dist1)/(dist1+dist2);

			traEdg.skip(x);
			traEdg.skip(map.phi_1(x));
			cmv.mark(x);

			if (or1 == Geom::OVER)
				cmv_over.mark(d);
			else
				cmv_over.mark(dd);
		}
		else
		{
			if (or1 == Geom::OVER)
			{
				cmv_over.mark(d);
				cmv_over.mark(dd);
			}
		}
	}

	TraversorW<MAP> traVol(map);
	for (Dart d = traVol.begin(); d != traVol.end(); d = traVol.next())
	{
		// turn in the volume to search if there are ? vertices marked as on the plane

	}
}

} // namespace Modelisation

} // namespace Volume

} // namespace Algo

} // namespace CGoGN
