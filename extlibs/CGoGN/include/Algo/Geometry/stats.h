/*******************************************************************************
* CGoGN: Combinatorial and Geometric modeling with Generic N-dimensional Maps  *
* version 0.1                                                                  *
* Copyright (C) 2009-2012, IGG Team, LSIIT, University of Strasbourg           *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Web site: http://cgogn.unistra.fr/                                           *
* Contact information: cgogn@unistra.fr                                        *
*                                                                              *
*******************************************************************************/

#ifndef STATS_H
#define STATS_H

namespace CGoGN
{

namespace Algo
{

namespace Geometry
{

template <typename PFP>
void statModele(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position)
{
	int nbFaces = 0;
	int nbVertex = 0;

	CellMarker<typename PFP::MAP, VERTEX> mVertex(map);

	float ratioMinMax = 0;
	int nbEdgePerVertex = 0;
	float lengthSeg = 0;
	int nbEdge = 0;

	TraversorF<typename PFP::MAP> tf(map) ;
	for(Dart d = tf.begin(); d != tf.end(); tf.next(d))
	{
		nbFaces++;
		bool init = true;
		float min = 0;
		float max = 0;

		Traversor2FV<typename PFP::MAP> tfe(map, d) ;
		for(Dart it = tfe.begin(); it != tfe.end(); it = tfe.next())
		{
			typename PFP::VEC3 segment = position[it] - position[map.phi1(it)] ;

			float len = segment.norm() ;

			lengthSeg += len;
			nbEdge++;

			if (init || len < min)
				min = len;
			if (init || len > max)
				max = len;

			init = false;

			if (!mVertex.isMarked(it))
			{
				mVertex.mark(it) ;
				nbVertex++ ;
				Traversor2VE<typename PFP::MAP> tve(map, it) ;
				for(Dart it2 = tve.begin(); it2 != tve.end(); it2 = tve.next())
					nbEdgePerVertex++ ;
			}
		}

		ratioMinMax += (min / max);
	}

//	for (Dart d = map.begin(); d != map.end(); map.next(d))
//	{
//		if (!mFace.isMarked(d))
//		{
//			nbFaces++;
//			bool init = true;
//			float min = 0;
//			float max = 0;
//			Dart e = d;
//			do
//			{
//				mFace.mark(e);
//				typename PFP::VEC3 segment = position[e] - position[map.phi1(e)] ;
//
//				float len = segment.norm() ;
//
//				lengthSeg += len;
//				nbEdge++;
//
//				if (init || len < min)
//					min = len;
//				if (init || len > max)
//					max = len;
//
//				init = false;
//				e = map.phi1(e);
//			}
//			while (e != d);
//
//			ratioMinMax += (min / max);
//		}
//
//		if (!mVertex.isMarked(d))
//		{
//			mVertex.mark(d) ;
//			nbVertex++ ;
//			Dart e = d;
//			do
//			{
//				nbEdgePerVertex++ ;
//				e = map.phi2_1(e) ;
//			}
//			while (e != d) ;
//		}
//	}

	CGoGNout << "number of faces                : " << nbFaces << CGoGNendl;
	CGoGNout << "number of vertices             : " << nbVertex << CGoGNendl;
	CGoGNout << "mean ratio min max             : " << (ratioMinMax / (float) nbFaces) << CGoGNendl;
	CGoGNout << "mean number of edge per vertex : " << ((float) nbEdgePerVertex / (float) nbVertex) << CGoGNendl;
	CGoGNout << "mean edge length               : " << lengthSeg / (float) nbEdge << CGoGNendl;
}

} // namespace Geometry

} // namespace Algo

} // namespace CGoGN

#endif
