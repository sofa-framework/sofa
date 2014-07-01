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

#include "Topology/generic/traversor/traversorCell.h"
#include "Topology/generic/traversor2.h"

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Filtering
{

/**
 * compute new position of vertices from normals (normalAverage & MMSE filters)
 * @param map the map
 */
template <typename PFP>
void computeNewPositionsFromFaceNormals(
	typename PFP::MAP& map,
	const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position,
	VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position2,
	const FaceAttribute<typename PFP::REAL, typename PFP::MAP>& faceArea,
	const FaceAttribute<typename PFP::VEC3, typename PFP::MAP>& faceCentroid,
	const FaceAttribute<typename PFP::VEC3, typename PFP::MAP>& faceNormal,
	const FaceAttribute<typename PFP::VEC3, typename PFP::MAP>& faceNewNormal)
{
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	TraversorV<typename PFP::MAP> t(map) ;
	for(Dart d = t.begin(); d != t.end(); d = t.next())
	{
		const VEC3& pos_d = position[d] ;

		VEC3 displ(0) ;
		REAL sumAreas = 0 ;

		Traversor2VF<typename PFP::MAP> tvf(map, d) ;
		for(Dart it = tvf.begin(); it != tvf.end(); it = tvf.next())
		{
			sumAreas += faceArea[it] ;
			VEC3 vT = faceCentroid[it] - pos_d ;
			vT = (vT * faceNewNormal[it]) * faceNormal[it] ;
			displ += faceArea[it] * vT ;
		}

		displ /= sumAreas ;
		position2[d] = pos_d + displ ;
	}
}

template <typename PFP>
void filterAverageNormals(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position2)
{
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	FaceAutoAttribute<REAL, MAP> faceArea(map, "faceArea") ;
	FaceAutoAttribute<VEC3, MAP> faceNormal(map, "faceNormal") ;
	FaceAutoAttribute<VEC3, MAP> faceCentroid(map, "faceCentroid") ;

	Algo::Surface::Geometry::computeAreaFaces<PFP>(map, position, faceArea) ;
	Algo::Surface::Geometry::computeNormalFaces<PFP>(map, position, faceNormal) ;
	Algo::Surface::Geometry::computeCentroidFaces<PFP>(map, position, faceCentroid) ;

	FaceAutoAttribute<VEC3, MAP> faceNewNormal(map, "faceNewNormal") ;

	// Compute new normals
	TraversorF<typename PFP::MAP> tf(map) ;
	for(Dart d = tf.begin(); d != tf.end(); d = tf.next())
	{
		REAL sumArea = 0 ;
		VEC3 meanFilter(0) ;

		// traversal of adjacent faces (by edges and vertices)
		Traversor2FFaV<typename PFP::MAP> taf(map, d) ;
		for(Dart it = taf.begin(); it != taf.end(); it = taf.next())
		{
			sumArea += faceArea[it] ;
			meanFilter += faceArea[it] * faceNormal[it] ;
		}

		// finalize the computation of meanFilter normal
		meanFilter /= sumArea ;
		meanFilter.normalize() ;
		// and store it
		faceNewNormal[d] = meanFilter ;
	}

	// Compute new vertices position
	computeNewPositionsFromFaceNormals<PFP>(
		map, position, position2, faceArea, faceCentroid, faceNormal, faceNewNormal) ;
}

template <typename PFP>
void filterMMSE(typename PFP::MAP& map, float sigmaN2, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position2)
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	FaceAutoAttribute<REAL, MAP> faceArea(map, "faceArea") ;
	FaceAutoAttribute<VEC3, MAP> faceNormal(map, "faceNormal") ;
	FaceAutoAttribute<VEC3, MAP> faceCentroid(map, "faceCentroid") ;

	Algo::Surface::Geometry::computeAreaFaces<PFP>(map, position, faceArea) ;
	Algo::Surface::Geometry::computeNormalFaces<PFP>(map, position, faceNormal) ;
	Algo::Surface::Geometry::computeCentroidFaces<PFP>(map, position, faceCentroid) ;

	FaceAutoAttribute<VEC3, MAP> faceNewNormal(map, "faceNewNormal") ;

	// Compute new normals
	TraversorF<typename PFP::MAP> tf(map) ;
	for(Dart d = tf.begin(); d != tf.end(); d = tf.next())
	{
		// traversal of neighbour vertices
		REAL sumArea = 0 ;
		REAL sigmaX2 = 0 ;
		REAL sigmaY2 = 0 ;
		REAL sigmaZ2 = 0 ;

		VEC3 meanFilter(0) ;

		// traversal of adjacent faces (by edges and vertices)
		Traversor2FFaV<typename PFP::MAP> taf(map, d) ;
		for(Dart it = taf.begin(); it != taf.end(); it = taf.next())
		{
			// get info from face embedding and sum
			REAL area = faceArea[it] ;
			sumArea += area ;
			VEC3 normal = faceNormal[it] ;
			meanFilter += area * normal ;
			sigmaX2 += area * normal[0] * normal[0] ;
			sigmaY2 += area * normal[1] * normal[1] ;
			sigmaZ2 += area * normal[2] * normal[2] ;
		}

		meanFilter /= sumArea ;
		sigmaX2 /= sumArea ;
		sigmaX2 -= meanFilter[0] * meanFilter[0] ;
		sigmaY2 /= sumArea ;
		sigmaY2 -= meanFilter[1] * meanFilter[1] ;
		sigmaZ2 /= sumArea ;
		sigmaZ2 -= meanFilter[2] * meanFilter[2] ;

		VEC3& oldNormal = faceNormal[d] ;
		VEC3 newNormal ;

		if(sigmaX2 < sigmaN2)
			newNormal[0] = meanFilter[0] ;
		else
		{
			newNormal[0] = (1 - (sigmaN2 / sigmaX2)) * oldNormal[0] ;
			newNormal[0] += (sigmaN2 / sigmaX2) * meanFilter[0] ;
		}
		if(sigmaY2 < sigmaN2)
			newNormal[1] = meanFilter[1] ;
		else
		{
			newNormal[1] = (1 - (sigmaN2 / sigmaY2)) * oldNormal[1] ;
			newNormal[1] += (sigmaN2 / sigmaY2) * meanFilter[1] ;
		}
		if(sigmaZ2 < sigmaN2)
			newNormal[2] = meanFilter[2] ;
		else
		{
			newNormal[2] = (1 - (sigmaN2 / sigmaZ2)) * oldNormal[2] ;
			newNormal[2] += (sigmaN2 / sigmaZ2) * meanFilter[2] ;
		}

		newNormal.normalize() ;
		faceNewNormal[d] = newNormal ;
	}

	// Compute new vertices position
	computeNewPositionsFromFaceNormals<PFP>(
		map, position, position2, faceArea, faceCentroid, faceNormal, faceNewNormal) ;
}

template <typename PFP>
void filterTNBA(typename PFP::MAP& map, float sigmaN2, float SUSANthreshold, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position2)
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	FaceAutoAttribute<REAL, MAP> faceArea(map, "faceArea") ;
	FaceAutoAttribute<VEC3, MAP> faceNormal(map, "faceNormal") ;
	FaceAutoAttribute<VEC3, MAP> faceCentroid(map, "faceCentroid") ;

	Algo::Surface::Geometry::computeAreaFaces<PFP>(map, position, faceArea) ;
	Algo::Surface::Geometry::computeNormalFaces<PFP>(map, position, faceNormal) ;
	Algo::Surface::Geometry::computeCentroidFaces<PFP>(map, position, faceCentroid) ;

	FaceAutoAttribute<VEC3, MAP> faceNewNormal(map, "faceNewNormal") ;

	// Compute new normals
	long nbTot = 0 ;
	long nbAdapt = 0 ;
	long nbSusan = 0 ;

	TraversorF<typename PFP::MAP> tf(map) ;
	for(Dart d = tf.begin(); d != tf.end(); d = tf.next())
	{
		const VEC3& normF = faceNormal[d] ;

		// traversal of neighbour vertices
		REAL sumArea = 0 ;
		REAL sigmaX2 = 0 ;
		REAL sigmaY2 = 0 ;
		REAL sigmaZ2 = 0 ;

		VEC3 meanFilter(0) ;
		bool SUSANregion = false ;

		// traversal of adjacent faces (by edges and vertices)
		Traversor2FFaV<typename PFP::MAP> taf(map, d) ;
		for(Dart it = taf.begin(); it != taf.end(); it = taf.next())
		{
			// get info from face embedding and sum
			const VEC3& normal = faceNormal[it] ;

			float angle = Geom::angle(normF, normal) ;
			if(angle <= SUSANthreshold)
			{
				REAL area = faceArea[it] ;
				sumArea += area ;
				meanFilter += area * normal ;
				sigmaX2 += area * normal[0] * normal[0] ;
				sigmaY2 += area * normal[1] * normal[1] ;
				sigmaZ2 += area * normal[2] * normal[2] ;
			}
			else SUSANregion = true ;
		}

		if(SUSANregion)
			++nbSusan ;

		++nbTot ;

		if(sumArea > 0.0f)
		{
			meanFilter /= sumArea ;
			sigmaX2 /= sumArea ;
			sigmaX2 -= meanFilter[0] * meanFilter[0] ;
			sigmaY2 /= sumArea ;
			sigmaY2 -= meanFilter[1] * meanFilter[1] ;
			sigmaZ2 /= sumArea ;
			sigmaZ2 -= meanFilter[2] * meanFilter[2] ;

			VEC3& oldNormal = faceNormal[d] ;
			VEC3 newNormal ;

			bool adapt = false ;
			if(sigmaX2 < sigmaN2)
				newNormal[0] = meanFilter[0] ;
			else
			{
				adapt = true ;
				newNormal[0] = (1 - (sigmaN2 / sigmaX2)) * oldNormal[0] ;
				newNormal[0] += (sigmaN2 / sigmaX2) * meanFilter[0] ;
			}
			if(sigmaY2 < sigmaN2)
				newNormal[1] = meanFilter[1] ;
			else
			{
				adapt = true ;
				newNormal[1] = (1 - (sigmaN2 / sigmaY2)) * oldNormal[1] ;
				newNormal[1] += (sigmaN2 / sigmaY2) * meanFilter[1] ;
			}
			if(sigmaZ2 < sigmaN2)
				newNormal[2] = meanFilter[2] ;
			else
			{
				adapt = true ;
				newNormal[2] = (1 - (sigmaN2 / sigmaZ2)) * oldNormal[2] ;
				newNormal[2] += (sigmaN2 / sigmaZ2) * meanFilter[2] ;
			}
			if(adapt)
				++nbAdapt ;

			newNormal.normalize() ;
			faceNewNormal[d] = newNormal;
		}
		else
		{
			faceNewNormal[d] = normF ;
		}
	}

	// Compute new vertices position
	computeNewPositionsFromFaceNormals<PFP>(
		map, position, position2, faceArea, faceCentroid, faceNormal, faceNewNormal) ;

//	CGoGNout <<" susan rate = "<< float(nbSusan)/float(nbTot)<<CGoGNendl;
//	CGoGNout <<" adaptive rate = "<< float(nbAdapt)/float(nbTot)<<CGoGNendl;
}

template <typename PFP>
void filterVNBA(typename PFP::MAP& map, float sigmaN2, float SUSANthreshold, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position, VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& position2, const VertexAttribute<typename PFP::VEC3, typename PFP::MAP>& normal)
{
	typedef typename PFP::MAP MAP ;
	typedef typename PFP::VEC3 VEC3 ;
	typedef typename PFP::REAL REAL ;

	FaceAutoAttribute<REAL, MAP> faceArea(map, "faceArea") ;
	FaceAutoAttribute<VEC3, MAP> faceNormal(map, "faceNormal") ;
	FaceAutoAttribute<VEC3, MAP> faceCentroid(map, "faceCentroid") ;

	Algo::Surface::Geometry::computeAreaFaces<PFP>(map, position, faceArea) ;
	Algo::Surface::Geometry::computeNormalFaces<PFP>(map, position, faceNormal) ;
	Algo::Surface::Geometry::computeCentroidFaces<PFP>(map, position, faceCentroid) ;

	VertexAutoAttribute<REAL, MAP> vertexArea(map, "vertexArea") ;
	FaceAutoAttribute<VEC3, MAP> faceNewNormal(map, "faceNewNormal") ;
	VertexAutoAttribute<VEC3, MAP> vertexNewNormal(map, "vertexNewNormal") ;

	long nbTot = 0 ;
	long nbAdapt = 0 ;
	long nbSusan = 0 ;

	TraversorV<typename PFP::MAP> tv(map) ;
	for(Dart d = tv.begin(); d != tv.end(); d = tv.next())
	{
		const VEC3& normV = normal[d] ;

		REAL sumArea = 0 ;
		REAL sigmaX2 = 0 ;
		REAL sigmaY2 = 0 ;
		REAL sigmaZ2 = 0 ;

		VEC3 meanFilter(0) ;

		bool SUSANregion = false ;

		// traversal of neighbour vertices
		Traversor2VVaE<typename PFP::MAP> tav(map, d) ;
		for(Dart it = tav.begin(); it != tav.end(); it = tav.next())
		{
			const VEC3& neighborNormal = normal[it] ;
			float angle = Geom::angle(normV, neighborNormal) ;
			if( angle <= SUSANthreshold )
			{
				REAL umbArea = Algo::Surface::Geometry::vertexOneRingArea<PFP>(map, it, position) ;
				vertexArea[it] = umbArea ;

				sumArea += umbArea ;
				sigmaX2 += umbArea * neighborNormal[0] * neighborNormal[0] ;
				sigmaY2 += umbArea * neighborNormal[1] * neighborNormal[1] ;
				sigmaZ2 += umbArea * neighborNormal[2] * neighborNormal[2] ;
				meanFilter += neighborNormal * umbArea ;
			}
			else SUSANregion = true ;
		}

		if(SUSANregion)
			++nbSusan ;

		++nbTot ;

		if(sumArea > 0.0f)
		{
			meanFilter /= sumArea ;
			sigmaX2 /= sumArea ;
			sigmaX2 -= meanFilter[0] * meanFilter[0] ;
			sigmaY2 /= sumArea ;
			sigmaY2 -= meanFilter[1] * meanFilter[1] ;
			sigmaZ2 /= sumArea ;
			sigmaZ2 -= meanFilter[2] * meanFilter[2] ;

			VEC3 newNormal ;
			bool adapt = false ;
			if(sigmaX2 < sigmaN2)
				newNormal[0] = meanFilter[0] ;
			else
			{
				adapt = true ;
				newNormal[0] = (1 - (sigmaN2 / sigmaX2)) * normV[0] ;
				newNormal[0] += (sigmaN2 / sigmaX2) * meanFilter[0] ;
			}
			if(sigmaY2 < sigmaN2)
				newNormal[1] = meanFilter[1] ;
			else
			{
				adapt = true ;
				newNormal[1] = (1 - (sigmaN2 / sigmaY2)) * normV[1] ;
				newNormal[1] += (sigmaN2 / sigmaY2) * meanFilter[1] ;
			}
			if(sigmaZ2 < sigmaN2)
				newNormal[2] = meanFilter[2] ;
			else
			{
				adapt = true ;
				newNormal[2] = (1 - (sigmaN2 / sigmaZ2)) * normV[2] ;
				newNormal[2] += (sigmaN2 / sigmaZ2) * meanFilter[2] ;
			}

			if(adapt)
				++nbAdapt ;

			newNormal.normalize() ;
			vertexNewNormal[d] = newNormal ;
		}
		else
		{
			vertexNewNormal[d] = normV ;
		}
	}

	// Compute face normals from vertex normals
	TraversorF<typename PFP::MAP> tf(map) ;
	for(Dart d = tf.begin(); d != tf.end(); d = tf.next())
	{
		VEC3 newNormal(0) ;
		REAL totArea = 0 ;
		Traversor2FV<typename PFP::MAP> tav(map, d) ;
		for(Dart it = tav.begin(); it != tav.end(); it = tav.next())
		{
			VEC3 vNorm = vertexNewNormal[it] ;
			REAL area = vertexArea[it] ;
			totArea += area ;
			vNorm *= area ;
			newNormal += vNorm ;
		}
		newNormal.normalize() ;
		faceNewNormal[d] = newNormal ;
	}

	// Compute new vertices position
	computeNewPositionsFromFaceNormals<PFP>(
		map, position, position2, faceArea, faceCentroid, faceNormal, faceNewNormal	) ;

//	CGoGNout <<" susan rate = "<< float(nbSusan)/float(nbTot)<<CGoGNendl;
//	CGoGNout <<" adaptive rate = "<< float(nbAdapt)/float(nbTot)<<CGoGNendl;
}

} // namespace Filtering

} // namespace Surface

} // namespace Algo

} // namespace CGoGN
