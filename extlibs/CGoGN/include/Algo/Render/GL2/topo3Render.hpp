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

#include "Geometry/vector_gen.h"
#include "Topology/generic/autoAttributeHandler.h"
#include "Topology/generic/dartmarker.h"
#include "Topology/generic/cellmarker.h"

#include "Topology/map/map3.h"
#include "Topology/gmap/gmap3.h"

#include "Topology/generic/traversorCell.h"
#include "Algo/Geometry/centroid.h"

#include "Geometry/distances.h"

namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace GL2
{
//template<typename PFP>
//void Topo3Render::updateData(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& positions, float ke, float kf, float kv)
//{
//	Map3* ptrMap3 = dynamic_cast<Map3*>(&map);
//	if (ptrMap3 != NULL)
//	{
//		updateDataMap3<PFP, VertexAttribute<typename PFP::VEC3>, typename PFP::VEC3 >(map,positions,ke,kf,kv);
//	}
//	GMap3* ptrGMap3 = dynamic_cast<GMap3*>(&map);
//	if (ptrGMap3 != NULL)
//	{
//		updateDataGMap3<PFP, VertexAttribute<typename PFP::VEC3>, typename PFP::VEC3>(map,positions,ke,kf,kv);
//	}
//}


template<typename PFP, typename EMBV>
void Topo3Render::updateData(typename PFP::MAP& map, const EMBV& positions, float ke, float kf, float kv)
{
	Map3* ptrMap3 = dynamic_cast<Map3*>(&map);
	if (ptrMap3 != NULL)
	{
		updateDataMap3<PFP,EMBV>(map,positions,ke,kf,kv);
	}
	GMap3* ptrGMap3 = dynamic_cast<GMap3*>(&map);
	if (ptrGMap3 != NULL)
	{
		updateDataGMap3<PFP,EMBV>(map,positions,ke,kf,kv);
	}
}




template<typename PFP, typename EMBV>
void Topo3Render::updateDataMap3(typename PFP::MAP& mapx, const EMBV& positions, float ke, float kf, float kv)
{
	typedef typename EMBV::DATA_TYPE VEC3;
	typedef typename PFP::REAL REAL;

	m_attIndex  = mapx.template getAttribute<unsigned int, DART>("dart_index3");

	if (!m_attIndex.isValid())
		m_attIndex  = mapx.template addAttribute<unsigned int, DART>("dart_index3");

	m_nbDarts = 0;
	for (Dart d = mapx.begin(); d != mapx.end(); mapx.next(d))
	{
		if (!mapx.isBoundaryMarked3(d)) // in the following code Traversor do not traverse boundary
			m_nbDarts++;
	}

	// compute center of each volumes
	CellMarker<VOLUME> cmv(mapx);
	VolumeAutoAttribute<VEC3> centerVolumes(mapx, "centerVolumes");

	Algo::Volume::Geometry::Parallel::computeCentroidELWVolumes<PFP>(mapx, positions, centerVolumes,3);

	// debut phi1
	DartAutoAttribute<VEC3> fv1(mapx);
	// fin phi1
	DartAutoAttribute<VEC3> fv11(mapx);

	// phi2
	DartAutoAttribute<VEC3> fv2(mapx);
	DartAutoAttribute<VEC3> fv2x(mapx);

	m_vbo4->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(VEC3), 0, GL_STREAM_DRAW);
	GLvoid* ColorDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	VEC3* colorDartBuf = reinterpret_cast<VEC3*>(ColorDartsBuffer);

	m_vbo0->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(VEC3), 0, GL_STREAM_DRAW);
	GLvoid* PositionDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	VEC3* positionDartBuf = reinterpret_cast<VEC3*>(PositionDartsBuffer);


	std::vector<Dart> vecDartFaces;
	vecDartFaces.reserve(m_nbDarts/3);
	unsigned int posDBI=0;

	// traverse each face of each volume
	TraversorCell<typename PFP::MAP, PFP::MAP::FACE_OF_PARENT> traFace(mapx);
	for (Dart d = traFace.begin(); d != traFace.end(); d = traFace.next())
	{
		vecDartFaces.push_back(d);

		std::vector<VEC3> vecPos;
		vecPos.reserve(16);

		// store the face & center
		float okv = 1.0f - kv;

		VEC3 vc = centerVolumes[d];
		
		VEC3 centerFace = Algo::Surface::Geometry::faceCentroidELW<PFP>(mapx,d,positions)*kv +vc*okv;

		//shrink the face
		float okf = 1.0f - kf;
		Dart dd = d;
		do
		{
			VEC3 P = centerFace*okf + (vc*okv + positions[dd]*kv)*kf;
			vecPos.push_back(P);
			dd = mapx.phi1(dd);
		} while (dd != d);
		
		unsigned int nb = vecPos.size();
		
		vecPos.push_back(vecPos.front()); // copy the first for easy computation on next loop

		// compute position of points to use for drawing topo
		float oke = 1.0f - ke;
		for (unsigned int i = 0; i < nb; ++i)
		{
			VEC3 P = vecPos[i]*ke + vecPos[i+1]*oke;
			VEC3 Q = vecPos[i+1]*ke + vecPos[i]*oke;

			m_attIndex[d] = posDBI;
			posDBI+=2;

			*positionDartBuf++ = P;
			*positionDartBuf++ = Q;
			*colorDartBuf++ = m_dartsColor;
			*colorDartBuf++ = m_dartsColor;

			fv1[d] = P*0.1f + Q*0.9f;
			fv11[d] = P*0.9f + Q*0.1f;

			fv2[d] = P*0.52f + Q*0.48f;
			fv2x[d] = P*0.48f + Q*0.52f;
			d = mapx.phi1(d);
		}
	}

	m_vbo0->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	m_vbo4->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);


	VEC3* positioniF1 = new VEC3[ 2*m_nbDarts];
	VEC3* positioniF2 = new VEC3[ 2*m_nbDarts];
	VEC3* positioniF3 = new VEC3[ 2*m_nbDarts];

	VEC3* positionF1 = positioniF1;
	VEC3* positionF2 = positioniF2;
	VEC3* positionF3 = positioniF3;

	m_nbRel1=0;
	m_nbRel2=0;
	m_nbRel3=0;

	for(std::vector<Dart>::iterator face = vecDartFaces.begin(); face != vecDartFaces.end(); ++face)
	{
		Dart d = *face;
		do
		{
			Dart e = mapx.phi2(d);
			if ((d < e))
			{
				*positionF2++ = fv2[d];
				*positionF2++ = fv2x[e];
				*positionF2++ = fv2[e];
				*positionF2++ = fv2x[d];
				m_nbRel2++;
			}
			e = mapx.phi3(d);
			if (!mapx.isBoundaryMarked3(e) && (d < e) )
			{
				*positionF3++ = fv2[d];
				*positionF3++ = fv2x[e];
				*positionF3++ = fv2[e];
				*positionF3++ = fv2x[d];
				m_nbRel3++;
			}
			e = mapx.phi1(d);
			*positionF1++ = fv1[d];
			*positionF1++ = fv11[e];
			m_nbRel1++;

			d = mapx.phi1(d);
		} while (d != *face );
	}

	m_vbo3->bind();
	glBufferData(GL_ARRAY_BUFFER, 4*m_nbRel3*sizeof(typename PFP::VEC3), positioniF3, GL_STREAM_DRAW);

	m_vbo2->bind();
	glBufferData(GL_ARRAY_BUFFER, 4*m_nbRel2*sizeof(typename PFP::VEC3), positioniF2, GL_STREAM_DRAW);

	m_vbo1->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbRel1*sizeof(typename PFP::VEC3), positioniF1, GL_STREAM_DRAW);

	delete[] positioniF1;
	delete[] positioniF2;
	delete[] positioniF3;
}

template<typename PFP>
void Topo3Render::setDartsIdColor(typename PFP::MAP& map)
{
	m_vbo4->bind();
	float* colorBuffer =  reinterpret_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
	unsigned int nb=0;

	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if ( !map.isBoundaryMarked3(d)) // topo3 Render do not traverse boundary
		{
			if (nb < m_nbDarts)
			{
				float r,g,b;
				dartToCol(d, r,g,b);

				float* local = colorBuffer+3*m_attIndex[d]; // get the right position in VBO
				*local++ = r;
				*local++ = g;
				*local++ = b;
				*local++ = r;
				*local++ = g;
				*local++ = b;
				nb++;
			}
			else
			{
				CGoGNerr << "Error buffer too small for color picking (change the selector parameter ?)" << CGoGNendl;
				break;
			}
		}
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

template<typename PFP, typename EMBV>
void Topo3Render::updateColorsGen(typename PFP::MAP& map, const EMBV& colors)
{
	typedef typename EMBV::DATA_TYPE EMB;

	m_vbo4->bind();
	EMB* colorBuffer =  reinterpret_cast<EMB*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
	unsigned int nb=0;

	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if ( !map.isBoundaryMarked3(d)) // topo3 Render do not traverse boundary
		{
			if (nb < m_nbDarts)
			{
				colorBuffer[m_attIndex[d]] = colors[d];
				nb++;
			}
			else
			{
				CGoGNerr << "Error buffer too small for color picking (change the selector parameter ?)" << CGoGNendl;
				break;
			}
		}
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

template<typename PFP>
void Topo3Render::updateColors(typename PFP::MAP& map, const VertexAttribute<Geom::Vec3f>& colors)
{
//	updateColorsGen<PFP, VertexAttribute<Geom::Vec3f> >(map,colors);
	updateColorsGen<PFP>(map,colors);
}

template<typename PFP>
Dart Topo3Render::picking(typename PFP::MAP& map, int x, int y)
{
	pushColors();
	setDartsIdColor<PFP>(map);
	Dart d = pickColor(x,y);
	popColors();
	return d;
}

template<typename PFP, typename EMBV>
void Topo3Render::updateDataGMap3(typename PFP::MAP& mapx, const EMBV& positions, float ke, float kf, float kv)
{
	typedef typename EMBV::DATA_TYPE VEC3;
	typedef typename PFP::REAL REAL;

	GMap3& map = dynamic_cast<GMap3&>(mapx);	// TODO reflechir comment virer ce warning quand on compile avec PFP::MAP=Map3

	if (m_attIndex.map() != &mapx)
		m_attIndex  = mapx.template getAttribute<unsigned int, DART>("dart_index3");
	if (!m_attIndex.isValid())
		m_attIndex  = mapx.template addAttribute<unsigned int, DART>("dart_index3");

	m_nbDarts = 0;
	for (Dart d = mapx.begin(); d != mapx.end(); mapx.next(d))
	{
		if (!map.isBoundaryMarked3(d)) // in the following code Traversor do not traverse boundary
			m_nbDarts++;
	}

	// compute center of each volumes
	VolumeAutoAttribute<VEC3> centerVolumes(mapx, "centerVolumes");
	Algo::Volume::Geometry::Parallel::computeCentroidELWVolumes<PFP>(mapx, positions, centerVolumes);

	// beta1
	DartAutoAttribute<VEC3> fv1(mapx);
	// beta2/3
	DartAutoAttribute<VEC3> fv2(mapx);
	DartAutoAttribute<VEC3> fv2x(mapx);

	m_vbo4->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(VEC3), 0, GL_STREAM_DRAW);
	GLvoid* ColorDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	VEC3* colorDartBuf = reinterpret_cast<VEC3*>(ColorDartsBuffer);


	if (m_bufferDartPosition!=NULL)
		delete m_bufferDartPosition;
	m_bufferDartPosition = new Geom::Vec3f[2*m_nbDarts];
	VEC3* positionDartBuf = reinterpret_cast<VEC3*>(m_bufferDartPosition);

//	m_vbo0->bind();
//	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(VEC3), 0, GL_STREAM_DRAW);
//	GLvoid* PositionDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
//	VEC3* positionDartBuf = reinterpret_cast<VEC3*>(PositionDartsBuffer);

	std::vector<Dart> vecDartFaces;
	vecDartFaces.reserve(m_nbDarts/6);
	unsigned int posDBI=0;

	//traverse each face of each volume
	TraversorCell<typename PFP::MAP, PFP::MAP::FACE_OF_PARENT> traFace(mapx);
	for (Dart d = traFace.begin(); d != traFace.end(); d = traFace.next())
	{
		vecDartFaces.push_back(d);

		std::vector<VEC3> vecPos;
		vecPos.reserve(16);

		// store the face & center
		float okv = 1.0f - kv;

		VEC3 vc = centerVolumes[d];
		
		
		VEC3 centerFace = Algo::Surface::Geometry::faceCentroidELW<PFP>(mapx,d,positions)*kv +vc*okv;

		//shrink the face
		float okf = 1.0f - kf;
		Dart dd = d;
		do
		{
			VEC3 P = centerFace*okf + (vc*okv + positions[dd]*kv)*kf;
			vecPos.push_back(P);
			dd = mapx.phi1(dd);
		} while (dd != d);
		
		unsigned int nb = vecPos.size();
		
		vecPos.push_back(vecPos.front()); // copy the first for easy computation on next loop

		// compute position of points to use for drawing topo
		float oke = 1.0f - ke;
		for (unsigned int i = 0; i < nb; ++i)
		{
			VEC3 P = vecPos[i]*ke + vecPos[i+1]*oke;
			VEC3 Q = vecPos[i+1]*ke + vecPos[i]*oke;

			VEC3 PP = 0.52f*P + 0.48f*Q;
			VEC3 QQ = 0.52f*Q + 0.48f*P;

			*positionDartBuf++ = P;
			*positionDartBuf++ = PP;
			*positionDartBuf++ = Q;
			*positionDartBuf++ = QQ;
			*colorDartBuf++ = m_dartsColor;
			*colorDartBuf++ = m_dartsColor;
			*colorDartBuf++ = m_dartsColor;
			*colorDartBuf++ = m_dartsColor;

			m_attIndex[d] = posDBI;
			posDBI+=2;


			fv1[d] = P*0.9f + PP*0.1f;
			fv2x[d] = P*0.52f + PP*0.48f;
			fv2[d] = P*0.48f + PP*0.52f;
			Dart dx = map.beta0(d);
			fv1[dx] = Q*0.9f + QQ*0.1f;
			fv2[dx] = Q*0.52f + QQ*0.48f;
			fv2x[dx] = Q*0.48f + QQ*0.52f;

			m_attIndex[dx] = posDBI;
			posDBI+=2;

			d = mapx.phi1(d);
		}
	}

	m_vbo0->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(VEC3), m_bufferDartPosition, GL_STREAM_DRAW);
//	m_vbo0->bind();
//	glUnmapBuffer(GL_ARRAY_BUFFER);

	m_vbo4->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	// beta3
	m_vbo1->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(typename PFP::VEC3), 0, GL_STREAM_DRAW);
	GLvoid* PositionBuffer1 = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);

	// beta3
	m_vbo2->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(typename PFP::VEC3), 0, GL_STREAM_DRAW);
	GLvoid* PositionBuffer2 = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);

	// beta3
	m_vbo3->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(typename PFP::VEC3), 0, GL_STREAM_DRAW);
	GLvoid* PositionBuffer3 = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);

	VEC3* positionF1 = reinterpret_cast<VEC3*>(PositionBuffer1);
	VEC3* positionF2 = reinterpret_cast<VEC3*>(PositionBuffer2);
	VEC3* positionF3 = reinterpret_cast<VEC3*>(PositionBuffer3);

	m_nbRel2=0;
	m_nbRel3=0;

	for(std::vector<Dart>::iterator face = vecDartFaces.begin(); face != vecDartFaces.end(); ++face)
	{
		Dart d = *face;
		do
		{
			Dart e = map.beta2(d);
			if (d < e)
			{
				*positionF2++ = fv2[d];
				*positionF2++ = fv2x[e];
				*positionF2++ = fv2[e];
				*positionF2++ = fv2x[d];
				m_nbRel2++;
			}
			e = map.beta3(d);
			if (!map.isBoundaryMarked3(e) && (d < e))
			{
				*positionF3++ = fv2[d];
				*positionF3++ = fv2x[e];
				*positionF3++ = fv2[e];
				*positionF3++ = fv2x[d];
				m_nbRel3++;
			}
			d = map.beta0(d);
			e = map.beta2(d);
			if (d < e)
			{
				*positionF2++ = fv2[d];
				*positionF2++ = fv2x[e];
				*positionF2++ = fv2[e];
				*positionF2++ = fv2x[d];
				m_nbRel2++;
			}
			e = map.beta3(d);
			if (!map.isBoundaryMarked3(e) && (d < e))
			{
				*positionF3++ = fv2[d];
				*positionF3++ = fv2x[e];
				*positionF3++ = fv2[e];
				*positionF3++ = fv2x[d];
				m_nbRel3++;
			}
			*positionF1++ = fv1[d];
			d = map.beta1(d);
			*positionF1++ = fv1[d];
			m_nbRel1++;
		} while (d != *face );
	}

	m_vbo3->bind();
	glUnmapBufferARB(GL_ARRAY_BUFFER);

	m_vbo2->bind();
	glUnmapBufferARB(GL_ARRAY_BUFFER);

	m_vbo1->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	m_vbo4->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

template<typename PFP>
void Topo3Render::computeDartMiddlePositions(typename PFP::MAP& map, DartAttribute<typename PFP::VEC3>& posExpl)
{
	m_vbo0->bind();
	typename PFP::VEC3* positionsPtr = reinterpret_cast<typename PFP::VEC3*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY));

	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		posExpl[d] = (positionsPtr[m_attIndex[d]] + positionsPtr[m_attIndex[d]+1])*0.5f;
	}

	m_vbo0->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

//template<typename PFP>
//void Topo3Render::updateDataMap3OldFashioned(typename PFP::MAP& mapx, const typename PFP::TVEC3& positions, float ke, float kf, float kv)
//{
//	Map3& map = reinterpret_cast<Map3&>(mapx);
//
//	typedef typename PFP::VEC3 VEC3;
//	typedef typename PFP::REAL REAL;
//
//
//	if (m_attIndex.map() != &map)
//	{
//		m_attIndex  = map.template getAttribute<unsigned int>(DART, "dart_index3");
//		if (!m_attIndex.isValid())
//			m_attIndex  = map.template addAttribute<unsigned int>(DART, "dart_index3");
//	}
//
//	m_nbDarts = 0;
//
//	// table of center of volume
//	std::vector<VEC3> vecCenters;
//	vecCenters.reserve(1000);
//	// table of nbfaces per volume
//	std::vector<unsigned int> vecNbFaces;
//	vecNbFaces.reserve(1000);
//	// table of face (one dart of each)
//	std::vector<Dart> vecDartFaces;
//	vecDartFaces.reserve(map.getNbDarts()/4);
//
//	unsigned int posDBI=0;
//
//	DartMarker mark(map);					// marker for darts
//	for (Dart d = map.begin(); d != map.end(); map.next(d))
//	{
//
//			CellMarkerStore markVert(map, VERTEX);		//marker for vertices
//			VEC3 center(0, 0, 0);
//			unsigned int nbv = 0;
//			unsigned int nbf = 0;
//			std::list<Dart> visitedFaces;	// Faces that are traversed
//			visitedFaces.push_back(d);		// Start with the face of d
//
//			// For every face added to the list
//			for (std::list<Dart>::iterator face = visitedFaces.begin(); face != visitedFaces.end(); ++face)
//			{
//				if (!mark.isMarked(*face))		// Face has not been visited yet
//				{
//					// store a dart of face
//					vecDartFaces.push_back(*face);
//					nbf++;
//					Dart dNext = *face ;
//					do
//					{
//						if (!markVert.isMarked(dNext))
//						{
//							markVert.mark(dNext);
//							center += positions[dNext];
//							nbv++;
//						}
//						mark.mark(dNext);					// Mark
//						m_nbDarts++;
//						Dart adj = map.phi2(dNext);				// Get adjacent face
//						if (adj != dNext && !mark.isMarked(adj))
//							visitedFaces.push_back(adj);	// Add it
//						dNext = map.phi1(dNext);
//					} while(dNext != *face);
//				}
//			}
//			center /= typename PFP::REAL(nbv);
//			vecCenters.push_back(center);
//			vecNbFaces.push_back(nbf);
//	}
//
//	// debut phi1
//	DartAutoAttribute<VEC3> fv1(map);
//	// fin phi1
//	DartAutoAttribute<VEC3> fv11(map);
//
//	// phi2
//	DartAutoAttribute<VEC3> fv2(map);
//	DartAutoAttribute<VEC3> fv2x(map);
//
//	m_vbo4->bind();
//	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(VEC3), 0, GL_STREAM_DRAW);
//	GLvoid* ColorDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
//	VEC3* colorDartBuf = reinterpret_cast<VEC3*>(ColorDartsBuffer);
//
//	m_vbo0->bind();
//	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(VEC3), 0, GL_STREAM_DRAW);
//	GLvoid* PositionDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
//	VEC3* positionDartBuf = reinterpret_cast<VEC3*>(PositionDartsBuffer);
//
//
//
//	std::vector<Dart>::iterator face = vecDartFaces.begin();
//	for (unsigned int iVol=0; iVol<vecNbFaces.size(); ++iVol)
//	{
//		for (unsigned int iFace = 0; iFace < vecNbFaces[iVol]; ++iFace)
//		{
//			Dart d = *face++;
//
//			std::vector<VEC3> vecPos;
//			vecPos.reserve(16);
//
//			// store the face & center
//			VEC3 center(0, 0, 0);
//			Dart dd = d;
//			do
//			{
//				const VEC3& P = positions[d];
//				vecPos.push_back(P);
//				center += P;
//				d = map.phi1(d);
//			} while (d != dd);
//			center /= REAL(vecPos.size());
//
//			//shrink the face
//			unsigned int nb = vecPos.size();
//			float okf = 1.0f - kf;
//			float okv = 1.0f - kv;
//			for (unsigned int i = 0; i < nb; ++i)
//			{
//				vecPos[i] = vecCenters[iVol]*okv + vecPos[i]*kv;
//				vecPos[i] = center*okf + vecPos[i]*kf;
//			}
//			vecPos.push_back(vecPos.front()); // copy the first for easy computation on next loop
//
//			// compute position of points to use for drawing topo
//			float oke = 1.0f - ke;
//			for (unsigned int i = 0; i < nb; ++i)
//			{
//				VEC3 P = vecPos[i]*ke + vecPos[i+1]*oke;
//				VEC3 Q = vecPos[i+1]*ke + vecPos[i]*oke;
//
//				m_attIndex[d] = posDBI;
//				posDBI+=2;
//
//				*positionDartBuf++ = P;
//				*positionDartBuf++ = Q;
//				*colorDartBuf++ = VEC3(1.,1.,1.0);
//				*colorDartBuf++ = VEC3(1.,1.,1.0);
//
//				fv1[d] = P*0.1f + Q*0.9f;
//				fv11[d] = P*0.9f + Q*0.1f;
//
//				fv2[d] = P*0.52f + Q*0.48f;
//				fv2x[d] = P*0.48f + Q*0.52f;
//
//				d = map.phi1(d);
//			}
//
//		}
//	}
//
//	m_vbo0->bind();
//	glUnmapBuffer(GL_ARRAY_BUFFER);
//
//	// phi1
//	m_vbo1->bind();
//	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(typename PFP::VEC3), 0, GL_STREAM_DRAW);
//	GLvoid* PositionBuffer1 = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);
//
//	//phi2
//	m_vbo2->bind();
//	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(typename PFP::VEC3), 0, GL_STREAM_DRAW);
//	GLvoid* PositionBuffer2 = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);
//
//	//phi3
//	m_vbo3->bind();
//	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(typename PFP::VEC3), 0, GL_STREAM_DRAW);
//	GLvoid* PositionBuffer3 = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);
//
//	VEC3* positionF1 = reinterpret_cast<VEC3*>(PositionBuffer1);
//	VEC3* positionF2 = reinterpret_cast<VEC3*>(PositionBuffer2);
//	VEC3* positionF3 = reinterpret_cast<VEC3*>(PositionBuffer3);
//
//	m_nbRel2=0;
//	m_nbRel3=0;
//
//	for(std::vector<Dart>::iterator face = vecDartFaces.begin(); face != vecDartFaces.end(); ++face)
//	{
//		Dart d = *face;
//		do
//		{
//			Dart e = map.phi2(d);
//			if (d < e)
//			{
//				*positionF2++ = fv2[d];
//				*positionF2++ = fv2x[e];
//				*positionF2++ = fv2[e];
//				*positionF2++ = fv2x[d];
//				m_nbRel2++;
//			}
//			e = map.phi3(d);
//			if (!map.isBoundaryMarked3(e) && (d < e))
//			{
//				*positionF3++ = fv2[d];
//				*positionF3++ = fv2x[e];
//				*positionF3++ = fv2[e];
//				*positionF3++ = fv2x[d];
//				m_nbRel3++;
//			}
//			e = map.phi1(d);
//			*positionF1++ = fv1[d];
//			*positionF1++ = fv11[e];
//
//			d = map.phi1(d);
//		} while (d != *face );
//	}
//
//	m_vbo3->bind();
//	glUnmapBufferARB(GL_ARRAY_BUFFER);
//
//	m_vbo2->bind();
//	glUnmapBufferARB(GL_ARRAY_BUFFER);
//
//	m_vbo1->bind();
//	glUnmapBuffer(GL_ARRAY_BUFFER);
//
//	m_vbo4->bind();
//	glUnmapBuffer(GL_ARRAY_BUFFER);
//}


template<typename PFP>
Dart Topo3Render::coneSelection(typename PFP::MAP& map, const Geom::Vec3f& rayA, const Geom::Vec3f& rayAB, float angle)
{
	float AB2 = rayAB*rayAB;
	Dart dFinal;
	double sin2 = sin(M_PI/180.0 * angle);
	sin2 = sin2*sin2;
	double dist2 = std::numeric_limits<double>::max();

	for(Dart d = map.begin(); d!=map.end(); map.next(d))
	{
		// get back position of segment PQ
		const Geom::Vec3f& P = m_bufferDartPosition[m_attIndex[d]];
		const Geom::Vec3f& Q =m_bufferDartPosition[m_attIndex[d]+1];
		float ld2 = Geom::squaredDistanceLine2Seg(rayA, rayAB, AB2, P, Q);
		Geom::Vec3f V = (P+Q)/2.0f - rayA;
		double d2 = double(V*V);
		double s2 = double(ld2) / d2;
		if (s2 < sin2)
		{
			if (d2<dist2)
			{
				dist2 = d2;
				dFinal = d;
			}
		}
	}
	return dFinal;
}

template<typename PFP>
Dart Topo3Render::raySelection(typename PFP::MAP& map, const Geom::Vec3f& rayA, const Geom::Vec3f& rayAB, float dmax)
{
	float AB2 = rayAB*rayAB;
	Dart dFinal;
	float dm2 = dmax*dmax;
	double dist2 = std::numeric_limits<double>::max();

	for(Dart d = map.begin(); d!=map.end(); map.next(d))
	{
		// get back position of segment PQ
		const Geom::Vec3f& P = m_bufferDartPosition[m_attIndex[d]];
		const Geom::Vec3f& Q =m_bufferDartPosition[m_attIndex[d]+1];
		float ld2 = Geom::squaredDistanceLine2Seg(rayA, rayAB, AB2, P, Q);
		if (ld2<dm2)
		{
			Geom::Vec3f V = (P+Q)/2.0f - rayA;
			double d2 = double(V*V);
			if (d2<dist2)
			{
				dist2 = d2;
				dFinal = d;
			}
		}
	}
	return dFinal;
}

// DART RAY SELECTION
//template<typename PFP>
//void edgesConeSelection(, const VertexAttribute<typename PFP::VEC3>& position, const typename PFP::VEC3& rayA, const typename PFP::VEC3& rayAB, float angle, std::vector<Dart>& vecEdges)
//{
//	typename PFP::REAL AB2 = rayAB * rayAB;
//
//	double sin2 = sin(M_PI/180.0 * angle);
//	sin2 = sin2*sin2;
//
//	// recuperation des aretes intersectees
//	vecEdges.reserve(256);
//	vecEdges.clear();
//
//	TraversorE<typename PFP::MAP> trav(map);
//	for(Dart d = trav.begin(); d!=trav.end(); d = trav.next())
//	{
//		// get back position of segment PQ
//		const typename PFP::VEC3& P = position[d];
//		const typename PFP::VEC3& Q = position[map.phi1(d)];
//		// the three distance to P, Q and (PQ) not used here
//		float ld2 = Geom::squaredDistanceLine2Seg(rayA, rayAB, AB2, P, Q);
//		typename PFP::VEC3 V = P - rayA;
//		double s2 = double(ld2) / double(V*V);
//		if (s2 < sin2)
//			vecEdges.push_back(d);
//	}
//
//	typedef std::pair<typename PFP::REAL, Dart> DartDist;
//	std::vector<DartDist> distndart;
//
//	unsigned int nbi = vecEdges.size();
//	distndart.resize(nbi);
//
//	// compute all distances to observer for each middle of intersected edge
//	// and put them in a vector for sorting
//	for (unsigned int i = 0; i < nbi; ++i)
//	{
//		Dart d = vecEdges[i];
//		distndart[i].second = d;
//		typename PFP::VEC3 V = (position[d] + position[map.phi1(d)]) / typename PFP::REAL(2);
//		V -= rayA;
//		distndart[i].first = V.norm2();
//	}
//
//	// sort the vector of pair dist/dart
//	std::sort(distndart.begin(), distndart.end(), distndartOrdering<PFP>);
//
//	// store sorted darts in returned vector
//	for (unsigned int i = 0; i < nbi; ++i)
//		vecEdges[i] = distndart[i].second;
//}







} //end namespace GL2

} //end namespace Render

} //end namespace Algo

} //end namespace CGoGN
