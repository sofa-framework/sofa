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

template<typename PFP>
void Topo3PrimalRender::setDartsIdColor(typename PFP::MAP& map)
{
	m_vbo2->bind();
	float* colorBuffer =  reinterpret_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
	unsigned int nb=0;

	for (Dart d = map.begin(); d != map.end(); map.next(d))
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
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

template<typename PFP>
void Topo3PrimalRender::updateColors(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& colors)
{
	m_vbo2->bind();
	Geom::Vec3f* colorBuffer =  reinterpret_cast<Geom::Vec3f*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
	unsigned int nb=0;

	for (Dart d = map.begin(); d != map.end(); map.next(d))
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
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

template<typename PFP>
Dart Topo3PrimalRender::picking(typename PFP::MAP& map, int x, int y)
{
	pushColors();
	setDartsIdColor<PFP>(map);
	Dart d = pickColor(x,y);
	popColors();
	return d;
}

//template<typename PFP>
//void Topo3PrimalRender::updateData(typename PFP::MAP& mapx, const VertexAttribute<typename PFP::VEC3>& positions, float ke, float kf)
//{
//    updateDataGen<PFP, VertexAttribute<typename PFP::VEC3>, typename PFP::VEC3>(mapx,positions,ke,kf);
//}

template<typename PFP, typename EMBV>
void Topo3PrimalRender::updateData(typename PFP::MAP& mapx, const EMBV& positions, float ke, float kf)
{
	typedef typename EMBV::DATA_TYPE VEC3;
	typedef typename PFP::REAL REAL;

	Map3& map = dynamic_cast<Map3&>(mapx);	// TODO reflechir comment virer ce warning quand on compile avec PFP::MAP=Map3


	if (m_attIndex.map() != &mapx)
		m_attIndex  = mapx.template getAttribute<unsigned int, DART>("dart_index");
	if (!m_attIndex.isValid())
		m_attIndex  = mapx.template addAttribute<unsigned int, DART>("dart_index");

//	m_nbDarts = 0;
//	for (Dart d = mapx.begin(); d != mapx.end(); mapx.next(d))
//	{
//		m_nbDarts++;
//	}

	m_nbDarts = mapx.getNbDarts();

	// beta2/3
	DartAutoAttribute<VEC3> fv2(mapx);

	m_vbo2->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(VEC3), 0, GL_STREAM_DRAW);
	GLvoid* ColorDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	VEC3* colorDartBuf = reinterpret_cast<VEC3*>(ColorDartsBuffer);


	if (m_bufferDartPosition!=NULL)
		delete m_bufferDartPosition;
	m_bufferDartPosition = new Geom::Vec3f[2*m_nbDarts];
	VEC3* positionDartBuf = reinterpret_cast<VEC3*>(m_bufferDartPosition);


	unsigned int posDBI=0;

	int nbf=0;
	//traverse each face of each volume
	TraversorF<typename PFP::MAP> traFace(mapx);
	for (Dart d = traFace.begin(); d != traFace.end(); d = traFace.next())
	{
		std::vector<VEC3> vecPos;
		vecPos.reserve(16);

		VEC3 centerFace = Algo::Surface::Geometry::faceCentroidELW<PFP>(mapx,d,positions);

		//shrink the face
		float okf = 1.0f - kf;
		Dart dd = d;
		do
		{
			VEC3 P = centerFace*okf + positions[dd]*kf;
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

//			VEC3 PP = 0.52f*P + 0.48f*Q;
//			VEC3 QQ = 0.52f*Q + 0.48f*P;

            VEC3 PP = 0.56f*P + 0.44f*Q;
            VEC3 QQ = 0.56f*Q + 0.44f*P;

			*positionDartBuf++ = P;
			*positionDartBuf++ = PP;
			if (map.isBoundaryMarked3(d))
			{
				*colorDartBuf++ = m_boundaryDartsColor;
				*colorDartBuf++ = m_boundaryDartsColor;
			}
			else
			{
				*colorDartBuf++ = m_dartsColor;
				*colorDartBuf++ = m_dartsColor;
			}

			m_attIndex[d] = posDBI;
			posDBI+=2;
			fv2[d] = (P+PP)*0.5f;


			*positionDartBuf++ = Q;
			*positionDartBuf++ = QQ;

			Dart dx = map.phi3(d);
			if (map.isBoundaryMarked3(dx))
			{
				*colorDartBuf++ = m_boundaryDartsColor;
				*colorDartBuf++ = m_boundaryDartsColor;
			}
			else
			{
				*colorDartBuf++ = m_dartsColor;
				*colorDartBuf++ = m_dartsColor;
			}

			m_attIndex[dx] = posDBI;
			posDBI+=2;
			fv2[dx] = (Q+QQ)*0.5f;

			d = mapx.phi1(d);
		}
		nbf++;
	}
	m_vbo2->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	m_vbo0->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(VEC3), m_bufferDartPosition, GL_STREAM_DRAW);

	// alpha2
	m_vbo1->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(typename PFP::VEC3), 0, GL_STREAM_DRAW);
	GLvoid* PositionBuffer2 = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);

	VEC3* positionF2 = reinterpret_cast<VEC3*>(PositionBuffer2);

	m_nbRel2=0;

	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		Dart e = map.phi2(map.phi3(d));
		//if (d < e)
		{

			*positionF2++ = fv2[d];
			*positionF2++ = fv2[e];
			m_nbRel2++;
		}
	}

	m_vbo1->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

template<typename PFP>
void Topo3PrimalRender::computeDartMiddlePositions(typename PFP::MAP& map, DartAttribute<typename PFP::VEC3>& posExpl)
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



template<typename PFP>
Dart Topo3PrimalRender::coneSelection(typename PFP::MAP& map, const Geom::Vec3f& rayA, const Geom::Vec3f& rayAB, float angle)
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
Dart Topo3PrimalRender::raySelection(typename PFP::MAP& map, const Geom::Vec3f& rayA, const Geom::Vec3f& rayAB, float dmax)
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


} //end namespace GL2

} //end namespace Render

} //end namespace Algo

} //end namespace CGoGN
