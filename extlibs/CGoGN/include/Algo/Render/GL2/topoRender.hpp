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
#include "Topology/generic/parameters.h"

#include "Topology/map/embeddedMap2.h"
#include "Topology/gmap/embeddedGMap2.h"

#include "Algo/Geometry/basic.h"
#include "Geometry/distances.h"
#include "Algo/Geometry/centroid.h"
#include "Algo/Geometry/normal.h"

#include "Container/containerBrowser.h"

namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace GL2
{


template<typename PFP>
void TopoRender::updateDataBoundary(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& positions, float ke, float kf,float ns)
{
	m_normalShift = ns;
	SelectorDartBoundary<typename PFP::MAP> sdb(map);
	DartContainerBrowserSelector browser(map,sdb);
	browser.enable();
	updateData<PFP>(map,positions, ke, kf,true);
	browser.disable();
	m_normalShift = 0.0f;
}


template<typename PFP>
void TopoRender::updateData(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& positions, float ke, float kf, bool withBoundary)
{
	Map2* ptrMap2 = dynamic_cast<Map2*>(&map);
	if (ptrMap2 != NULL)
	{
		updateDataMap<PFP>(map, positions, ke, kf, withBoundary);
		return;
	}
	GMap2* ptrGMap2 = dynamic_cast<GMap2*>(&map);
	if (ptrGMap2 != NULL)
	{
		updateDataGMap<PFP>(map, positions, ke, kf, withBoundary);
		return;
	}
}

template<typename PFP>
void TopoRender::updateDataMap(typename PFP::MAP& mapx, const VertexAttribute<typename PFP::VEC3>& positions, float ke, float kf, bool withBoundary)
{
	//Map2& map = reinterpret_cast<Map2&>(mapx);

	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

	std::vector<Dart> vecDarts;
	vecDarts.reserve(mapx.getNbDarts());  // no problem dart is int: no problem of memory

	m_attIndex = mapx.template getAttribute<unsigned int, DART>("dart_index2");

	if (!m_attIndex.isValid())
		m_attIndex  = mapx.template addAttribute<unsigned int, DART>("dart_index2");

	for(Dart d = mapx.begin(); d!= mapx.end(); mapx.next(d))
	{
		if (withBoundary || !mapx.isBoundaryMarked2(d))
			vecDarts.push_back(d);

	}
	m_nbDarts = vecDarts.size();

	// debut phi1
	DartAutoAttribute<VEC3> fv1(mapx);
	// fin phi1
	DartAutoAttribute<VEC3> fv11(mapx);
	// phi2
	DartAutoAttribute<VEC3> fv2(mapx);

	m_vbo3->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(VEC3), 0, GL_STREAM_DRAW);
	GLvoid* ColorDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	VEC3* colorDartBuf = reinterpret_cast<VEC3*>(ColorDartsBuffer);

//	m_vbo0->bind();
//	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(VEC3), 0, GL_STREAM_DRAW);
//	GLvoid* PositionDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
//	VEC3* positionDartBuf = reinterpret_cast<VEC3*>(PositionDartsBuffer);

	if (m_bufferDartPosition!=NULL)
		delete m_bufferDartPosition;
	m_bufferDartPosition = new Geom::Vec3f[2*m_nbDarts];
	VEC3* positionDartBuf = reinterpret_cast<VEC3*>(m_bufferDartPosition);

	std::vector<VEC3> vecPos;
	vecPos.reserve(16);

	unsigned int indexDC=0;

	DartMarker mf(mapx);
	for(std::vector<Dart>::iterator id = vecDarts.begin(); id!= vecDarts.end(); id++)
	{
		Dart d = *id;
		if (!mf.isMarked(d))
		{
			vecPos.clear();
			if (!mapx.isBoundaryMarked2(d))
			{
				//VEC3 center = Algo::Surface::Geometry::faceCentroidELW<PFP>(mapx,d,positions);
				VEC3 center = Algo::Surface::Geometry::faceCentroid<PFP>(mapx,d,positions);
				float k = 1.0f - kf;
				Dart dd = d;
				do
				{
					vecPos.push_back(center*k + positions[dd]*kf);
					dd = mapx.phi1(dd);
				} while (dd != d);


				if (m_normalShift > 0.0f)
				{
					VEC3 normal = Algo::Surface::Geometry::newellNormal<PFP>(mapx,d,positions);
					for (typename std::vector<VEC3>::iterator pit = vecPos.begin(); pit != vecPos.end(); ++pit)
					{
						*pit -= normal*m_normalShift;
					}
				}

				unsigned int nb = vecPos.size();
				vecPos.push_back(vecPos.front()); // copy the first for easy computation on next loop

				k = 1.0f - ke;
				for (unsigned int i = 0; i < nb; ++i)
				{

					VEC3 P = vecPos[i]*ke + vecPos[i+1]*k;
					VEC3 Q = vecPos[i+1]*ke + vecPos[i]*k;

					m_attIndex[d] = indexDC;
					indexDC+=2;
					*positionDartBuf++ = P;
					*positionDartBuf++ = Q;
					*colorDartBuf++ = m_dartsColor;
					*colorDartBuf++ = m_dartsColor;
					VEC3 f = P*0.5f + Q*0.5f;
					fv2[d] = f;
					f = P*0.1f + Q*0.9f;
					fv1[d] = f;
					f = P*0.9f + Q*0.1f;
					fv11[d] = f;
					d = mapx.phi1(d);
				}
				mf.markOrbit<FACE>(d);
			}
			else if (withBoundary)
			{

				Dart dd = d;
				do
				{
					Dart ee = mapx.phi2(dd);
					VEC3 normal = Algo::Surface::Geometry::newellNormal<PFP>(mapx,ee,positions);
					VEC3 vd = Algo::Surface::Geometry::vectorOutOfDart<PFP>(mapx,ee,positions);
					VEC3 v = vd ^ normal;
					v.normalize();
					VEC3 P = positions[mapx.phi1(ee)] + v* m_boundShift;
					vecPos.push_back(P);
					dd = mapx.phi1(dd);
					ee = mapx.phi2(dd);
					P = positions[mapx.phi1(ee)] + v* m_boundShift;
					vecPos.push_back(P);
				} while (dd != d);

				unsigned int nb = vecPos.size()/2;
				float k = 1.0f - ke;
				for (unsigned int i = 0; i < nb; ++i)
				{

					VEC3 P = vecPos[2*i]*ke + vecPos[2*i+1]*k;
					VEC3 Q = vecPos[2*i+1]*ke + vecPos[2*i]*k;

					m_attIndex[d] = indexDC;
					indexDC+=2;
					*positionDartBuf++ = P;
					*positionDartBuf++ = Q;
					*colorDartBuf++ = m_dartsBoundaryColor;
					*colorDartBuf++ = m_dartsBoundaryColor;
					VEC3 f = P*0.5f + Q*0.5f;
					fv2[d] = f;
					f = P*0.1f + Q*0.9f;
					fv1[d] = f;
					f = P*0.9f + Q*0.1f;
					fv11[d] = f;
					d = mapx.phi1(d);
				}
				mf.markOrbit<FACE>(d);
			}
		}
	}

	m_vbo0->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(VEC3), m_bufferDartPosition, GL_STREAM_DRAW);
//	glUnmapBuffer(GL_ARRAY_BUFFER);

	m_vbo3->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	m_vbo1->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(typename PFP::VEC3), 0, GL_STREAM_DRAW);
	GLvoid* PositionBuffer1 = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);

	m_vbo2->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(typename PFP::VEC3), 0, GL_STREAM_DRAW);
	GLvoid* PositionBuffer2 = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);

	VEC3* positionF1 = reinterpret_cast<VEC3*>(PositionBuffer1);
	VEC3* positionF2 = reinterpret_cast<VEC3*>(PositionBuffer2);

	m_nbRel2 =0;
	for(std::vector<Dart>::iterator id = vecDarts.begin(); id!= vecDarts.end(); id++)
	{
		Dart d = *id;

		Dart e = mapx.phi2(d);

//		if (good(e) && (e.index > d.index))
		if ( (withBoundary || !mapx.isBoundaryMarked2(e)) && (e.index > d.index))
		{
			*positionF2++ = fv2[d];
			*positionF2++ = fv2[e];
			m_nbRel2++;
		}

		e = mapx.phi1(d);
		*positionF1++ = fv1[d];
		*positionF1++ = fv11[e];
	}
	m_nbRel1 = vecDarts.size();

	m_vbo1->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	m_vbo2->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

template<typename PFP>
void TopoRender::updateDataGMap(typename PFP::MAP& mapx, const VertexAttribute<typename PFP::VEC3>& positions, float ke, float kf, bool withBoundary)
{
	GMap2& map = dynamic_cast<GMap2&>(mapx);

	typedef typename PFP::VEC3 VEC3;
	typedef typename PFP::REAL REAL;

	std::vector<Dart> vecDarts;
	vecDarts.reserve(map.getNbDarts()); // no problem dart is int: no problem of memory

	if (m_attIndex.map() != &map)
		m_attIndex  = map.template getAttribute<unsigned int, DART>("dart_index2");

	if (!m_attIndex.isValid())
		m_attIndex  = map.template addAttribute<unsigned int, DART>("dart_index2");


	for(Dart d = map.begin(); d!= map.end(); map.next(d))
	{
		if (withBoundary || !map.isBoundaryMarked2(d))
			vecDarts.push_back(d);
	}
	m_nbDarts = vecDarts.size();

	// debut phi1
	DartAutoAttribute<VEC3> fv1(map);
	// fin phi1
	DartAutoAttribute<VEC3> fv11(map);
	// phi2
	DartAutoAttribute<VEC3> fv2(map);

	m_vbo3->bind();
	glBufferData(GL_ARRAY_BUFFER, 4*m_nbDarts*sizeof(VEC3), 0, GL_STREAM_DRAW);
	GLvoid* ColorDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	VEC3* colorDartBuf = reinterpret_cast<VEC3*>(ColorDartsBuffer);

	m_vbo0->bind();
	glBufferData(GL_ARRAY_BUFFER, 4*m_nbDarts*sizeof(VEC3), 0, GL_STREAM_DRAW);
	GLvoid* PositionDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	VEC3* positionDartBuf = reinterpret_cast<VEC3*>(PositionDartsBuffer);

	std::vector<VEC3> vecPos;
	vecPos.reserve(16);

	unsigned int indexDC=0;

	DartMarker mf(map);
	for(std::vector<Dart>::iterator id = vecDarts.begin(); id!= vecDarts.end(); id++)
	{
		Dart d = *id;

		if (!mf.isMarked(d))
		{
			vecPos.clear();
			VEC3 center = Algo::Surface::Geometry::faceCentroidELW<PFP>(mapx,d,positions);
			
			float k = 1.0f - kf;
			Dart dd = d;
			do
			{
				vecPos.push_back(center*k + positions[dd]*kf);
				dd = map.phi1(dd);
			} while (dd != d);


			if (m_normalShift > 0.0f)
			{
				VEC3 normal = Algo::Surface::Geometry::newellNormal<PFP>(mapx,d,positions);
				for (typename std::vector<VEC3>::iterator pit = vecPos.begin(); pit != vecPos.end(); ++pit)
				{
					*pit -= normal*m_normalShift;
				}
			}
			unsigned int nb = vecPos.size();
			vecPos.push_back(vecPos.front()); // copy the first for easy computation on next loop


			k = 1.0f - ke;
			for (unsigned int i = 0; i < nb; ++i)
			{
				VEC3 P = vecPos[i]*ke + vecPos[i+1]*k;
				VEC3 Q = vecPos[i+1]*ke + vecPos[i]*k;
				VEC3 PP = REAL(0.52)*P + REAL(0.48)*Q;
				VEC3 QQ = REAL(0.52)*Q + REAL(0.48)*P;

				m_attIndex[d] = indexDC;
				indexDC+=2;
				*positionDartBuf++ = P;
				*colorDartBuf++ = m_dartsColor;
				*positionDartBuf++ = PP;
				*colorDartBuf++ = m_dartsColor;
				*positionDartBuf++ = Q;
				*colorDartBuf++ = m_dartsColor;
				*positionDartBuf++ = QQ;
				*colorDartBuf++ = m_dartsColor;

				VEC3 f = P*0.5f + PP*0.5f;
				fv2[d] = f;
				f = P*0.9f + PP*0.1f;
				fv1[d] = f;

				dd = map.beta0(d);
				f = Q*0.5f + QQ*0.5f;
				fv2[dd] = f;
				f = Q*0.9f + QQ*0.1f;
				fv1[dd] = f;
				m_attIndex[dd] = indexDC;
				indexDC+=2;


				d = map.phi1(d);
			}
			mf.markOrbit<FACE>(d);
		}
	}

	m_vbo0->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	m_vbo3->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	m_vbo1->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(typename PFP::VEC3), 0, GL_STREAM_DRAW);
	GLvoid* PositionBuffer1 = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);

	m_vbo2->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(typename PFP::VEC3), 0, GL_STREAM_DRAW);
	GLvoid* PositionBuffer2 = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);

	VEC3* positionF1 = reinterpret_cast<VEC3*>(PositionBuffer1);
	VEC3* positionF2 = reinterpret_cast<VEC3*>(PositionBuffer2);

	m_nbRel2 = 0;
	for(std::vector<Dart>::iterator id = vecDarts.begin(); id!= vecDarts.end(); id++)
	{
		Dart d = *id;
		Dart e = map.beta2(d);
//		if (d < e )
		if ( (withBoundary || !map.isBoundaryMarked2(e)) && (d < e ))
		{
			*positionF2++ = fv2[d];
			*positionF2++ = fv2[e];
			m_nbRel2++;
		}

		e = map.beta1(d);
		*positionF1++ = fv1[d];
		*positionF1++ = fv1[e];
	}
	m_nbRel1 = vecDarts.size()/2;

	m_vbo1->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	m_vbo2->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

template<typename PFP>
void TopoRender::setDartsIdColor(typename PFP::MAP& map, bool withBoundary)
{
	m_vbo3->bind();
	float* colorBuffer = reinterpret_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
	unsigned int nb = 0;

	m_attIndex = map.template getAttribute<unsigned int, DART>("dart_index2");
	if (!m_attIndex.isValid())
	{
		CGoGNerr << "Error attribute_dartIndex does not exist during TopoRender::picking" << CGoGNendl;
		return;
	}

	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if (withBoundary || !map.isBoundaryMarked2(d))
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
				CGoGNerr << "Error buffer too small for color picking (change the good parameter ?)" << CGoGNendl;
				CGoGNerr << "NB = " << nb << "   NBDARTs = "<< m_nbDarts<<CGoGNendl;
				break;
			}
		}
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

template<typename PFP>
Dart TopoRender::picking(typename PFP::MAP& map,int x, int y, bool withBoundary)
{
	pushColors();
	setDartsIdColor<PFP>(map,withBoundary);
	Dart d = pickColor(x,y);
	popColors();
	return d;

}



template<typename PFP>
Dart TopoRender::coneSelection(typename PFP::MAP& map, const Geom::Vec3f& rayA, const Geom::Vec3f& rayAB, float angle)
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
Dart TopoRender::raySelection(typename PFP::MAP& map, const Geom::Vec3f& rayA, const Geom::Vec3f& rayAB, float dmax)
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


}//end namespace GL2

}//end namespace Algo

}//end namespace Render

}//end namespace CGoGN
