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
TopoRender<PFP>::TopoRender(float bs):
	m_nbDarts(0),
	m_nbRel2(0),
	m_topo_dart_width(2.0f),
	m_topo_relation_width(3.0f),
	m_normalShift(0.0f),
	m_boundShift(bs),
	m_dartsColor(1.0f,1.0f,1.0f),
	m_dartsBoundaryColor(0.7f,1.0f,0.7f),
	m_bufferDartPosition(NULL)
{
	m_vbo0 = new Utils::VBO();
	m_vbo1 = new Utils::VBO();
	m_vbo2 = new Utils::VBO();
	m_vbo3 = new Utils::VBO();

	m_vbo0->setDataSize(3);
	m_vbo1->setDataSize(3);
	m_vbo2->setDataSize(3);
	m_vbo3->setDataSize(3);

	m_shader1 = new Utils::ShaderSimpleColor();
	m_shader2 = new Utils::ShaderColorPerVertex();

	// binding VBO - VA
	m_vaId = m_shader1->setAttributePosition(m_vbo1);

	m_shader2->setAttributePosition(m_vbo0);
	m_shader2->setAttributeColor(m_vbo3);

	// registering for auto matrices update
	Utils::GLSLShader::registerShader(NULL, m_shader1);
	Utils::GLSLShader::registerShader(NULL, m_shader2);
}

template<typename PFP>
TopoRender<PFP>::~TopoRender()
{
	Utils::GLSLShader::unregisterShader(NULL, m_shader2);
	Utils::GLSLShader::unregisterShader(NULL, m_shader1);

	delete m_shader2;
	delete m_shader1;
	delete m_vbo3;
	delete m_vbo2;
	delete m_vbo1;
	delete m_vbo0;

	if (m_attIndex.isValid())
		m_attIndex.map()->removeAttribute(m_attIndex);

	if (m_bufferDartPosition!=NULL)
		delete[] m_bufferDartPosition;
}

template<typename PFP>
void TopoRender<PFP>::setDartWidth(float dw)
{
	m_topo_dart_width = dw;
}

template<typename PFP>
void TopoRender<PFP>::setRelationWidth(float pw)
{
	m_topo_relation_width = pw;
}

template<typename PFP>
void TopoRender<PFP>::setDartColor(Dart d, float r, float g, float b)
{
	float RGB[6];
	RGB[0]=r; RGB[1]=g; RGB[2]=b;
	RGB[3]=r; RGB[4]=g; RGB[5]=b;
	m_vbo3->bind();
	glBufferSubData(GL_ARRAY_BUFFER, m_attIndex[d]*3*sizeof(float), 6*sizeof(float),RGB);
}

template<typename PFP>
void TopoRender<PFP>::setAllDartsColor(float r, float g, float b)
{
	m_vbo3->bind();
	GLvoid* ColorDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	float* colorDartBuf = reinterpret_cast<float*>(ColorDartsBuffer);
	for (unsigned int i=0; i < 2*m_nbDarts; ++i)
	{
		*colorDartBuf++ = r;
		*colorDartBuf++ = g;
		*colorDartBuf++ = b;
	}

	m_vbo3->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

template<typename PFP>
void TopoRender<PFP>::setInitialDartsColor(float r, float g, float b)
{
	m_dartsColor = Geom::Vec3f(r,g,b);
}

template<typename PFP>
void TopoRender<PFP>::setInitialBoundaryDartsColor(float r, float g, float b)
{
	m_dartsBoundaryColor = Geom::Vec3f(r,g,b);
}

template<typename PFP>
void TopoRender<PFP>::drawDarts()
{
	if (m_nbDarts==0)
		return;

	m_shader2->enableVertexAttribs();

	glLineWidth(m_topo_dart_width);
	glDrawArrays(GL_LINES, 0, m_nbDarts*2);

	// change the stride to take 1/2 vertices
	m_shader2->enableVertexAttribs(6*sizeof(GL_FLOAT));

	glPointSize(2.0f*m_topo_dart_width);
	glDrawArrays(GL_POINTS, 0, m_nbDarts);

	m_shader2->disableVertexAttribs();
}

template<typename PFP>
void TopoRender<PFP>::drawRelation1()
{
	if (m_nbDarts==0)
		return;

	glLineWidth(m_topo_relation_width);

	m_shader1->changeVA_VBO(m_vaId, m_vbo1);
	m_shader1->setColor(Geom::Vec4f(0.0f,1.0f,1.0f,0.0f));
	m_shader1->enableVertexAttribs();

	glDrawArrays(GL_LINES, 0, m_nbDarts*2);

	m_shader1->disableVertexAttribs();
}

template<typename PFP>
void TopoRender<PFP>::drawRelation2()
{
	if (m_nbRel2==0)
		return;

	glLineWidth(m_topo_relation_width);

	m_shader1->changeVA_VBO(m_vaId, m_vbo2);
	m_shader1->setColor(Geom::Vec4f(1.0f,0.0f,0.0f,0.0f));
	m_shader1->enableVertexAttribs();

	glDrawArrays(GL_LINES, 0, m_nbRel2*2);

	m_shader1->disableVertexAttribs();
}

template<typename PFP>
void TopoRender<PFP>::drawTopo()
{
	drawDarts();
	drawRelation1();
	drawRelation2();
}

template<typename PFP>
void TopoRender<PFP>::overdrawDart(Dart d, float width, float r, float g, float b)
{
	unsigned int indexDart = m_attIndex[d];

	m_shader1->changeVA_VBO(m_vaId, m_vbo0);
	m_shader1->setColor(Geom::Vec4f(r,g,b,0.0f));
	m_shader1->enableVertexAttribs();

	glLineWidth(width);
	glDrawArrays(GL_LINES, indexDart, 2);

	glPointSize(2.0f*width);
	glDrawArrays(GL_POINTS, indexDart, 1);

	m_shader2->disableVertexAttribs();
}

template<typename PFP>
Dart TopoRender<PFP>::colToDart(float* color)
{
	unsigned int r = (unsigned int)(color[0]*255.0f);
	unsigned int g = (unsigned int)(color[1]*255.0f);
	unsigned int b = (unsigned int)(color[2]*255.0f);

	unsigned int id = r + 255*g +255*255*b;

	if (id == 0)
		return NIL;
	return Dart(id-1);
}

template<typename PFP>
void TopoRender<PFP>::dartToCol(Dart d, float& r, float& g, float& b)
{
	// here use d.index beacause it is what we want (and not map.dartIndex(d) !!)
	unsigned int lab = d.index + 1; // add one to avoid picking the black of screen

	r = float(lab%255) / 255.0f; lab = lab/255;
	g = float(lab%255) / 255.0f; lab = lab/255;
	b = float(lab%255) / 255.0f; lab = lab/255;
	if (lab!=0)
		CGoGNerr << "Error picking color, too many darts"<< CGoGNendl;
}

template<typename PFP>
Dart TopoRender<PFP>::pickColor(unsigned int x, unsigned int y)
{
	//more easy picking for
	unsigned int dw = m_topo_dart_width;
	m_topo_dart_width+=2;

	// save clear color and set to zero
	float cc[4];
	glGetFloatv(GL_COLOR_CLEAR_VALUE,cc);

	glClearColor(0.0f,0.0f,0.0f,0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDisable(GL_LIGHTING);
	// draw in back buffer (not shown on screen)
	drawDarts();

	// restore dart width
	m_topo_dart_width = dw;

	// read the pixel under the mouse in back buffer
	glReadBuffer(GL_BACK);
	float color[3];
	glReadPixels(x,y,1,1,GL_RGB,GL_FLOAT,color);

	glClearColor(cc[0], cc[1], cc[2], cc[3]);

	return colToDart(color);
}

template<typename PFP>
void TopoRender<PFP>::pushColors()
{
	m_color_save = new float[6*m_nbDarts];
	m_vbo3->bind();
	void* colorBuffer = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);

	memcpy(m_color_save, colorBuffer, 6*m_nbDarts*sizeof(float));
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

template<typename PFP>
void TopoRender<PFP>::popColors()
{
	m_vbo3->bind();
	void* colorBuffer = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);

	memcpy(colorBuffer, m_color_save, 6*m_nbDarts*sizeof(float));
	glUnmapBuffer(GL_ARRAY_BUFFER);

	delete[] m_color_save;
	m_color_save=NULL;
}

template<typename PFP>
void TopoRender<PFP>::svgout2D(const std::string& filename, const glm::mat4& model, const glm::mat4& proj)
{
	Utils::SVG::SVGOut svg(filename,model,proj);
	toSVG(svg);
	svg.write();
}

template<typename PFP>
void TopoRender<PFP>::toSVG(Utils::SVG::SVGOut& svg)
{
//	svg.setWidth(m_topo_relation_width);
//
//	// PHI2 / beta2
//	const Geom::Vec3f* ptr = reinterpret_cast<Geom::Vec3f*>(m_vbo2->lockPtr());
//	svg.beginLines();
//	for (unsigned int i=0; i<m_nbRel2; ++i)
//		svg.addLine(ptr[2*i], ptr[2*i+1],Geom::Vec3f(0.8f,0.0f,0.0f));
//	svg.endLines();
//
//	m_vbo2->releasePtr();
//
//	//PHI1 /beta1
//	ptr = reinterpret_cast<Geom::Vec3f*>(m_vbo1->lockPtr());
//	svg.beginLines();
//	for (unsigned int i=0; i<m_nbRel1; ++i)
//		svg.addLine(ptr[2*i], ptr[2*i+1],Geom::Vec3f(0.0f,0.7f,0.7f));
//	svg.endLines();
//	m_vbo1->releasePtr();
//
//
//	const Geom::Vec3f* colorsPtr = reinterpret_cast<const Geom::Vec3f*>(m_vbo3->lockPtr());
//	ptr= reinterpret_cast<Geom::Vec3f*>(m_vbo0->lockPtr());
//
//	svg.setWidth(m_topo_dart_width);
//	svg.beginLines();
//	for (unsigned int i=0; i<m_nbDarts; ++i)
//		svg.addLine(ptr[2*i], ptr[2*i+1], colorsPtr[2*i]);
//	svg.endLines();
//
//	svg.beginPoints();
//	for (unsigned int i=0; i<m_nbDarts; ++i)
//			svg.addPoint(ptr[2*i], colorsPtr[2*i]);
//	svg.endPoints();
//
//	m_vbo0->releasePtr();
//	m_vbo3->releasePtr();

	// PHI2 / beta2
	Utils::SVG::SvgGroup* svg1 = new Utils::SVG::SvgGroup("phi2", svg.m_model, svg.m_proj);
	svg1->setToLayer();
	const Geom::Vec3f* ptr = reinterpret_cast<Geom::Vec3f*>(m_vbo2->lockPtr());
	svg1->setWidth(m_topo_relation_width);
	svg1->beginLines();
	for (unsigned int i=0; i<m_nbRel2; ++i)
		svg1->addLine(ptr[2*i], ptr[2*i+1],Geom::Vec3f(0.8f,0.0f,0.0f));
	svg1->endLines();
	m_vbo2->releasePtr();

	svg.addGroup(svg1);

	//PHI1 /beta1
	Utils::SVG::SvgGroup* svg2 = new Utils::SVG::SvgGroup("phi1", svg.m_model, svg.m_proj);
	svg2->setToLayer();
	ptr = reinterpret_cast<Geom::Vec3f*>(m_vbo1->lockPtr());
	svg2->setWidth(m_topo_relation_width);
	svg2->beginLines();
	for (unsigned int i=0; i<m_nbRel1; ++i)
		svg2->addLine(ptr[2*i], ptr[2*i+1],Geom::Vec3f(0.0f,0.7f,0.7f));
	svg2->endLines();
	m_vbo1->releasePtr();

	svg.addGroup(svg2);

	const Geom::Vec3f* colorsPtr = reinterpret_cast<const Geom::Vec3f*>(m_vbo3->lockPtr());
	ptr= reinterpret_cast<Geom::Vec3f*>(m_vbo0->lockPtr());

	Utils::SVG::SvgGroup* svg3 = new Utils::SVG::SvgGroup("darts", svg.m_model, svg.m_proj);
	svg3->setToLayer();
	svg3->setWidth(m_topo_dart_width);
	svg3->beginLines();
	for (unsigned int i=0; i<m_nbDarts; ++i)
		svg3->addLine(ptr[2*i], ptr[2*i+1], colorsPtr[2*i]);
	svg3->endLines();

	svg.addGroup(svg3);

	Utils::SVG::SvgGroup* svg4 = new Utils::SVG::SvgGroup("dartEmb", svg.m_model, svg.m_proj);
	svg4->setWidth(m_topo_dart_width);
	svg4->setToLayer();
	svg4->beginPoints();
	for (unsigned int i=0; i<m_nbDarts; ++i)
			svg4->addPoint(ptr[2*i], colorsPtr[2*i]);
	svg4->endPoints();

	svg.addGroup(svg4);

	m_vbo0->releasePtr();
	m_vbo3->releasePtr();
}

template<typename PFP>
void TopoRender<PFP>::setNormalShift(float ns)
{
	m_normalShift = ns;
}

template<typename PFP>
void TopoRender<PFP>::setBoundaryShift(float bs)
{
	m_boundShift = bs;
}

template<typename PFP>
void TopoRender<PFP>::updateDataBoundary(MAP& map, const VertexAttribute<VEC3, MAP>& positions, float ke, float kf,float ns)
{
	m_normalShift = ns;
	SelectorDartBoundary<MAP> sdb(map);
	DartContainerBrowserSelector<MAP> browser(map, sdb);
	browser.enable();
	updateData(map, positions, ke, kf, true);
	browser.disable();
	m_normalShift = 0.0f;
}

//template<typename PFP>
//void TopoRender<PFP>::updateData(MAP& map, const VertexAttribute<VEC3, MAP>& positions, float ke, float kf, bool withBoundary)
//{
//	std::string typeName = map.mapTypeName();
//	if (typeName[0] == 'M') // "Map2"
//	{
//		updateDataMap(map, positions, ke, kf, withBoundary);
//		return;
//	}
//	if (typeName[0] == 'G') // "GMap2"
//	{
//		updateDataGMap(map, positions, ke, kf, withBoundary);
//		return;
//	}
//}

template<typename PFP>
void TopoRenderMap<PFP>::updateData(MAP& mapx, const VertexAttribute<VEC3, MAP>& positions, float ke, float kf, bool withBoundary)
{
	//Map2& map = reinterpret_cast<Map2&>(mapx);

	std::vector<Dart> vecDarts;
	vecDarts.reserve(mapx.getNbDarts());  // no problem dart is int: no problem of memory

	this->m_attIndex = mapx.template getAttribute<unsigned int, DART, MAP>("dart_index2");

	if (!this->m_attIndex.isValid())
		this->m_attIndex  = mapx.template addAttribute<unsigned int, DART, MAP>("dart_index2");

	for(Dart d = mapx.begin(); d != mapx.end(); mapx.next(d))
	{
		if (withBoundary || !mapx.template isBoundaryMarked<2>(d))
			vecDarts.push_back(d);
	}
	this->m_nbDarts = vecDarts.size();

	// debut phi1
	DartAutoAttribute<VEC3, MAP> fv1(mapx);
	// fin phi1
	DartAutoAttribute<VEC3, MAP> fv11(mapx);
	// phi2
	DartAutoAttribute<VEC3, MAP> fv2(mapx);

	this->m_vbo3->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*this->m_nbDarts*sizeof(Geom::Vec3f), 0, GL_STREAM_DRAW);
	GLvoid* ColorDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	Geom::Vec3f* colorDartBuf = reinterpret_cast<Geom::Vec3f*>(ColorDartsBuffer);

//	m_vbo0->bind();
//	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(VEC3), 0, GL_STREAM_DRAW);
//	GLvoid* PositionDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
//	VEC3* positionDartBuf = reinterpret_cast<VEC3*>(PositionDartsBuffer);

	if (this->m_bufferDartPosition!=NULL)
		delete this->m_bufferDartPosition;
	this->m_bufferDartPosition = new Geom::Vec3f[2*this->m_nbDarts];
	Geom::Vec3f* positionDartBuf = reinterpret_cast<Geom::Vec3f*>(this->m_bufferDartPosition);

	std::vector<VEC3> vecPos;
	vecPos.reserve(16);

	unsigned int indexDC = 0;

	DartMarker<MAP> mf(mapx);
	for(std::vector<Dart>::iterator id = vecDarts.begin(); id != vecDarts.end(); id++)
	{
		Dart d = *id;
		if (!mf.isMarked(d))
		{
			vecPos.clear();
			if (!mapx.template isBoundaryMarked<2>(d))
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

				if (this->m_normalShift > 0.0f)
				{
					VEC3 normal = Algo::Surface::Geometry::newellNormal<PFP>(mapx,d,positions);
					for (typename std::vector<VEC3>::iterator pit = vecPos.begin(); pit != vecPos.end(); ++pit)
					{
						*pit -= normal * this->m_normalShift;
					}
				}

				unsigned int nb = vecPos.size();
				vecPos.push_back(vecPos.front()); // copy the first for easy computation on next loop

				k = 1.0f - ke;
				for (unsigned int i = 0; i < nb; ++i)
				{

					VEC3 P = vecPos[i]*ke + vecPos[i+1]*k;
					VEC3 Q = vecPos[i+1]*ke + vecPos[i]*k;

					this->m_attIndex[d] = indexDC;
					indexDC+=2;
					*positionDartBuf++ = PFP::toVec3f(P);
					*positionDartBuf++ = PFP::toVec3f(Q);
					*colorDartBuf++ = this->m_dartsColor;
					*colorDartBuf++ = this->m_dartsColor;
					VEC3 f = P*0.5f + Q*0.5f;
					fv2[d] = f;
					f = P*0.1f + Q*0.9f;
					fv1[d] = f;
					f = P*0.9f + Q*0.1f;
					fv11[d] = f;
					d = mapx.phi1(d);
				}
				mf.template markOrbit<FACE>(d);
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
					VEC3 P = positions[mapx.phi1(ee)] + v* this->m_boundShift;
					vecPos.push_back(P);
					dd = mapx.phi1(dd);
					ee = mapx.phi2(dd);
					P = positions[mapx.phi1(ee)] + v* this->m_boundShift;
					vecPos.push_back(P);
				} while (dd != d);

				unsigned int nb = vecPos.size()/2;
				float k = 1.0f - ke;
				for (unsigned int i = 0; i < nb; ++i)
				{

					VEC3 P = vecPos[2*i]*ke + vecPos[2*i+1]*k;
					VEC3 Q = vecPos[2*i+1]*ke + vecPos[2*i]*k;

					this->m_attIndex[d] = indexDC;
					indexDC+=2;
					*positionDartBuf++ = PFP::toVec3f(P);
					*positionDartBuf++ = PFP::toVec3f(Q);
					*colorDartBuf++ = this->m_dartsBoundaryColor;
					*colorDartBuf++ = this->m_dartsBoundaryColor;
					VEC3 f = P*0.5f + Q*0.5f;
					fv2[d] = f;
					f = P*0.1f + Q*0.9f;
					fv1[d] = f;
					f = P*0.9f + Q*0.1f;
					fv11[d] = f;
					d = mapx.phi1(d);
				}
				mf.template markOrbit<FACE>(d);
			}
		}
	}

	this->m_vbo0->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*this->m_nbDarts*sizeof(Geom::Vec3f), this->m_bufferDartPosition, GL_STREAM_DRAW);
//	glUnmapBuffer(GL_ARRAY_BUFFER);

	this->m_vbo3->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	this->m_vbo1->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*this->m_nbDarts*sizeof(Geom::Vec3f), 0, GL_STREAM_DRAW);
	GLvoid* PositionBuffer1 = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);

	this->m_vbo2->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*this->m_nbDarts*sizeof(Geom::Vec3f), 0, GL_STREAM_DRAW);
	GLvoid* PositionBuffer2 = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);

	Geom::Vec3f* positionF1 = reinterpret_cast<Geom::Vec3f*>(PositionBuffer1);
	Geom::Vec3f* positionF2 = reinterpret_cast<Geom::Vec3f*>(PositionBuffer2);

	this->m_nbRel2 = 0;
	for(std::vector<Dart>::iterator id = vecDarts.begin(); id!= vecDarts.end(); id++)
	{
		Dart d = *id;

		Dart e = mapx.phi2(d);

//		if (good(e) && (e.index > d.index))
		if ( (withBoundary || !mapx.template isBoundaryMarked<2>(e)) && (e.index > d.index))
		{
			*positionF2++ = PFP::toVec3f(fv2[d]);
			*positionF2++ = PFP::toVec3f(fv2[e]);
			this->m_nbRel2++;
		}

		e = mapx.phi1(d);
		*positionF1++ = PFP::toVec3f(fv1[d]);
		*positionF1++ = PFP::toVec3f(fv11[e]);
	}
	this->m_nbRel1 = vecDarts.size();

	this->m_vbo1->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	this->m_vbo2->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);
}


template<typename PFP>
void TopoRenderGMap<PFP>::updateData(MAP& map, const VertexAttribute<VEC3, MAP>& positions, float ke, float kf, bool withBoundary)
{
//	GMap2& map = dynamic_cast<GMap2&>(mapx);

	std::vector<Dart> vecDarts;
	vecDarts.reserve(map.getNbDarts()); // no problem dart is int: no problem of memory

	if (this->m_attIndex.map() != &map)
		this->m_attIndex  = map.template getAttribute<unsigned int, DART, MAP>("dart_index2");

	if (!this->m_attIndex.isValid())
		this->m_attIndex  = map.template addAttribute<unsigned int, DART, MAP>("dart_index2");

	for(Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if (withBoundary || !map.template isBoundaryMarked<2>(d))
			vecDarts.push_back(d);
	}
	this->m_nbDarts = vecDarts.size();

	// debut phi1
	DartAutoAttribute<VEC3, MAP> fv1(map);
	// fin phi1
	DartAutoAttribute<VEC3, MAP> fv11(map);
	// phi2
	DartAutoAttribute<VEC3, MAP> fv2(map);

	this->m_vbo3->bind();
	glBufferData(GL_ARRAY_BUFFER, 4*this->m_nbDarts*sizeof(Geom::Vec3f), 0, GL_STREAM_DRAW);
	GLvoid* ColorDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	Geom::Vec3f* colorDartBuf = reinterpret_cast<Geom::Vec3f*>(ColorDartsBuffer);

	this->m_vbo0->bind();
	glBufferData(GL_ARRAY_BUFFER, 4*this->m_nbDarts*sizeof(Geom::Vec3f), 0, GL_STREAM_DRAW);
	GLvoid* PositionDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	Geom::Vec3f* positionDartBuf = reinterpret_cast<Geom::Vec3f*>(PositionDartsBuffer);

	std::vector<VEC3> vecPos;
	vecPos.reserve(16);

	unsigned int indexDC = 0;

	DartMarker<MAP> mf(map);
	for(std::vector<Dart>::iterator id = vecDarts.begin(); id!= vecDarts.end(); id++)
	{
		Dart d = *id;

		if (!mf.isMarked(d))
		{
			vecPos.clear();
			VEC3 center = Algo::Surface::Geometry::faceCentroidELW<PFP>(map, d, positions);
			
			float k = 1.0f - kf;
			Dart dd = d;
			do
			{
				vecPos.push_back(center*k + positions[dd]*kf);
				dd = map.phi1(dd);
			} while (dd != d);


			if (this->m_normalShift > 0.0f)
			{
				VEC3 normal = Algo::Surface::Geometry::newellNormal<PFP>(map, d, positions);
				for (typename std::vector<VEC3>::iterator pit = vecPos.begin(); pit != vecPos.end(); ++pit)
				{
					*pit -= normal * this->m_normalShift;
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

				this->m_attIndex[d] = indexDC;
				indexDC+=2;
				*positionDartBuf++ = PFP::toVec3f(P);
				*colorDartBuf++ = this->m_dartsColor;
				*positionDartBuf++ = PFP::toVec3f(PP);
				*colorDartBuf++ = this->m_dartsColor;
				*positionDartBuf++ = PFP::toVec3f(Q);
				*colorDartBuf++ = this->m_dartsColor;
				*positionDartBuf++ = PFP::toVec3f(QQ);
				*colorDartBuf++ = this->m_dartsColor;

				VEC3 f = P*0.5f + PP*0.5f;
				fv2[d] = f;
				f = P*0.9f + PP*0.1f;
				fv1[d] = f;

				dd = map.beta0(d);
				f = Q*0.5f + QQ*0.5f;
				fv2[dd] = f;
				f = Q*0.9f + QQ*0.1f;
				fv1[dd] = f;
				this->m_attIndex[dd] = indexDC;
				indexDC+=2;

				d = map.phi1(d);
			}
			mf.template markOrbit<FACE>(d);
		}
	}

	this->m_vbo0->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	this->m_vbo3->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	this->m_vbo1->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*this->m_nbDarts*sizeof(Geom::Vec3f), 0, GL_STREAM_DRAW);
	GLvoid* PositionBuffer1 = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);

	this->m_vbo2->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*this->m_nbDarts*sizeof(Geom::Vec3f), 0, GL_STREAM_DRAW);
	GLvoid* PositionBuffer2 = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);

	Geom::Vec3f* positionF1 = reinterpret_cast<Geom::Vec3f*>(PositionBuffer1);
	Geom::Vec3f* positionF2 = reinterpret_cast<Geom::Vec3f*>(PositionBuffer2);

	this->m_nbRel2 = 0;
	for(std::vector<Dart>::iterator id = vecDarts.begin(); id!= vecDarts.end(); id++)
	{
		Dart d = *id;
		Dart e = map.beta2(d);
//		if (d < e )
		if ( (withBoundary || !map.template isBoundaryMarked<2>(e)) && (d < e ))
		{
			*positionF2++ = PFP::toVec3f(fv2[d]);
			*positionF2++ = PFP::toVec3f(fv2[e]);
			this->m_nbRel2++;
		}

		e = map.beta1(d);
		*positionF1++ = PFP::toVec3f(fv1[d]);
		*positionF1++ = PFP::toVec3f(fv1[e]);
	}
	this->m_nbRel1 = vecDarts.size()/2;

	this->m_vbo1->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	this->m_vbo2->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

template<typename PFP>
void TopoRender<PFP>::setDartsIdColor(MAP& map, bool withBoundary)
{
	m_vbo3->bind();
	float* colorBuffer = reinterpret_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
	unsigned int nb = 0;

	m_attIndex = map.template getAttribute<unsigned int, DART, MAP>("dart_index2");
	if (!m_attIndex.isValid())
	{
		CGoGNerr << "Error attribute_dartIndex does not exist during TopoRender<PFP>::picking" << CGoGNendl;
		return;
	}

	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if (withBoundary || !map.template isBoundaryMarked<2>(d))
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
Dart TopoRender<PFP>::picking(MAP& map,int x, int y, bool withBoundary)
{
	pushColors();
	setDartsIdColor(map,withBoundary);
	Dart d = pickColor(x,y);
	popColors();
	return d;
}

template<typename PFP>
Dart TopoRender<PFP>::coneSelection(MAP& map, const Geom::Vec3f& rayA, const Geom::Vec3f& rayAB, float angle)
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
Dart TopoRender<PFP>::raySelection(MAP& map, const Geom::Vec3f& rayA, const Geom::Vec3f& rayAB, float dmax)
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

} // namespace GL2

} // namespace Algo

} // namespace Render

} // namespace CGoGN
