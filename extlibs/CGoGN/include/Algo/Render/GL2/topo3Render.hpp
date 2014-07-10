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

#include "Topology/generic/traversor/traversorCell.h"
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
Topo3Render<PFP>::Topo3Render():
	m_nbDarts(0),
	m_nbRel1(0),
	m_nbRel2(0),
	m_nbRel3(0),
	m_topo_dart_width(2.0f),
	m_topo_relation_width(3.0f),
	m_color_save(NULL),
	m_dartsColor(1.0f,1.0f,1.0f),
	m_bufferDartPosition(NULL)
{
	m_vbo0 = new Utils::VBO();
	m_vbo1 = new Utils::VBO();
	m_vbo2 = new Utils::VBO();
	m_vbo3 = new Utils::VBO();
	m_vbo4 = new Utils::VBO();

	m_vbo0->setDataSize(3);
	m_vbo1->setDataSize(3);
	m_vbo2->setDataSize(3);
	m_vbo3->setDataSize(3);
	m_vbo4->setDataSize(3);

	m_shader1 = new Utils::ShaderSimpleColor();
	m_shader2 = new Utils::ShaderColorPerVertex();

	// binding VBO - VA
	m_vaId = m_shader1->setAttributePosition(m_vbo1);

	m_shader2->setAttributePosition(m_vbo0);
	m_shader2->setAttributeColor(m_vbo4);

	// registering for auto matrices update
	Utils::GLSLShader::registerShader(NULL, m_shader1);
	Utils::GLSLShader::registerShader(NULL, m_shader2);
}

template<typename PFP>
Topo3Render<PFP>::~Topo3Render()
{
	Utils::GLSLShader::unregisterShader(NULL, m_shader2);
	Utils::GLSLShader::unregisterShader(NULL, m_shader1);

	delete m_shader2;
	delete m_shader1;
	delete m_vbo4;
	delete m_vbo3;
	delete m_vbo2;
	delete m_vbo1;
	delete m_vbo0;

	if (m_attIndex.map() != NULL)
		m_attIndex.map()->removeAttribute(m_attIndex);

	if (m_color_save != NULL)
		delete[] m_color_save;

	if (m_bufferDartPosition != NULL)
		delete[] m_bufferDartPosition;
}

template<typename PFP>
void Topo3Render<PFP>::setDartWidth(float dw)
{
	m_topo_dart_width = dw;
}

template<typename PFP>
void Topo3Render<PFP>::setRelationWidth(float pw)
{
	m_topo_relation_width = pw;
}

template<typename PFP>
void Topo3Render<PFP>::setDartColor(Dart d, float r, float g, float b)
{
	float RGB[6];
	RGB[0]=r; RGB[1]=g; RGB[2]=b;
	RGB[3]=r; RGB[4]=g; RGB[5]=b;
	m_vbo4->bind();
	glBufferSubData(GL_ARRAY_BUFFER, m_attIndex[d]*3*sizeof(float), 6*sizeof(float),RGB);
}

template<typename PFP>
void Topo3Render<PFP>::setAllDartsColor(float r, float g, float b)
{
	m_vbo4->bind();
	GLvoid* ColorDartsBuffer = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);
	float* colorDartBuf = reinterpret_cast<float*>(ColorDartsBuffer);
	for (unsigned int i=0; i < 2*m_nbDarts; ++i)
	{
		*colorDartBuf++ = r;
		*colorDartBuf++ = g;
		*colorDartBuf++ = b;
	}
	glUnmapBufferARB(GL_ARRAY_BUFFER);

}

template<typename PFP>
void Topo3Render<PFP>::setInitialDartsColor(float r, float g, float b)
{
	m_dartsColor = Geom::Vec3f(r,g,b);
}

template<typename PFP>
void Topo3Render<PFP>::drawDarts()
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

//
////	glColor3f(1.0f,1.0f,1.0f);
//	glLineWidth(m_topo_dart_width);
//	glPointSize(2.0f*m_topo_dart_width);
//
//	glBindBufferARB(GL_ARRAY_BUFFER, m_VBOBuffers[4]);
//	glColorPointer(3, GL_FLOAT, 0, 0);
//	glEnableClientState(GL_COLOR_ARRAY);
//
//	glBindBufferARB(GL_ARRAY_BUFFER, m_VBOBuffers[0]);
//	glVertexPointer(3, GL_FLOAT, 0, 0);
//	glEnableClientState(GL_VERTEX_ARRAY);
//	glDrawArrays(GL_LINES, 0, m_nbDarts*2);
//
// 	glVertexPointer(3, GL_FLOAT, 6*sizeof(GL_FLOAT), 0);
//
//	glBindBufferARB(GL_ARRAY_BUFFER, m_VBOBuffers[4]);
// 	glColorPointer(3, GL_FLOAT, 6*sizeof(GL_FLOAT), 0);
// 	glDrawArrays(GL_POINTS, 0, m_nbDarts)
// 	;
//	glDisableClientState(GL_COLOR_ARRAY);
//	glDisableClientState(GL_VERTEX_ARRAY);

}

template<typename PFP>
void Topo3Render<PFP>::drawRelation1()
{
	if (m_nbDarts==0)
		return;

	glLineWidth(m_topo_relation_width);

	m_shader1->changeVA_VBO(m_vaId, m_vbo1);
	m_shader1->setColor(Geom::Vec4f(0.0f,1.0f,1.0f,0.0f));
	m_shader1->enableVertexAttribs();

	glDrawArrays(GL_LINES, 0, m_nbRel1*2);

	m_shader1->disableVertexAttribs();

//	glLineWidth(m_topo_relation_width);
//	glColor3f(0.0f,1.0f,1.0f);
//	glBindBufferARB(GL_ARRAY_BUFFER, m_VBOBuffers[1]);
//	glVertexPointer(3, GL_FLOAT, 0, 0);
//
//	glEnableClientState(GL_VERTEX_ARRAY);
//	glDrawArrays(GL_LINES, 0, m_nbDarts*2);
//	glDisableClientState(GL_VERTEX_ARRAY);
}

template<typename PFP>
void Topo3Render<PFP>::drawRelation2()
{
	if (m_nbRel2==0)
		return;

	m_shader1->changeVA_VBO(m_vaId, m_vbo2);
	m_shader1->setColor(Geom::Vec4f(1.0f,0.0f,0.0f,0.0f));
	m_shader1->enableVertexAttribs();

	glDrawArrays(GL_QUADS, 0, m_nbRel2*4);

	m_shader1->disableVertexAttribs();

//	glLineWidth(m_topo_relation_width);
//	glColor3f(1.0f,0.0f,0.0f);
//	glBindBufferARB(GL_ARRAY_BUFFER, m_VBOBuffers[2]);
//	glVertexPointer(3, GL_FLOAT, 0, 0);
//
//	glEnableClientState(GL_VERTEX_ARRAY);
//	glDrawArrays(GL_QUADS, 0, m_nbRel2*4);
//	glDisableClientState(GL_VERTEX_ARRAY);
}

template<typename PFP>
void Topo3Render<PFP>::drawRelation3(Geom::Vec4f c)
{
	if (m_nbRel3==0)
		return;

	m_shader1->changeVA_VBO(m_vaId, m_vbo3);
	m_shader1->setColor(c);
	m_shader1->enableVertexAttribs();

	glDrawArrays(GL_QUADS, 0, m_nbRel3*4);

	m_shader1->disableVertexAttribs();

//	glLineWidth(m_topo_relation_width);
//	glColor3f(1.0f,1.0f,0.0f);
//	glBindBufferARB(GL_ARRAY_BUFFER, m_VBOBuffers[3]);
//	glVertexPointer(3, GL_FLOAT, 0, 0);
//
//	glEnableClientState(GL_VERTEX_ARRAY);
//	glDrawArrays(GL_QUADS, 0, m_nbRel3*4);
//	glDisableClientState(GL_VERTEX_ARRAY);
}

template<typename PFP>
void Topo3Render<PFP>::drawTopo()
{
	drawDarts();
	drawRelation1();
	drawRelation2();
	drawRelation3(Geom::Vec4f(1.0f,1.0f,0.0f,0.0f));
}

template<typename PFP>
void Topo3Render<PFP>::overdrawDart(Dart d, float width, float r, float g, float b)
{
	unsigned int indexDart =  m_attIndex[d];

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
void Topo3Render<PFP>::pushColors()
{
	m_color_save = new float[6*m_nbDarts];
	m_vbo4->bind();
	void* colorBuffer = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);

	memcpy(m_color_save, colorBuffer, 6*m_nbDarts*sizeof(float));
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

template<typename PFP>
void Topo3Render<PFP>::popColors()
{
	m_vbo4->bind();
	void* colorBuffer = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);

	memcpy(colorBuffer, m_color_save, 6*m_nbDarts*sizeof(float));
	glUnmapBuffer(GL_ARRAY_BUFFER);

	delete[] m_color_save;
	m_color_save=0;
}

template<typename PFP>
Dart Topo3Render<PFP>::colToDart(float* color)
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
void Topo3Render<PFP>::dartToCol(Dart d, float& r, float& g, float& b)
{
	// here use dart.index beacause it is what we want (and not map.dartIndex(d) !!)
	unsigned int lab = d.index + 1; // add one to avoid picking the black of screen

	r = float(lab%255) / 255.0f; lab = lab/255;
	g = float(lab%255) / 255.0f; lab = lab/255;
	b = float(lab%255) / 255.0f; lab = lab/255;
	if (lab!=0)
		CGoGNerr << "Error picking color, too many darts"<< CGoGNendl;
}

template<typename PFP>
Dart Topo3Render<PFP>::pickColor(unsigned int x, unsigned int y)
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

	// restore dart with
	m_topo_dart_width = dw;

	// read the pixel under the mouse in back buffer
	glReadBuffer(GL_BACK);
	float color[3];
	glReadPixels(x,y,1,1,GL_RGB,GL_FLOAT,color);

	glClearColor(cc[0], cc[1], cc[2], cc[3]);


	std::cout << color[0] << ", "<<color[1] << ", "<<color[2] <<std::endl;
	return colToDart(color);
}

template<typename PFP>
void Topo3Render<PFP>::svgout2D(const std::string& filename, const glm::mat4& model, const glm::mat4& proj)
{
	Utils::SVG::SVGOut svg(filename,model,proj);
	toSVG(svg);
	svg.write();
}

template<typename PFP>
void Topo3Render<PFP>::toSVG(Utils::SVG::SVGOut& svg)
{
	// PHI3 / beta3
	Utils::SVG::SvgGroup* svg1 = new Utils::SVG::SvgGroup("phi3", svg.m_model, svg.m_proj);
	const Geom::Vec3f* ptr = reinterpret_cast<Geom::Vec3f*>(m_vbo3->lockPtr());
	svg1->setWidth(m_topo_relation_width);
	svg1->beginLines();
	for (unsigned int i=0; i<m_nbRel3; ++i)
	{
		Geom::Vec3f P = (ptr[4*i]+ ptr[4*i+3])/2.0f;
		Geom::Vec3f Q = (ptr[4*i+1]+ ptr[4*i+2])/2.0f;
		svg1->addLine(P, Q,Geom::Vec3f(0.8f,0.8f,0.0f));
	}
	svg1->endLines();
	m_vbo3->releasePtr();

	svg.addGroup(svg1);

	// PHI2 / beta2
	Utils::SVG::SvgGroup* svg2 = new Utils::SVG::SvgGroup("phi2", svg.m_model, svg.m_proj);
	ptr = reinterpret_cast<Geom::Vec3f*>(m_vbo2->lockPtr());
	svg2->setWidth(m_topo_relation_width);
	svg2->beginLines();
	for (unsigned int i=0; i<m_nbRel2; ++i)
	{
		Geom::Vec3f P = (ptr[4*i]+ ptr[4*i+3])/2.0f;
		Geom::Vec3f Q = (ptr[4*i+1]+ ptr[4*i+2])/2.0f;
		svg2->addLine(P, Q,Geom::Vec3f(0.8f,0.0f,0.0f));
	}
	svg2->endLines();
	m_vbo2->releasePtr();

	svg.addGroup(svg2);

	//PHI1 /beta1
	Utils::SVG::SvgGroup* svg3 = new Utils::SVG::SvgGroup("phi1", svg.m_model, svg.m_proj);
	ptr = reinterpret_cast<Geom::Vec3f*>(m_vbo1->lockPtr());
	svg3->setWidth(m_topo_relation_width);
	svg3->beginLines();
	for (unsigned int i=0; i<m_nbRel1; ++i)
		svg3->addLine(ptr[2*i], ptr[2*i+1],Geom::Vec3f(0.0f,0.7f,0.7f));
	svg3->endLines();
	m_vbo1->releasePtr();

	svg.addGroup(svg3);

	const Geom::Vec3f* colorsPtr = reinterpret_cast<const Geom::Vec3f*>(m_vbo4->lockPtr());
	ptr= reinterpret_cast<Geom::Vec3f*>(m_vbo0->lockPtr());

	Utils::SVG::SvgGroup* svg4 = new Utils::SVG::SvgGroup("darts", svg.m_model, svg.m_proj);
	svg4->setWidth(m_topo_dart_width);

	svg4->beginLines();
	for (unsigned int i=0; i<m_nbDarts; ++i)
	{
		Geom::Vec3f col = colorsPtr[2*i];
		if (col.norm2()>2.9f)
			col = Geom::Vec3f(1.0f,1.0f,1.0f) - col;
		svg4->addLine(ptr[2*i], ptr[2*i+1], col);
	}
	svg4->endLines();

	svg.addGroup(svg4);

	Utils::SVG::SvgGroup* svg5 = new Utils::SVG::SvgGroup("dartEmb", svg.m_model, svg.m_proj);
	svg5->setWidth(m_topo_dart_width);
	svg5->beginPoints();
	for (unsigned int i=0; i<m_nbDarts; ++i)
	{
		Geom::Vec3f col = colorsPtr[2*i];
		if (col.norm2()>2.9f)
			col = Geom::Vec3f(1.0f,1.0f,1.0f) - col;
		svg5->addPoint(ptr[2*i], col);
	}
	svg5->endPoints();

	svg.addGroup(svg5);

	m_vbo0->releasePtr();
	m_vbo4->releasePtr();
}

//template<typename PFP>
//void Topo3Render<PFP>::updateData(typename PFP::MAP& map, const VertexAttribute<typename PFP::VEC3>& positions, float ke, float kf, float kv)
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

//template<typename PFP>
//void Topo3Render<PFP>::updateData(MAP& map, const VertexAttribute<VEC3, MAP>& positions, float ke, float kf, float kv)
//{
//	std::string typeName = map.mapTypeName();
//	if (typeName[0] == 'M') // "Map3"
//	{
//		updateDataMap3(map, positions, ke, kf, kv);
//		return;
//	}
//	if (typeName[0] == 'G') // "GMap3"
//	{
//		updateDataGMap3(map, positions, ke, kf, kv);
//		return;
//	}
//}

template<typename PFP>
void Topo3Render<PFP>::setDartsIdColor(typename PFP::MAP& map)
{
	m_vbo4->bind();
	float* colorBuffer =  reinterpret_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
	unsigned int nb=0;

	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		if ( !map.template isBoundaryMarked<3>(d)) // topo3 Render do not traverse boundary
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

template<typename PFP>
void Topo3Render<PFP>::updateColors(MAP& map, const VertexAttribute<Geom::Vec3f, MAP>& colors)
{
	m_vbo4->bind();
	VEC3* colorBuffer = reinterpret_cast<VEC3*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
	unsigned int nb = 0;

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
Dart Topo3Render<PFP>::picking(MAP& map, int x, int y)
{
	pushColors();
	setDartsIdColor(map);
	Dart d = pickColor(x,y);
	popColors();
	return d;
}






template<typename PFP>
void Topo3RenderMap<PFP>::updateData(MAP& mapx, const VertexAttribute<VEC3, MAP>& positions, float ke, float kf, float kv)
{
    this->m_attIndex = mapx.template getAttribute<unsigned int, DART, MAP>("dart_index3");

    if (!this->m_attIndex.isValid())
        this->m_attIndex  = mapx.template addAttribute<unsigned int, DART, MAP>("dart_index3");

    this->m_nbDarts = 0;
    for (Dart d = mapx.begin(); d != mapx.end(); mapx.next(d))
    {
        if (!mapx.template isBoundaryMarked<3>(d)) // in the following code Traversor do not traverse boundary
            this->m_nbDarts++;
    }

    // compute center of each volumes
    CellMarker<MAP, VOLUME> cmv(mapx);
    VolumeAutoAttribute<VEC3, MAP> centerVolumes(mapx, "centerVolumes");

    Algo::Volume::Geometry::Parallel::computeCentroidELWVolumes<PFP>(mapx, positions, centerVolumes);

    // debut phi1
    DartAutoAttribute<VEC3, MAP> fv1(mapx);
    // fin phi1
    DartAutoAttribute<VEC3, MAP> fv11(mapx);

    // phi2
    DartAutoAttribute<VEC3, MAP> fv2(mapx);
    DartAutoAttribute<VEC3, MAP> fv2x(mapx);

    this->m_vbo4->bind();
    glBufferData(GL_ARRAY_BUFFER, 2*this->m_nbDarts*sizeof(VEC3), 0, GL_STREAM_DRAW);
    GLvoid* ColorDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
    Geom::Vec3f* colorDartBuf = reinterpret_cast<Geom::Vec3f*>(ColorDartsBuffer);

    this->m_vbo0->bind();
    glBufferData(GL_ARRAY_BUFFER, 2*this->m_nbDarts*sizeof(VEC3), 0, GL_STREAM_DRAW);
    GLvoid* PositionDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
    Geom::Vec3f* positionDartBuf = reinterpret_cast<Geom::Vec3f*>(PositionDartsBuffer);

    std::vector<Dart> vecDartFaces;
    vecDartFaces.reserve(this->m_nbDarts/3);
    unsigned int posDBI = 0;

    // traverse each face of each volume
    TraversorCell<MAP, PFP::MAP::FACE_OF_PARENT> traFace(mapx);
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

            this->m_attIndex[d] = posDBI;
            posDBI+=2;

            *positionDartBuf++ = PFP::toVec3f(P);
            *positionDartBuf++ = PFP::toVec3f(Q);
            *colorDartBuf++ = this->m_dartsColor;
            *colorDartBuf++ = this->m_dartsColor;

            fv1[d] = P*0.1f + Q*0.9f;
            fv11[d] = P*0.9f + Q*0.1f;

            fv2[d] = P*0.52f + Q*0.48f;
            fv2x[d] = P*0.48f + Q*0.52f;
            d = mapx.phi1(d);
        }
    }

    this->m_vbo0->bind();
    glUnmapBuffer(GL_ARRAY_BUFFER);

    this->m_vbo4->bind();
    glUnmapBuffer(GL_ARRAY_BUFFER);

    Geom::Vec3f* positioniF1 = new Geom::Vec3f[ 2*this->m_nbDarts];
    Geom::Vec3f* positioniF2 = new Geom::Vec3f[ 2*this->m_nbDarts];
    Geom::Vec3f* positioniF3 = new Geom::Vec3f[ 2*this->m_nbDarts];

    Geom::Vec3f* positionF1 = positioniF1;
    Geom::Vec3f* positionF2 = positioniF2;
    Geom::Vec3f* positionF3 = positioniF3;

    this->m_nbRel1 = 0;
    this->m_nbRel2 = 0;
    this->m_nbRel3 = 0;

    for(std::vector<Dart>::iterator face = vecDartFaces.begin(); face != vecDartFaces.end(); ++face)
    {
        Dart d = *face;
        do
        {
            Dart e = mapx.phi2(d);
            if ((d < e))
            {
                *positionF2++ = PFP::toVec3f(fv2[d]);
                *positionF2++ = PFP::toVec3f(fv2x[e]);
                *positionF2++ = PFP::toVec3f(fv2[e]);
                *positionF2++ = PFP::toVec3f(fv2x[d]);
                this->m_nbRel2++;
            }
            e = mapx.phi3(d);
            if (!mapx.template isBoundaryMarked<3>(e) && (d < e) )
            {
                *positionF3++ = PFP::toVec3f(fv2[d]);
                *positionF3++ = PFP::toVec3f(fv2x[e]);
                *positionF3++ = PFP::toVec3f(fv2[e]);
                *positionF3++ = PFP::toVec3f(fv2x[d]);
                this->m_nbRel3++;
            }
            e = mapx.phi1(d);
            *positionF1++ = PFP::toVec3f(fv1[d]);
            *positionF1++ = PFP::toVec3f(fv11[e]);
            this->m_nbRel1++;

            d = mapx.phi1(d);
        } while (d != *face );
    }

    this->m_vbo3->bind();
    glBufferData(GL_ARRAY_BUFFER, 4*this->m_nbRel3*sizeof(Geom::Vec3f), positioniF3, GL_STREAM_DRAW);

    this->m_vbo2->bind();
    glBufferData(GL_ARRAY_BUFFER, 4*this->m_nbRel2*sizeof(Geom::Vec3f), positioniF2, GL_STREAM_DRAW);

    this->m_vbo1->bind();
    glBufferData(GL_ARRAY_BUFFER, 2*this->m_nbRel1*sizeof(Geom::Vec3f), positioniF1, GL_STREAM_DRAW);

    delete[] positioniF1;
    delete[] positioniF2;
    delete[] positioniF3;
}






template<typename PFP>
void Topo3RenderGMap<PFP>::updateData(MAP& mapx, const VertexAttribute<VEC3, MAP>& positions, float ke, float kf, float kv)
{
//	GMap3& map = dynamic_cast<GMap3&>(mapx);	// TODO reflechir comment virer ce warning quand on compile avec PFP::MAP=Map3

	if (this->m_attIndex.map() != &mapx)
		this->m_attIndex = mapx.template getAttribute<unsigned int, DART, MAP>("dart_index3");
	if (!this->m_attIndex.isValid())
		this->m_attIndex = mapx.template addAttribute<unsigned int, DART, MAP>("dart_index3");

	this->m_nbDarts = 0;
	for (Dart d = mapx.begin(); d != mapx.end(); mapx.next(d))
	{
		if (!mapx.template isBoundaryMarked<3>(d)) // in the following code Traversor do not traverse boundary
			this->m_nbDarts++;
	}

	// compute center of each volumes
	VolumeAutoAttribute<VEC3, MAP> centerVolumes(mapx, "centerVolumes");
	Algo::Volume::Geometry::Parallel::computeCentroidELWVolumes<PFP>(mapx, positions, centerVolumes);

	// beta1
	DartAutoAttribute<VEC3, MAP> fv1(mapx);
	// beta2/3
	DartAutoAttribute<VEC3, MAP> fv2(mapx);
	DartAutoAttribute<VEC3, MAP> fv2x(mapx);

	this->m_vbo4->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*this->m_nbDarts*sizeof(Geom::Vec3f), 0, GL_STREAM_DRAW);
	GLvoid* ColorDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	Geom::Vec3f* colorDartBuf = reinterpret_cast<Geom::Vec3f*>(ColorDartsBuffer);

	if (this->m_bufferDartPosition != NULL)
		delete this->m_bufferDartPosition;
	this->m_bufferDartPosition = new Geom::Vec3f[2*this->m_nbDarts];
	Geom::Vec3f* positionDartBuf = reinterpret_cast<Geom::Vec3f*>(this->m_bufferDartPosition);

//	m_vbo0->bind();
//	glBufferData(GL_ARRAY_BUFFER, 2*m_nbDarts*sizeof(VEC3), 0, GL_STREAM_DRAW);
//	GLvoid* PositionDartsBuffer = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
//	VEC3* positionDartBuf = reinterpret_cast<VEC3*>(PositionDartsBuffer);

	std::vector<Dart> vecDartFaces;
	vecDartFaces.reserve(this->m_nbDarts/6);
	unsigned int posDBI = 0;

	//traverse each face of each volume
	TraversorCell<MAP, PFP::MAP::FACE_OF_PARENT> traFace(mapx);
	for (Dart d = traFace.begin(); d != traFace.end(); d = traFace.next())
	{
		vecDartFaces.push_back(d);

		std::vector<VEC3> vecPos;
		vecPos.reserve(16);

		// store the face & center
		float okv = 1.0f - kv;

		VEC3 vc = centerVolumes[d];
		
		VEC3 centerFace = Algo::Surface::Geometry::faceCentroidELW<PFP>(mapx, d, positions)*kv +vc*okv;

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

			*positionDartBuf++ = PFP::toVec3f(P);
			*positionDartBuf++ = PFP::toVec3f(PP);
			*positionDartBuf++ = PFP::toVec3f(Q);
			*positionDartBuf++ = PFP::toVec3f(QQ);
			*colorDartBuf++ = this->m_dartsColor;
			*colorDartBuf++ = this->m_dartsColor;
			*colorDartBuf++ = this->m_dartsColor;
			*colorDartBuf++ = this->m_dartsColor;

			this->m_attIndex[d] = posDBI;
			posDBI+=2;

			fv1[d] = P*0.9f + PP*0.1f;
			fv2x[d] = P*0.52f + PP*0.48f;
			fv2[d] = P*0.48f + PP*0.52f;
			Dart dx = mapx.beta0(d);
			fv1[dx] = Q*0.9f + QQ*0.1f;
			fv2[dx] = Q*0.52f + QQ*0.48f;
			fv2x[dx] = Q*0.48f + QQ*0.52f;

			this->m_attIndex[dx] = posDBI;
			posDBI+=2;

			d = mapx.phi1(d);
		}
	}

	this->m_vbo0->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*this->m_nbDarts*sizeof(Geom::Vec3f), this->m_bufferDartPosition, GL_STREAM_DRAW);
//	m_vbo0->bind();
//	glUnmapBuffer(GL_ARRAY_BUFFER);

	this->m_vbo4->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	// beta3
	this->m_vbo1->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*this->m_nbDarts*sizeof(Geom::Vec3f), 0, GL_STREAM_DRAW);
	GLvoid* PositionBuffer1 = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);

	// beta3
	this->m_vbo2->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*this->m_nbDarts*sizeof(Geom::Vec3f), 0, GL_STREAM_DRAW);
	GLvoid* PositionBuffer2 = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);

	// beta3
	this->m_vbo3->bind();
	glBufferData(GL_ARRAY_BUFFER, 2*this->m_nbDarts*sizeof(Geom::Vec3f), 0, GL_STREAM_DRAW);
	GLvoid* PositionBuffer3 = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);

	Geom::Vec3f* positionF1 = reinterpret_cast<Geom::Vec3f*>(PositionBuffer1);
	Geom::Vec3f* positionF2 = reinterpret_cast<Geom::Vec3f*>(PositionBuffer2);
	Geom::Vec3f* positionF3 = reinterpret_cast<Geom::Vec3f*>(PositionBuffer3);

	this->m_nbRel2 = 0;
	this->m_nbRel3 = 0;

	for(std::vector<Dart>::iterator face = vecDartFaces.begin(); face != vecDartFaces.end(); ++face)
	{
		Dart d = *face;
		do
		{
			Dart e = mapx.beta2(d);
			if (d < e)
			{
				*positionF2++ = PFP::toVec3f(fv2[d]);
				*positionF2++ = PFP::toVec3f(fv2x[e]);
				*positionF2++ = PFP::toVec3f(fv2[e]);
				*positionF2++ = PFP::toVec3f(fv2x[d]);
				this->m_nbRel2++;
			}
			e = mapx.beta3(d);
			if (!mapx.template isBoundaryMarked<3>(e) && (d < e))
			{
				*positionF3++ = PFP::toVec3f(fv2[d]);
				*positionF3++ = PFP::toVec3f(fv2x[e]);
				*positionF3++ = PFP::toVec3f(fv2[e]);
				*positionF3++ = PFP::toVec3f(fv2x[d]);
				this->m_nbRel3++;
			}
			d = mapx.beta0(d);
			e = mapx.beta2(d);
			if (d < e)
			{
				*positionF2++ = PFP::toVec3f(fv2[d]);
				*positionF2++ = PFP::toVec3f(fv2x[e]);
				*positionF2++ = PFP::toVec3f(fv2[e]);
				*positionF2++ = PFP::toVec3f(fv2x[d]);
				this->m_nbRel2++;
			}
			e = mapx.beta3(d);
			if (!mapx.template isBoundaryMarked<3>(e) && (d < e))
			{
				*positionF3++ = PFP::toVec3f(fv2[d]);
				*positionF3++ = PFP::toVec3f(fv2x[e]);
				*positionF3++ = PFP::toVec3f(fv2[e]);
				*positionF3++ = PFP::toVec3f(fv2x[d]);
				this->m_nbRel3++;
			}
			*positionF1++ = PFP::toVec3f(fv1[d]);
			d = mapx.beta1(d);
			*positionF1++ = PFP::toVec3f(fv1[d]);
			this->m_nbRel1++;
		} while (d != *face );
	}

	this->m_vbo3->bind();
	glUnmapBufferARB(GL_ARRAY_BUFFER);

	this->m_vbo2->bind();
	glUnmapBufferARB(GL_ARRAY_BUFFER);

	this->m_vbo1->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);

	this->m_vbo4->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

template<typename PFP>
void Topo3Render<PFP>::computeDartMiddlePositions(MAP& map, DartAttribute<VEC3, MAP>& posExpl)
{
	m_vbo0->bind();
	Geom::Vec3f* positionsPtr = reinterpret_cast<Geom::Vec3f*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY));

	for (Dart d = map.begin(); d != map.end(); map.next(d))
	{
		const Geom::Vec3f& v = (positionsPtr[m_attIndex[d]] + positionsPtr[m_attIndex[d]+1])*0.5f;
		posExpl[d] = PFP::toVec3f(v);
	}

	m_vbo0->bind();
	glUnmapBuffer(GL_ARRAY_BUFFER);
}

//template<typename PFP>
//void Topo3Render<PFP>::updateDataMap3OldFashioned(typename PFP::MAP& mapx, const typename PFP::TVEC3& positions, float ke, float kf, float kv)
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
Dart Topo3Render<PFP>::coneSelection(MAP& map, const Geom::Vec3f& rayA, const Geom::Vec3f& rayAB, float angle)
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
Dart Topo3Render<PFP>::raySelection(MAP& map, const Geom::Vec3f& rayA, const Geom::Vec3f& rayAB, float dmax)
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
