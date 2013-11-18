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

#include "Algo/Render/GL2/topoPrimalRender.h"
#include "Topology/generic/dart.h"
#include "Utils/Shaders/shaderSimpleColor.h"
#include "Utils/Shaders/shaderColorPerVertex.h"

#include <string.h>

namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace GL2
{

TopoPrimalRender::TopoPrimalRender():
	m_nbDarts(0),
	m_nbRel1(0),
	m_topo_dart_width(2.0f),
	m_topo_relation_width(3.0f),
	m_color_save(NULL),
	m_dartsColor(1.0f,1.0f,1.0f),
	m_boundaryDartsColor(0.5f,0.5f,0.5f),
	m_bufferDartPosition(NULL)
{
	m_vbo0 = new Utils::VBO();
	m_vbo1 = new Utils::VBO();
	m_vbo2 = new Utils::VBO();

	m_vbo0->setDataSize(3);
	m_vbo1->setDataSize(3);
	m_vbo2->setDataSize(3);


	m_shader1 = new Utils::ShaderSimpleColor();
	m_shader2 = new Utils::ShaderColorPerVertex();

	// binding VBO - VA
	m_vaId = m_shader1->setAttributePosition(m_vbo1);

	m_shader2->setAttributePosition(m_vbo0);
	m_shader2->setAttributeColor(m_vbo2);

	// registering for auto matrices update
	Utils::GLSLShader::registerShader(NULL, m_shader1);
	Utils::GLSLShader::registerShader(NULL, m_shader2);
}

TopoPrimalRender::~TopoPrimalRender()
{
	Utils::GLSLShader::unregisterShader(NULL, m_shader2);
	Utils::GLSLShader::unregisterShader(NULL, m_shader1);

	delete m_shader2;
	delete m_shader1;
	delete m_vbo2;
	delete m_vbo1;
	delete m_vbo0;

	if (m_attIndex.map() != NULL)
		static_cast<AttribMap*>(m_attIndex.map())->removeAttribute(m_attIndex);

	if (m_color_save!=NULL)
	{
		delete[] m_color_save;
	}

	if (m_bufferDartPosition!=NULL)
		delete[] m_bufferDartPosition;
}

void TopoPrimalRender::setDartWidth(float dw)
{
	m_topo_dart_width = dw;
}

void TopoPrimalRender::setRelationWidth(float pw)
{
	m_topo_relation_width = pw;
}

void TopoPrimalRender::setDartColor(Dart d, float r, float g, float b)
{
	float RGB[6];
	RGB[0]=r; RGB[1]=g; RGB[2]=b;
	RGB[3]=r; RGB[4]=g; RGB[5]=b;
	m_vbo2->bind();
	glBufferSubData(GL_ARRAY_BUFFER, m_attIndex[d]*3*sizeof(float), 6*sizeof(float),RGB);
}

void TopoPrimalRender::setAllDartsColor(float r, float g, float b)
{
	m_vbo2->bind();
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

void TopoPrimalRender::setInitialDartsColor(float r, float g, float b)
{
	m_dartsColor = Geom::Vec3f(r,g,b);
}

void TopoPrimalRender::setInitialBoundaryDartsColor(float r, float g, float b)
{
	m_boundaryDartsColor = Geom::Vec3f(r,g,b);
}


void TopoPrimalRender::drawDarts()
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


void TopoPrimalRender::drawRelation1()
{
	if (m_nbRel1==0)
		return;

	m_shader1->changeVA_VBO(m_vaId, m_vbo1);
	m_shader1->setColor(Geom::Vec4f(1.0f,0.0f,0.0f,0.0f));
	m_shader1->enableVertexAttribs();

	glLineWidth(m_topo_relation_width);
	glDrawArrays(GL_LINES, 0, m_nbRel1*2);

	m_shader1->disableVertexAttribs();
}



void TopoPrimalRender::drawTopo()
{
	drawDarts();
	drawRelation1();
}


void TopoPrimalRender::overdrawDart(Dart d, float width, float r, float g, float b)
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


void TopoPrimalRender::pushColors()
{
	m_color_save = new float[6*m_nbDarts];
	m_vbo2->bind();
	void* colorBuffer = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);

	memcpy(m_color_save, colorBuffer, 6*m_nbDarts*sizeof(float));
	glUnmapBuffer(GL_ARRAY_BUFFER);
}


void TopoPrimalRender::popColors()
{
	m_vbo2->bind();
	void* colorBuffer = glMapBufferARB(GL_ARRAY_BUFFER, GL_READ_WRITE);

	memcpy(colorBuffer, m_color_save, 6*m_nbDarts*sizeof(float));
	glUnmapBuffer(GL_ARRAY_BUFFER);

	delete[] m_color_save;
	m_color_save=0;
}


Dart TopoPrimalRender::colToDart(float* color)
{
	unsigned int r = (unsigned int)(color[0]*255.0f);
	unsigned int g = (unsigned int)(color[1]*255.0f);
	unsigned int b = (unsigned int)(color[2]*255.0f);

	unsigned int id = r + 255*g +255*255*b;

	if (id == 0)
		return NIL;
	return Dart(id-1);
}


void TopoPrimalRender::dartToCol(Dart d, float& r, float& g, float& b)
{
	// here use dart.index beacause it is what we want (and not map.dartIndex(d) !!)
	unsigned int lab = d.index + 1; // add one to avoid picking the black of screen

	r = float(lab%255) / 255.0f; lab = lab/255;
	g = float(lab%255) / 255.0f; lab = lab/255;
	b = float(lab%255) / 255.0f; lab = lab/255;
	if (lab!=0)
		CGoGNerr << "Error picking color, too many darts"<< CGoGNendl;
}




Dart TopoPrimalRender::pickColor(unsigned int x, unsigned int y)
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

void TopoPrimalRender::svgout2D(const std::string& filename, const glm::mat4& model, const glm::mat4& proj)
{
	Utils::SVG::SVGOut svg(filename,model,proj);
	toSVG(svg);
	svg.write();
}

void TopoPrimalRender::toSVG(Utils::SVG::SVGOut& svg)
{
	// alpha2
	Utils::SVG::SvgGroup* svg2 = new Utils::SVG::SvgGroup("alpha2", svg.m_model, svg.m_proj);
	Geom::Vec3f* ptr = reinterpret_cast<Geom::Vec3f*>(m_vbo1->lockPtr());
	svg2->setWidth(m_topo_relation_width);
	svg2->beginLines();
	for (unsigned int i=0; i<m_nbRel1; ++i)
	{
		Geom::Vec3f P = ptr[2*i];
		Geom::Vec3f Q = ptr[2*i+1];
		svg2->addLine(P, Q, Geom::Vec3f(0.8f,0.0f,0.0f));
	}
	svg2->endLines();
	m_vbo1->releasePtr();
	svg.addGroup(svg2);

	const Geom::Vec3f* colorsPtr = reinterpret_cast<const Geom::Vec3f*>(m_vbo2->lockPtr());
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
	m_vbo2->releasePtr();

}


}//end namespace GL2

}//end namespace Render

}//end namespace Algo

}//end namespace CGoGN
