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

#include "GL/glew.h"

#define CGoGN_UTILS_DLL_EXPORT 1
#include "Utils/drawer.h"
#include "Utils/Shaders/shaderColorPerVertex.h"

#include "Utils/vbo_base.h"
#include "Utils/svg.h"

namespace CGoGN
{

namespace Utils
{

Drawer::Drawer() : m_currentWidth(1.0f)
{
	m_vboPos = new Utils::VBO();
	m_vboPos->setDataSize(3);

	m_vboCol = new Utils::VBO();
	m_vboCol->setDataSize(3);

	m_shader = new Utils::ShaderColorPerVertex();

	m_shader->setAttributePosition(m_vboPos);
	m_shader->setAttributeColor(m_vboCol);

	Utils::GLSLShader::registerShader(NULL, m_shader);

	m_dataPos.reserve(128);
	m_dataCol.reserve(128);
	m_begins.reserve(16);
//	m_modes.reserve(16);
}

Drawer::~Drawer()
{
	Utils::GLSLShader::unregisterShader(NULL, m_shader);
	delete m_vboPos;
	delete m_vboCol;
	delete m_shader;
}

Utils::ShaderColorPerVertex* Drawer::getShader()
{
	return m_shader;
}

void Drawer::lineWidth(float lw)
{
	m_currentWidth = lw;
}

void Drawer::pointSize(float ps)
{
	m_currentSize = ps;
}

int Drawer::begin(GLenum mode)
{
	int res = int(m_begins.size());
	if (mode == GL_POINTS)
        m_begins.push_back(PrimParam(m_dataPos.size(), mode, m_currentSize));
	else
        m_begins.push_back(PrimParam(m_dataPos.size(), mode, m_currentWidth));
	return res;
}

void Drawer::end()
{
	if (m_begins.empty())
		return;

    m_begins.back().nb = m_dataPos.size() - m_begins.back().begin;
}

void Drawer::color(const Geom::Vec3f& col)
{
	if (m_dataPos.size() == m_dataCol.size())
		m_dataCol.push_back(col);
	else
		m_dataCol.back() = col;
}

void Drawer::color3f(float r, float g, float b)
{
	color(Geom::Vec3f(r,g,b));
}

unsigned int Drawer::vertex(const Geom::Vec3f& v)
{
	if (m_dataPos.size() == m_dataCol.size())
	{
		if (m_dataCol.empty())
			m_dataCol.push_back(Geom::Vec3f(1.0f, 1.0f, 1.0f));
		else
			m_dataCol.push_back( m_dataCol.back());
	}
	m_dataPos.push_back(v);
    return  m_dataPos.size()-1u;
}

unsigned int Drawer::vertex3f(float r, float g, float b)
{
	return vertex(Geom::Vec3f(r,g,b));
}

void Drawer::newList(GLenum comp)
{
	m_compile = comp;
	m_dataPos.clear();
	m_dataCol.clear();
	m_begins.clear();
}

void Drawer::endList()
{
    unsigned int nbElts = m_dataPos.size();
	
	if (nbElts == 0)
		return;

	m_vboPos->bind();
	glBufferData(GL_ARRAY_BUFFER, nbElts * sizeof(Geom::Vec3f), &(m_dataPos[0]), GL_STREAM_DRAW);

	m_vboCol->bind();
	glBufferData(GL_ARRAY_BUFFER, nbElts * sizeof(Geom::Vec3f), &(m_dataCol[0]), GL_STREAM_DRAW);

	// free memory
	std::vector<Geom::Vec3f> tempo;
	tempo.swap(m_dataPos);
	std::vector<Geom::Vec3f> tempo2;
	tempo2.swap(m_dataCol);

	if (m_compile != GL_COMPILE)
		callList();
}


void Drawer::updatePositions(unsigned int first, unsigned int nb, const Geom::Vec3f* P)
{
	m_vboPos->bind();
	glBufferSubData(GL_ARRAY_BUFFER, first * sizeof(Geom::Vec3f), nb * sizeof(Geom::Vec3f), P);
}

void Drawer::updatePositions(unsigned int first, unsigned int nb, const float* P)
{
	m_vboPos->bind();
	glBufferSubData(GL_ARRAY_BUFFER, first * 3 * sizeof(float), nb * 3 * sizeof(float), P);
}


void Drawer::callList(float opacity)
{
	if (m_begins.empty())
		return;

	m_shader->setOpacity(opacity);

	m_shader->enableVertexAttribs();
	for (std::vector<PrimParam>::iterator pp = m_begins.begin(); pp != m_begins.end(); ++pp)
	{
		if (pp->mode == GL_POINTS)
			glPointSize(pp->width);
		if ((pp->mode == GL_LINES) || (pp->mode == GL_LINE_LOOP) || (pp->mode == GL_LINE_STRIP))
			glLineWidth(pp->width);
		glDrawArrays(pp->mode, pp->begin, pp->nb);
	}
 	m_shader->disableVertexAttribs();
}


void Drawer::callSubList(int index, float opacity)
{
	if (index >= int(m_begins.size()))
		return;

	m_shader->setOpacity(opacity);

	m_shader->enableVertexAttribs();

	PrimParam* pp = & (m_begins[index]);

	if (pp->mode == GL_POINTS)
		glPointSize(pp->width);
	if ((pp->mode == GL_LINES) || (pp->mode == GL_LINE_LOOP) || (pp->mode == GL_LINE_STRIP))
		glLineWidth(pp->width);
	glDrawArrays(pp->mode, pp->begin, pp->nb);

	m_shader->disableVertexAttribs();
}

void Drawer::callSubLists(int first, int nb, float opacity)
{
	m_shader->setOpacity(opacity);
	m_shader->enableVertexAttribs();

	int last = first+nb;
	for (int i = first; i< last; ++i)
		if (i < int(m_begins.size()))
		{
			PrimParam* pp = & (m_begins[i]);

			if (pp->mode == GL_POINTS)
				glPointSize(pp->width);
			if ((pp->mode == GL_LINES) || (pp->mode == GL_LINE_LOOP) || (pp->mode == GL_LINE_STRIP))
				glLineWidth(pp->width);
			glDrawArrays(pp->mode, pp->begin, pp->nb);
		}

	m_shader->disableVertexAttribs();
}

void Drawer::callSubLists(std::vector<int> indices, float opacity)
{
	m_shader->setOpacity(opacity);

	m_shader->enableVertexAttribs();

	for (std::vector<int>::iterator it = indices.begin(); it != indices.end(); ++it)
		if (*it < int(m_begins.size()))
		{
			PrimParam* pp = & (m_begins[*it]);

			if (pp->mode == GL_POINTS)
				glPointSize(pp->width);
			if ((pp->mode == GL_LINES) || (pp->mode == GL_LINE_LOOP) || (pp->mode == GL_LINE_STRIP))
				glLineWidth(pp->width);
			glDrawArrays(pp->mode, pp->begin, pp->nb);
		}

	m_shader->disableVertexAttribs();
}


void Drawer::toSVG(Utils::SVG::SVGOut& svg)
{
	const Geom::Vec3f* ptrP = reinterpret_cast<Geom::Vec3f*>(m_vboPos->lockPtr());
	const Geom::Vec3f* ptrC = reinterpret_cast<Geom::Vec3f*>(m_vboCol->lockPtr());

	Utils::SVG::SvgGroup* svg1 = new Utils::SVG::SvgGroup("points", svg.m_model, svg.m_proj);
	Utils::SVG::SvgGroup* svg2 = new Utils::SVG::SvgGroup("lines", svg.m_model, svg.m_proj);
	Utils::SVG::SvgGroup* svg3 = new Utils::SVG::SvgGroup("faces", svg.m_model, svg.m_proj);

	for (std::vector<PrimParam>::iterator pp = m_begins.begin(); pp != m_begins.end(); ++pp)
	{
		svg1->setWidth(pp->width);
		if (pp->mode == GL_POINTS)
		{
			unsigned int end = pp->begin + pp->nb;
			svg1->beginPoints();
			for (unsigned int i=pp->begin; i<end; ++i)
				svg1->addPoint(ptrP[i], ptrC[i]);
			svg1->endPoints();
		}

		svg2->setWidth(pp->width);
		if (pp->mode == GL_LINES)
		{
			unsigned int end = pp->begin + pp->nb;
			svg2->beginLines();
			for (unsigned int i=pp->begin; i<end; i+=2)
				svg2->addLine(ptrP[i], ptrP[i+1], ptrC[i]);
			svg2->endLines();
		}

		svg3->setWidth(pp->width);
		if ((pp->mode == GL_LINE_LOOP) || (pp->mode == GL_POLYGON))
		{
			unsigned int end = pp->begin + pp->nb-1;
			svg3->beginLines();
			for (unsigned int i=pp->begin; i<=end; ++i)
				svg3->addLine(ptrP[i], ptrP[i+1], ptrC[i]);
			svg3->addLine(ptrP[end], ptrP[pp->begin], ptrC[end]);
			svg3->endLines();
		}
		if (pp->mode == GL_TRIANGLES)
		{
			unsigned int end = pp->begin + pp->nb;
			svg3->beginLines();
			for (unsigned int i=pp->begin; i<end; i+=3)
			{
				svg3->addLine(ptrP[i],   ptrP[i+1], ptrC[i]);
				svg3->addLine(ptrP[i+1], ptrP[i+2], ptrC[i+1]);
				svg3->addLine(ptrP[i+2], ptrP[i],   ptrC[i+2]);
			}
			svg3->endLines();
		}
		if (pp->mode == GL_QUADS)
		{
			unsigned int end = pp->begin + pp->nb;
			svg3->beginLines();
			for (unsigned int i=pp->begin; i<end; i+=4)
			{
				svg3->addLine(ptrP[i],   ptrP[i+1], ptrC[i]);
				svg3->addLine(ptrP[i+1], ptrP[i+2], ptrC[i+1]);
				svg3->addLine(ptrP[i+2], ptrP[i+3], ptrC[i+2]);
				svg3->addLine(ptrP[i+3], ptrP[i],   ptrC[i+3]);
			}
			svg3->endLines();
		}
	}

	svg.addGroup(svg1);
	svg.addGroup(svg2);
	svg.addGroup(svg3);

	m_vboPos->releasePtr();
	m_vboCol->releasePtr();
}

} // namespace Utils

} // namespace CGoGN
