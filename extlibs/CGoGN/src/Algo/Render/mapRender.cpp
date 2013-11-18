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

#include "Algo/Render/GL2/mapRender.h"
#include "Utils/GLSLShader.h"

namespace CGoGN
{

namespace Algo
{

namespace Render
{

namespace GL2
{

MapRender::MapRender()
{
	glGenBuffers(SIZE_BUFFER, m_indexBuffers) ;
	for(unsigned int i = 0; i < SIZE_BUFFER; ++i)
	{
		m_nbIndices[i] = 0 ;
		m_indexBufferUpToDate[i] = false;
	}
}

MapRender::~MapRender()
{
	glDeleteBuffers(4, m_indexBuffers);
}

void MapRender::initPrimitives(int prim, std::vector<GLuint>& tableIndices)
{
	m_nbIndices[prim] = tableIndices.size();
	m_indexBufferUpToDate[prim] = true;

	// setup du buffer d'indices
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffers[prim]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_nbIndices[prim] * sizeof(GLuint), &(tableIndices[0]), GL_STREAM_DRAW);
}

void MapRender::draw(Utils::GLSLShader* sh, int prim)
{
	sh->enableVertexAttribs();

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffers[prim]);
	switch(prim)
	{
		case POINTS:
			glDrawElements(GL_POINTS, m_nbIndices[POINTS], GL_UNSIGNED_INT, 0) ;
			break;
		case LINES:
			glDrawElements(GL_LINES, m_nbIndices[LINES], GL_UNSIGNED_INT, 0);
			break;
		case TRIANGLES:
			glDrawElements(GL_TRIANGLES, m_nbIndices[TRIANGLES], GL_UNSIGNED_INT, 0);
			break;
		case BOUNDARY:
			glDrawElements(GL_LINES, m_nbIndices[BOUNDARY], GL_UNSIGNED_INT, 0);
			break;
		default:
			break;
	}

	sh->disableVertexAttribs();
}

unsigned int MapRender::drawSub(Utils::GLSLShader* sh, int prim, unsigned int nb_elm)
{
	sh->enableVertexAttribs();

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffers[prim]);
	switch(prim)
	{
		case POINTS:
			if (nb_elm > m_nbIndices[POINTS])
				nb_elm = m_nbIndices[POINTS];
			glDrawElements(GL_POINTS, nb_elm, GL_UNSIGNED_INT, 0) ;
			break;
		case LINES:
			if (2*nb_elm > m_nbIndices[LINES])
				nb_elm = m_nbIndices[LINES]/2;
			glDrawElements(GL_LINES, 2*nb_elm, GL_UNSIGNED_INT, 0);
			break;
		case TRIANGLES:
			if (3*nb_elm > m_nbIndices[TRIANGLES])
				nb_elm = m_nbIndices[TRIANGLES]/3;
			glDrawElements(GL_TRIANGLES, 3*nb_elm, GL_UNSIGNED_INT, 0);
			break;
		default:
			break;
	}

	sh->disableVertexAttribs();
	return nb_elm;
}

} // namespace GL2

} // namespace Render

} // namespace Algo

} // namespace CGoGN
