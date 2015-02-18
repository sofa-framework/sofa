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

#include "vector"

#include "Utils/vboRender.h"
#include "Utils/GLSLShader.h"

namespace CGoGN
{

namespace Utils
{

VBORender::VBORender()
{
	glGenBuffers(1, &m_indexBuffer) ;
	m_nbIndices = 0 ;
	m_primitiveType = POINTS;
}

VBORender::~VBORender()
{
	glDeleteBuffers(1, &m_indexBuffer) ;
}

void VBORender::setConnectivity(std::vector<GLuint>& tableIndices, int primitiveType)
{
	m_primitiveType = primitiveType ;
	m_nbIndices = tableIndices.size() ;
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer) ;
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_nbIndices * sizeof(GLuint), &(tableIndices[0]), GL_STREAM_DRAW) ;
}

void VBORender::draw(Utils::GLSLShader* sh)
{
	sh->enableVertexAttribs() ;

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer) ;
	switch(m_primitiveType)
	{
		case POINTS:
			glDrawElements(GL_POINTS, m_nbIndices, GL_UNSIGNED_INT, 0) ;
			break;
		case LINES:
			glDrawElements(GL_LINES, m_nbIndices, GL_UNSIGNED_INT, 0);
			break;
		case TRIANGLES:
			glDrawElements(GL_TRIANGLES, m_nbIndices, GL_UNSIGNED_INT, 0);
			break;
		default:
			break;
	}

	sh->disableVertexAttribs() ;
}

} // namespace Utils

} // namespace CGoGN
