/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include<sofa/helper/gl/ps3gl_compat.h>

ImmediateModeState glImmediateMode;
void (*glutDisplay)()=NULL;
void (*glutIdle)()=NULL;

ImmediateModeState::ImmediateModeState()
{
	m_vertices = (float*) memalign(128, 65536*4*sizeof(float));
	m_normals = (float*) memalign(128, 65536*3*sizeof(float));
	m_colors = (float*) memalign(128, 65536*4*sizeof(float));
	m_texcoords = (float*) memalign(128, 65536*4*sizeof(float));
	m_numVertices = 0;
	m_numColors = 0;
	m_numNormals = 0;
	m_numTexcoords = 0;

	m_ComponentCountPerTexCoords = 0;
	m_ComponentCountPerColors = 0;
	m_ComponentCountPerVertices = 0;
};

void ImmediateModeState::Reset()
{

}

void ImmediateModeState::Flush()
{
	if(m_numVertices)
	{
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(m_ComponentCountPerVertices, GL_FLOAT, 0, m_vertices);
	}

	if(m_numColors && !m_bDefaultColor)
	{
		glEnableClientState(GL_COLOR_ARRAY);
		glColorPointer(m_ComponentCountPerColors, GL_FLOAT, m_ComponentCountPerColors*sizeof(float), m_colors);
	}
	else if(m_bDefaultColor)
	{
		::glColor4f(m_colors[0], m_colors[1], m_colors[2], 1.0f);
	}
	
	if(m_numNormals)
	{
		glEnableClientState(GL_NORMAL_ARRAY);
		glNormalPointer(GL_FLOAT, 3*sizeof(float), m_normals);
	}
	
	if(m_numTexcoords)
	{
		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glTexCoordPointer(m_ComponentCountPerTexCoords, GL_FLOAT, 0, m_texcoords);
	}

	
	if(m_numVertices)
	{
		GLuint numIndices =  m_numVertices/m_ComponentCountPerVertices;
		glDrawArrays(m_mode, 0, numIndices);
	}
	
	if(m_numColors) glDisableClientState(GL_COLOR_ARRAY);
	if(m_numNormals) glDisableClientState(GL_NORMAL_ARRAY);
	if(m_numVertices) glDisableClientState(GL_VERTEX_ARRAY);
	if(m_numTexcoords) glDisableClientState(GL_TEXTURE_COORD_ARRAY);

	m_numVertices = 0;
	m_numColors = 0;
	m_numNormals = 0;
	m_numTexcoords = 0;

	m_ComponentCountPerTexCoords = 0;
	m_ComponentCountPerColors = 0;
	m_ComponentCountPerVertices = 0;

}
