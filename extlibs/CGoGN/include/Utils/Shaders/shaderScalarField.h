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

#ifndef __CGOGN_SHADER_SCALARFIELD__
#define __CGOGN_SHADER_SCALARFIELD__

#include "Utils/GLSLShader.h"
#include "Utils/clippingShader.h"
#include "Geometry/vector_gen.h"

#include "Utils/dll.h"

namespace CGoGN
{

namespace Utils
{

class CGoGN_UTILS_API ShaderScalarField : public ClippingShader
{
protected:
	// shader sources
    static std::string vertexShaderText;
    static std::string fragmentShaderText;

	CGoGNGLuint m_uniform_minValue;
	CGoGNGLuint m_uniform_maxValue;
	CGoGNGLuint m_uniform_colorMap;
	CGoGNGLuint m_uniform_expansion;

	VBO* m_vboPos;
	VBO* m_vboScal;

	float m_minValue;
	float m_maxValue;
	int m_colorMap;
	int m_expansion;

	void getLocations();

	void sendParams();

    void restoreUniformsAttribs();

public:
    ShaderScalarField();

    unsigned int setAttributePosition(VBO* vbo);

    unsigned int setAttributeScalar(VBO* vbo);

	void setMinValue(float f);

	void setMaxValue(float f);

	void setColorMap(int i);

	void setExpansion(int i);
};

} // namespace Utils

} // namespace CGoGN

#endif
