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

#ifndef __CGOGN_SHADER_CPV__
#define __CGOGN_SHADER_CPV__

#include "Utils/GLSLShader.h"
#include "Utils/clippingShader.h"
#include "Geometry/vector_gen.h"

namespace CGoGN
{

namespace Utils
{

class ShaderColorPerVertex : public ClippingShader
{
protected:
	// shader sources
    static std::string vertexShaderText;
    static std::string fragmentShaderText;

    VBO* m_vboPos;
    VBO* m_vboCol;

    CGoGNGLuint m_unif_alpha;
    float m_opacity;

    void restoreUniformsAttribs();

public:
    ShaderColorPerVertex(bool black_is_transparent = false);

	/**
	 * set the VBO of position (vec3)
	 */
    unsigned int setAttributePosition(VBO* vbo);

	/**
	 * set the VBO of color (vec3)
	 */
    unsigned int setAttributeColor(VBO* vbo);

	/**
	 * set opacity (0=transparent / 1=opaque)
	 */
	void setOpacity(float op);

	float getOpacity() const  { return m_opacity;}
};

} // namespace Utils

} // namespace CGoGN

#endif
