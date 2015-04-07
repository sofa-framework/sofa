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

#ifndef __CGOGN_SHADER_FLAT_COLOR_PER_FACE__
#define __CGOGN_SHADER_FLAT_COLOR_PER_FACE__

#include "Utils/GLSLShader.h"
#include "Geometry/vector_gen.h"

#include "Utils/dll.h"

namespace CGoGN
{

namespace Utils
{

class CGoGN_UTILS_API ShaderFlatColor : public GLSLShader
{
protected:
	// shader sources
    static std::string vertexShaderText;
    static std::string fragmentShaderText;
    static std::string geometryShaderText;

    // uniform locations
	CGoGNGLuint m_unif_ambiant;
	CGoGNGLuint m_unif_lightPos;
	CGoGNGLuint m_unif_explode;

	float m_explode;
	Geom::Vec4f m_ambiant;
	Geom::Vec3f m_light_pos;

	VBO* m_vboPos;
	VBO* m_vboColor;

	void getLocations();

	void restoreUniformsAttribs();

public:
	ShaderFlatColor(bool averageColor=false);

	void setExplode(float explode);

	void setAmbiant(const Geom::Vec4f& ambiant);

	void setLightPosition(const Geom::Vec3f& lp);

	void setParams(float explode, const Geom::Vec4f& ambiant, const Geom::Vec3f& lightPos);

	unsigned int setAttributePosition(VBO* vbo);
	
	unsigned int setAttributeColor(VBO* vbo);
};

} // namespace Utils

} // namespace CGoGN

#endif
