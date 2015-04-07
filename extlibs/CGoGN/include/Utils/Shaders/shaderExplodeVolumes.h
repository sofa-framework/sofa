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

#ifndef __CGOGN_SHADER_EXPLODE_VOLUMES__
#define __CGOGN_SHADER_EXPLODE_VOLUMES__

#include "Utils/GLSLShader.h"
#include "Geometry/vector_gen.h"

#include "Utils/dll.h"

namespace CGoGN
{

namespace Utils
{

class CGoGN_UTILS_API ShaderExplodeVolumes : public GLSLShader
{
protected:
	// shader sources
    static std::string vertexShaderText;
    static std::string fragmentShaderText;
    static std::string geometryShaderText;

    // uniform locations
	CGoGNGLuint m_unif_ambiant;
	CGoGNGLuint m_unif_backColor;
	CGoGNGLuint m_unif_lightPos;
	CGoGNGLuint m_unif_explodeV;
	CGoGNGLuint m_unif_explodeF;
	CGoGNGLuint m_unif_plane;

//	local storage for uniforms
	float m_explodeV;
	float m_explodeF;
	Geom::Vec4f m_ambiant;
	Geom::Vec4f m_backColor;
	Geom::Vec3f m_light_pos;
	Geom::Vec4f m_plane;

	// VBO
	VBO* m_vboPos;
	VBO* m_vboColors;

	bool m_wcpf;
	bool m_wef;

	void getLocations();

	void restoreUniformsAttribs();

public:
	ShaderExplodeVolumes(bool withColorPerFace=false, bool withExplodeFace=false);

	void setExplodeVolumes(float explode);

	void setExplodeFaces(float explode);

	void setAmbiant(const Geom::Vec4f& ambiant);

	void setBackColor(const Geom::Vec4f& backColor);

	void setLightPosition(const Geom::Vec3f& lp);

	void setClippingPlane(const Geom::Vec4f& plane);

	void setParams(float explodeV, float explodeF, const Geom::Vec4f& ambiant, const Geom::Vec4f& diffuse, const Geom::Vec3f& lightPos, const Geom::Vec4f& plane);

	unsigned int setAttributePosition(VBO* vbo);

	unsigned int setAttributeColor(VBO* vbo);
};

} // namespace Utils

} // namespace CGoGN

#endif
