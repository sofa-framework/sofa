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

#ifndef __CGOGN_SHADER_EXPLODE_VOLUMES_LINES__
#define __CGOGN_SHADER_EXPLODE_VOLUMES_LINES__

#include "Utils/GLSLShader.h"
#include "Geometry/vector_gen.h"

#include "Utils/dll.h"

namespace CGoGN
{

namespace Utils
{

class CGoGN_UTILS_API ShaderExplodeVolumesLines : public GLSLShader
{
protected:
	// shader sources
    static std::string vertexShaderText;
    static std::string fragmentShaderText;
    static std::string geometryShaderText;

    // uniform locations
	CGoGNGLuint m_unif_color;
	CGoGNGLuint m_unif_explodeV;
	CGoGNGLuint m_unif_plane;

	float m_explodeV;
	Geom::Vec4f m_color;
	Geom::Vec4f m_plane;

	VBO* m_vboPos;

	void getLocations();

	void restoreUniformsAttribs();

public:
	ShaderExplodeVolumesLines();

	void setExplodeVolumes(float explode);

	void setColor(const Geom::Vec4f& color);

	const Geom::Vec4f& getColor() const;

	void setClippingPlane(const Geom::Vec4f& plane);

	void setParams(float explodeV, const Geom::Vec4f& color, const Geom::Vec4f& plane);

	unsigned int setAttributePosition(VBO* vbo);
};

} // namespace Utils

} // namespace CGoGN

#endif
