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

#ifndef __POINT_SPRITE_3D__
#define __POINT_SPRITE_3D__

#include "Utils/GLSLShader.h"
#include "Geometry/vector_gen.h"

namespace CGoGN { namespace Utils { class VBO; } }

namespace CGoGN
{

namespace Utils
{

class PointSprite : public Utils::GLSLShader
{
protected:
	static std::string vertexShaderText;
	static std::string geometryShaderText;
	static std::string fragmentShaderText;

	bool colorPerVertex;
	bool plane;

	CGoGNGLuint m_uniform_size;
	CGoGNGLuint m_uniform_color;
	CGoGNGLuint m_uniform_ambiant;
	CGoGNGLuint m_uniform_lightPos;
	CGoGNGLuint m_uniform_eyePos;

	VBO* m_vboPos;
	VBO* m_vboColor;

	float m_size;
	Geom::Vec4f m_color;
	Geom::Vec3f m_lightPos;
	Geom::Vec3f m_ambiant;
	Geom::Vec3f m_eyePos;

	void getLocations();

	void sendParams();

	void restoreUniformsAttribs();

public:
	/**
	 * init shaders & variables
	 * @param withColorPerVertex if true use setAttributeColor for per vertex color, else use setColor(color) for global color
	 * @param withPlane
	 */
	PointSprite(bool withColorPerVertex = false, bool withPlane = false);

	unsigned int setAttributePosition(VBO* vbo);

	unsigned int setAttributeColor(VBO* vbo);

	void setSize(float size);

	void setColor(const Geom::Vec4f& color);

	void setLightPosition(const Geom::Vec3f& pos);

	void setAmbiantColor(const Geom::Vec3f& amb);
	
	/**
	* set the plane of rendering for VR rendering
	*/
	void setEyePosition(const Geom::Vec3f& ep);
};

} // namespace Utils

} // namespace CGoGN

#endif
