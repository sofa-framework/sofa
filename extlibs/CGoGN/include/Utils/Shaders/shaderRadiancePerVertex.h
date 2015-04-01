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

#ifndef __CGOGN_SHADER_RADIANCEPERVERTEX__
#define __CGOGN_SHADER_RADIANCEPERVERTEX__

#include "Utils/GLSLShader.h"
#include "Utils/clippingShader.h"
#include "Utils/textures.h"
#include "Geometry/vector_gen.h"

#include "Utils/dll.h"

namespace CGoGN
{

namespace Utils
{

class CGoGN_UTILS_API ShaderRadiancePerVertex : public ClippingShader
{
protected:
	// shader sources
    static std::string vertexShaderText;
    static std::string geometryShaderText;
    static std::string fragmentShaderText;

	static std::string vertexShaderInterpText;
	static std::string geometryShaderInterpText;
	static std::string fragmentShaderInterpText;

	CGoGNGLuint m_uniform_resolution;
	CGoGNGLuint m_uniform_tex;
	CGoGNGLuint m_uniform_K_tab;
	CGoGNGLuint m_uniform_cam;

    VBO* m_vboPos;
	VBO* m_vboNorm;
	VBO* m_vboParam;
	Utils::Texture<2, Geom::Vec3f>* m_tex_ptr;

	unsigned int m_tex_unit;
	int m_resolution;
	float* K_tab;
	Geom::Vec3f m_camera;

	bool m_fragInterp;

	static int index (int l, int m) { return l*(l+1)+m; } // compute indices in K_tab

public:
	ShaderRadiancePerVertex(int resolution, bool fraginterp = false);

	~ShaderRadiancePerVertex();

	void compile();

	void setCamera(const Geom::Vec3f& camera);

	void setFragInterp(bool fraginterp);
	bool getFragInterp() { return m_fragInterp; }

	unsigned int setAttributePosition(VBO* vbo);

	unsigned int setAttributeNormal(VBO* vbo);

	unsigned int setAttributeRadiance(VBO* vbo, Utils::Texture<2, Geom::Vec3f>* texture, GLenum texunit = GL_TEXTURE0);
};

} // namespace Utils

} // namespace CGoGN

#endif
