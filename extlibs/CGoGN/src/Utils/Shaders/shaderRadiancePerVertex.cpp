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
#define CGoGN_UTILS_DLL_EXPORT 1

#include "Utils/Shaders/shaderRadiancePerVertex.h"

namespace CGoGN
{

namespace Utils
{

#include "shaderRadiancePerVertex.vert"
#include "shaderRadiancePerVertex.geom"
#include "shaderRadiancePerVertex.frag"

#include "shaderRadiancePerVertexInterp.vert"
#include "shaderRadiancePerVertexInterp.geom"
#include "shaderRadiancePerVertexInterp.frag"

ShaderRadiancePerVertex::ShaderRadiancePerVertex(int resolution, bool fraginterp) :
	m_vboPos(NULL),
	m_vboNorm(NULL),
	m_vboParam(NULL),
	m_tex_ptr(NULL),
	m_tex_unit(-1),
	m_resolution(resolution),
	K_tab(NULL),
	m_fragInterp(fraginterp)
{
	compile();
}

ShaderRadiancePerVertex::~ShaderRadiancePerVertex()
{
	if (K_tab != NULL)
		delete[] K_tab ;
}

void ShaderRadiancePerVertex::compile()
{
	if (m_fragInterp)
	{
		m_nameVS = "ShaderRadiancePerVertexInterp_vs";
		m_nameFS = "ShaderRadiancePerVertexInterp_fs";
		m_nameGS = "ShaderRadiancePerVertexInterp_gs";
	}
	else
	{
		m_nameVS = "ShaderRadiancePerVertex_vs";
		m_nameFS = "ShaderRadiancePerVertex_fs";
		m_nameGS = "ShaderRadiancePerVertex_gs";
	}

	const int nb_coefs = (m_resolution+1)*(m_resolution+1);
	if (m_resolution != -1)
	{
		if (K_tab != NULL)
			delete[] K_tab;
		K_tab = new float[nb_coefs];
		for(int l = 0; l <= m_resolution; l++)
		{
			// recursive computation of the squares
			K_tab[index(l,0)] = (2*l+1) / (4*M_PI);
			for (int m = 1; m <= l; m++)
				K_tab[index(l,m)] = K_tab[index(l,m-1)] / (l-m+1) / (l+m);
			// square root + symmetry
			K_tab[index(l,0)] = sqrt(K_tab[index(l,0)]);
			for (int m = 1; m <= l; m++)
			{
				K_tab[index(l,m)] = sqrt(K_tab[index(l,m)]);
				K_tab[index(l,-m)] = K_tab[index(l,m)];
			}
		}
	}

	std::stringstream s ;
	s << "#define NB_COEFS " << nb_coefs << std::endl ;

	std::string glxvert(GLSLShader::defines_gl());
	if (!m_fragInterp)
	{
		glxvert.append(s.str()) ;
		glxvert.append(vertexShaderText);
	}
	else
		glxvert.append(vertexShaderInterpText);

	std::string glxgeom = GLSLShader::defines_Geom("triangles", "triangle_strip", 3) ;
	if (!m_fragInterp)
		glxgeom.append(geometryShaderText);
	else
		glxgeom.append(geometryShaderInterpText);

	std::string glxfrag(GLSLShader::defines_gl());
	if (!m_fragInterp)
		glxfrag.append(fragmentShaderText) ;
	else
	{
		glxfrag.append(s.str()) ;
		glxfrag.append(fragmentShaderInterpText) ;
	}

	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str(), glxgeom.c_str(), GL_TRIANGLES, GL_TRIANGLE_STRIP, 3);

	bind();
	*m_uniform_resolution = glGetUniformLocation(this->program_handler(), "resolution");
	*m_uniform_K_tab = glGetUniformLocation(this->program_handler(), "K_tab");
	*m_uniform_cam = glGetUniformLocation(this->program_handler(), "camera");
	*m_uniform_tex = glGetUniformLocation(this->program_handler(), "texture");

	glUniform1i(*m_uniform_resolution, m_resolution);
	glUniform1fv(*m_uniform_K_tab, nb_coefs, K_tab);
	unbind();
}

void ShaderRadiancePerVertex::setCamera(const Geom::Vec3f& camera)
{
	m_camera = camera;
	bind();
	glUniform3fv(*m_uniform_cam, 1, m_camera.data()) ;
	unbind();
}

unsigned int ShaderRadiancePerVertex::setAttributePosition(VBO* vbo)
{
	m_vboPos = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexPosition", vbo);
	unbind();
	return id;
}

unsigned int ShaderRadiancePerVertex::setAttributeNormal(VBO* vbo)
{
	m_vboNorm = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexNormal", vbo);
	unbind();
	return id;
}

void ShaderRadiancePerVertex::setFragInterp(bool fraginterp)
{
	m_fragInterp = fraginterp ;
	compile() ;
}

unsigned int ShaderRadiancePerVertex::setAttributeRadiance(VBO* vbo, Utils::Texture<2, Geom::Vec3f>* texture, GLenum tex_unit)
{
	m_vboParam = vbo;
	m_tex_ptr = texture;
	m_tex_unit = tex_unit - GL_TEXTURE0;

	bind();
	unsigned int id = bindVA_VBO("VertexParam", vbo);
	glUniform1iARB(*m_uniform_tex, m_tex_unit);
	glActiveTexture(tex_unit);
	if (m_tex_ptr)
		m_tex_ptr->bind() ;
	unbind();

	return id;
}

} // namespace Utils

} // namespace CGoGN
