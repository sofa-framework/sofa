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

#ifndef __CGOGN_SHADER_PHONG_TEXTURE__
#define __CGOGN_SHADER_PHONG_TEXTURE__


#include "Geometry/vector_gen.h"
#include "Utils/GLSLShader.h"
#include "Utils/clippingShader.h"
#include "Utils/textures.h"
#include "Utils/gl_def.h"

#include "Utils/dll.h"

namespace CGoGN
{

namespace Utils
{

class CGoGN_UTILS_API ShaderPhongTexture : public ClippingShader
{
protected:
	// shader sources
	static std::string vertexShaderText;
	static std::string fragmentShaderText;

	CGoGNGLuint m_unif_unit;
	int m_unit;
	Utils::GTexture* m_tex_ptr;

	// flag color per vertex or not
	bool m_with_eyepos;	

    // uniform locations
	CGoGNGLuint m_unif_ambient;
	CGoGNGLuint m_unif_specular;
	CGoGNGLuint m_unif_shininess;
	CGoGNGLuint m_unif_lightPos;
	CGoGNGLuint m_unif_eyePos;

	//values
	float m_ambient;
	Geom::Vec4f m_specular;
	float m_shininess;
	Geom::Vec3f m_lightPos;
	Geom::Vec3f m_eyePos;

	VBO* m_vboPos;
	VBO* m_vboNormal;
	VBO* m_vboTexCoord;

	void getLocations();

	void sendParams();

	void restoreUniformsAttribs();




public:
	ShaderPhongTexture(bool doubleSided = false, bool withEyePosition=false);

	/**
	 * choose the texture unit engine to use for this texture
	 */
	void setTextureUnit(GLenum texture_unit);

	/**
	 * set the texture to use
	 */
	void setTexture(Utils::GTexture* tex);

	/**
	 * activation of texture unit with set texture
	 */
	void activeTexture();
	
	/**
	 * activation of texture unit with texture id
	 */
	void activeTexture(CGoGNGLuint texId);


	// inviduals parameter setting functions
	void setAmbient(float ambient);

	void setSpecular(const Geom::Vec4f& specular);

	void setShininess(float shininess);

	void setLightPosition(const Geom::Vec3f& lp);
	
	/// set eye position for VR environement
	void setEyePosition(const Geom::Vec3f& ep);

	float getAmbiant() const { return m_ambient; }

	const Geom::Vec4f& getSpecular() const { return m_specular; }

	float getShininess() const { return m_shininess; }

	const Geom::Vec3f& getLightPosition() const { return m_lightPos; }

	/**
	 * set all parameter in on call (one bind also)
	 */
	void setParams(float ambient, const Geom::Vec4f& specular, float shininess, const Geom::Vec3f& lightPos);

	unsigned int setAttributePosition(VBO* vbo);

	unsigned int setAttributeNormal(VBO* vbo);

	unsigned int setAttributeTexCoord(VBO* vbo);
};

} // namespace Utils

} // namespace CGoGN


#endif /* __CGOGN_SHADER_SIMPLETEXTURE__ */
