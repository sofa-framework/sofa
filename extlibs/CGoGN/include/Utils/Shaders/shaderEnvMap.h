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

#ifndef __CGOGN_SHADER_ENVMAP__
#define __CGOGN_SHADER_ENVMAP__

#include "Utils/GLSLShader.h"
#include "Utils/clippingShader.h"
#include "Geometry/vector_gen.h"

#include <string>

#ifdef CGOGN_WITH_QT
#include <QImage>
#endif

namespace CGoGN
{

namespace Utils
{
/**
 * Class for shader environment mapping (cube mapping)
 *
 * @warning shader code is GL2.0 compatible for GL3.0 replace textureCube by texture in fragment shader !
 */
class ShaderEnvMap : public ClippingShader
{
protected:
	// flag color per vertex or not
	bool m_with_color;
	// flag color per vertex or not
	bool m_with_eyepos;

	// shader sources OGL3
    static std::string vertexShaderText;
    static std::string fragmentShaderText;

    // uniform locations
	CGoGNGLuint m_unif_ambiant;
	CGoGNGLuint m_unif_diffuse;
	CGoGNGLuint m_unif_blend;
	CGoGNGLuint m_unif_lightPos;
	CGoGNGLuint m_unif_eyePos;
	CGoGNGLuint m_unif_envMap;

	//values
	Geom::Vec4f m_ambiant;
	Geom::Vec4f m_diffuse;
	float m_blend;
	Geom::Vec3f m_lightPos;
	Geom::Vec3f m_eyePos;

	CGoGNGLuint m_texId;

	VBO* m_vboPos;
	VBO* m_vboNormal;
	VBO* m_vboColor;

	void getLocations();

	void sendParams();

	void restoreUniformsAttribs();

public:
	ShaderEnvMap(bool doubleSided = false, bool withEyePosition=false);

	// inviduals parameter setting functions
	void setAmbiant(const Geom::Vec4f& ambiant);

	void setDiffuse(const Geom::Vec4f& diffuse);

	void setSpecular(const Geom::Vec4f& specular);

	void setBlendCoef(float blend);

	void setLightPosition(const Geom::Vec3f& lp);

	/// set eye position for VR environement
	void setEyePosition(const Geom::Vec3f& ep);

	const Geom::Vec4f& getAmbiant() const { return m_ambiant; }

	const Geom::Vec4f& getDiffuse() const { return m_diffuse; }

	float getBlendCoef() const { return m_blend; }

	const Geom::Vec3f& getLightPosition() const { return m_lightPos; }

	/**
	 * set all parameter in on call (one bind also)
	 */
	void setParams(const Geom::Vec4f& ambiant, const Geom::Vec4f& diffuse, float blend, const Geom::Vec3f& lightPos);

	// attributes
	unsigned int setAttributePosition(VBO* vbo);

	unsigned int setAttributeNormal(VBO* vbo);

	// optional attributes
	unsigned int setAttributeColor(VBO* vbo);
	void unsetAttributeColor();


	/**
	 * need to be called just before draw
	 */
	void predraw();

	/**
	 * need to be called just after draw
	 */
	void postdraw();

	void setCubeMap(unsigned int sz, unsigned char* Xpos, unsigned char* Ypos, unsigned char* Zpos, unsigned char* Xneg, unsigned char* Yneg, unsigned char* Zneg);

	/**
	 * set envmap texture image
	 * image looks  likre:
	 *  Y
	 * XZXZ
	 *  Y
	 * @parameter sz size of edge of cube in pixel
	 * @parameter ptr pointer on image data (RGB RGB RGB ....)
	 */
	bool setCubeMapImg(unsigned int sz, unsigned char *ptr);

	/**
	 * set colored plane for testing
	 */
	bool setCubeMapColored();

	/**
	 * set colored plane for testing
	 */
	bool setCubeMapCheckered();

#ifdef CGOGN_WITH_QT
	bool setCubeMap(const std::string& filename);
#endif

};

} // namespace Utils

} // namespace CGoGN

#endif
