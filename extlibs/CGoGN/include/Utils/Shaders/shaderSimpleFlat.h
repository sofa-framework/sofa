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

#ifndef __CGOGN_SHADER_SFLAT__
#define __CGOGN_SHADER_SFLAT__

#include "Utils/GLSLShader.h"
#include "Utils/clippingShader.h"
#include "Geometry/vector_gen.h"

#include <string>

namespace CGoGN
{

namespace Utils
{

class ShaderSimpleFlat : public ClippingShader
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
	CGoGNGLuint m_unif_lightPos;

	//values
	Geom::Vec4f m_ambiant;
	Geom::Vec4f m_diffuse;
	Geom::Vec3f m_lightPos;

	VBO* m_vboPos;
	VBO* m_vboColor;

	void getLocations();

	void sendParams();

	void restoreUniformsAttribs();

public:
	ShaderSimpleFlat(bool doubleSided = false);

	// inviduals parameter setting functions
	void setAmbiant(const Geom::Vec4f& ambiant);

	void setDiffuse(const Geom::Vec4f& diffuse);


	void setLightPosition(const Geom::Vec3f& lp);
	
	const Geom::Vec4f& getAmbiant() const { return m_ambiant; }

	const Geom::Vec4f& getDiffuse() const { return m_diffuse; }

	const Geom::Vec3f& getLightPosition() const { return m_lightPos; }

	/**
	 * set all parameter in on call (one bind also)
	 */
	void setParams(const Geom::Vec4f& ambiant, const Geom::Vec4f& diffuse, const Geom::Vec3f& lightPos);

	// attributes
	unsigned int setAttributePosition(VBO* vbo);

	// optional attributes
	unsigned int setAttributeColor(VBO* vbo);
	void unsetAttributeColor();
};

} // namespace Utils

} // namespace CGoGN

#endif
