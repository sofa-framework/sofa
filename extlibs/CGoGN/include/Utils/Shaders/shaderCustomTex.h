#ifndef __SHADER_CUSTOMTEX__
#define __SHADER_CUSTOMTEX__

#include "Utils/GLSLShader.h"
#include "Geometry/vector_gen.h"

#include "Utils/clippingShader.h"
#include "Utils/textures.h"
#include "Utils/gl_def.h"
#include "Geometry/matrix.h"

using namespace CGoGN;

#include "Utils/dll.h"

class CGoGN_UTILS_API ShaderCustomTex : public Utils::ClippingShader
{
protected:
	// shader sources
	static std::string vertexShaderText;
	static std::string fragmentShaderText;
	static std::string geometryShaderText;

	CGoGNGLuint m_unif_unit;
	int m_unit;

	Geom::Vec4f m_col;

	Utils::GTexture* m_tex_ptr;
	Utils::VBO* m_vboPos;
	Utils::VBO* m_vboNormal;
	Utils::VBO* m_vboTexCoord;

	void restoreUniformsAttribs();

public:
	ShaderCustomTex();

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

	unsigned int setAttributePosition(Utils::VBO* vbo);

	unsigned int setAttributeNormal(Utils::VBO* vbo);

	unsigned int setAttributeTexCoord(Utils::VBO* vbo);

	void setBaseColor(Geom::Vec4f col);

	void setTransformation(Geom::Matrix44f t);
};

#endif
