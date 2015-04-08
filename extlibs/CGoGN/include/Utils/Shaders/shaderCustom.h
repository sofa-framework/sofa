#ifndef __SHADER_CUSTOM__
#define __SHADER_CUSTOM__

#include "Utils/GLSLShader.h"
#include "Geometry/vector_gen.h"
#include "Geometry/matrix.h"


using namespace CGoGN;

#include "Utils/dll.h"

class CGoGN_UTILS_API ShaderCustom : public Utils::GLSLShader
{
protected:
	// shader sources
    static std::string vertexShaderText;
    static std::string fragmentShaderText;
    static std::string geometryShaderText;

    // uniform locations
	CGoGNGLuint m_unif_ambiant;
	CGoGNGLuint m_unif_diffuse;
	CGoGNGLuint m_unif_lightPos;
	CGoGNGLuint m_unif_explode;

	float m_explode;
	Geom::Vec4f m_ambiant;
	Geom::Vec4f m_diffuse;
	Geom::Vec3f m_light_pos;

	Utils::VBO* m_vboPos;

	void getLocations();

	void restoreUniformsAttribs();

public:
	ShaderCustom();

	void setExplode(float explode);

	void setAmbiant(const Geom::Vec4f& ambiant);

	void setDiffuse(const Geom::Vec4f& diffuse);

	void setLightPosition(const Geom::Vec3f& lp);

	void setParams(float explode, const Geom::Vec4f& ambiant, const Geom::Vec4f& diffuse, const Geom::Vec3f& lightPos);

	unsigned int setAttributePosition(Utils::VBO* vbo);

	void setTransformation(Geom::Matrix44f t);
};

#endif
