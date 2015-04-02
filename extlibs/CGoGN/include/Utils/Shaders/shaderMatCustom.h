#ifndef SHADER_MATCUST
#define SHADER_MATCUST

#include "Utils/GLSLShader.h"
#include "Geometry/vector_gen.h"
#include "Geometry/matrix.h"

#include <string>

using namespace CGoGN;

#include "Utils/dll.h"

class CGoGN_UTILS_API ShaderMatCustom : public Utils::GLSLShader
{
protected:
	// flag color per vertex or not
	bool m_with_color;
	// flag color per vertex or not
	bool m_with_eyepos;	

	// shader sources OGL3
    static std::string vertexShaderText;
    static std::string fragmentShaderText;
    static std::string geometryShaderText;

    // uniform locations
	CGoGNGLuint m_unif_ambiant;
	CGoGNGLuint m_unif_diffuse;
	CGoGNGLuint m_unif_specular;
	CGoGNGLuint m_unif_shininess;
	CGoGNGLuint m_unif_lightPos;
	CGoGNGLuint m_unif_eyePos;

	//values
	Geom::Vec4f m_ambiant;
	Geom::Vec4f m_diffuse;
	Geom::Vec4f m_specular;
	float m_shininess;
	Geom::Vec3f m_lightPos;
	Geom::Vec3f m_eyePos;

	Utils::VBO* m_vboPos;
	Utils::VBO* m_vboNormal;
	Utils::VBO* m_vboColor;

	void getLocations();

	void sendParams();

	void restoreUniformsAttribs();

public:
	ShaderMatCustom(bool doubleSided = false, bool withEyePosition=false);

	// inviduals parameter setting functions
	void setAmbiant(const Geom::Vec4f& ambiant);

	void setDiffuse(const Geom::Vec4f& diffuse);

	void setSpecular(const Geom::Vec4f& specular);

	void setShininess(float shininess);

	void setLightPosition(const Geom::Vec3f& lp);
	
	/// set eye position for VR environement
	void setEyePosition(const Geom::Vec3f& ep);

	const Geom::Vec4f& getAmbiant() const { return m_ambiant; }

	const Geom::Vec4f& getDiffuse() const { return m_diffuse; }

	const Geom::Vec4f& getSpecular() const { return m_specular; }

	float getShininess() const { return m_shininess; }

	const Geom::Vec3f& getLightPosition() const { return m_lightPos; }

	/**
	 * set all parameter in on call (one bind also)
	 */
	void setParams(const Geom::Vec4f& ambiant, const Geom::Vec4f& diffuse, const Geom::Vec4f& specular, float shininess, const Geom::Vec3f& lightPos);

	// attributes
	unsigned int setAttributePosition(Utils::VBO* vbo);

	unsigned int setAttributeNormal(Utils::VBO* vbo);

	// optional attributes
	unsigned int setAttributeColor(::Utils::VBO* vbo);
	void unsetAttributeColor();

	void setTransformation(Geom::Matrix44f t);
};

#endif
