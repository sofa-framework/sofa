#define CGoGN_UTILS_DLL_EXPORT 1

#include <GL/glew.h>
#include <string.h>
#include "Utils/Shaders/shaderMatCustom.h"

#include "shaderMatCustom.vert"
#include "shaderMatCustom.frag"
//#include "shaderMatCustom.geom"

ShaderMatCustom::ShaderMatCustom(bool doubleSided, bool withEyePosition):
	m_with_color(false),
	m_with_eyepos(withEyePosition),
	m_ambiant(Geom::Vec4f(0.05f,0.05f,0.1f,0.0f)),
	m_diffuse(Geom::Vec4f(0.1f,1.0f,0.1f,0.0f)),
	m_specular(Geom::Vec4f(1.0f,1.0f,1.0f,0.0f)),
	m_shininess(100.0f),
	m_lightPos(Geom::Vec3f(10.0f,10.0f,1000.0f)),
	m_vboPos(NULL),
	m_vboNormal(NULL),
	m_vboColor(NULL)
{
	m_nameVS = "ShaderMatCustom_vs";
	m_nameFS = "ShaderMatCustom_fs";
	m_nameGS = "ShaderMatCustom_gs";

	std::string glxvert(GLSLShader::defines_gl());
	// get choose GL defines (2 or 3)
	// ans compile shaders
	if (m_with_eyepos)
		glxvert.append("#define WITH_EYEPOSITION");
	glxvert.append(vertexShaderText);

//	std::string glxgeom = GLSLShader::defines_Geom("triangles", "triangle_strip", 3);
//	glxgeom.append(geometryShaderText);

	std::string glxfrag(GLSLShader::defines_gl());
	// Use double sided lighting if set
	if (doubleSided)
		glxfrag.append("#define DOUBLE_SIDED\n");
	glxfrag.append(fragmentShaderText);

	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str());
//	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str(), glxgeom.c_str(), GL_TRIANGLES, GL_TRIANGLE_STRIP, 3);

	// and get and fill uniforms
	getLocations();

	Geom::Matrix44f id;
	id.identity();
	setTransformation(id);

	sendParams();
}

void ShaderMatCustom::getLocations()
{
	bind();
	*m_unif_ambiant   = glGetUniformLocation(this->program_handler(), "materialAmbient");
	*m_unif_diffuse   = glGetUniformLocation(this->program_handler(), "materialDiffuse");
	*m_unif_specular  = glGetUniformLocation(this->program_handler(), "materialSpecular");
	*m_unif_shininess = glGetUniformLocation(this->program_handler(), "shininess");
	*m_unif_lightPos  = glGetUniformLocation(this->program_handler(), "lightPosition");
	if (m_with_eyepos)
		*m_unif_eyePos  = glGetUniformLocation(this->program_handler(), "eyePosition");
	unbind();
}

void ShaderMatCustom::sendParams()
{
	bind();
	glUniform4fv(*m_unif_ambiant,  1, m_ambiant.data());
	glUniform4fv(*m_unif_diffuse,  1, m_diffuse.data());
	glUniform4fv(*m_unif_specular, 1, m_specular.data());
	glUniform1f(*m_unif_shininess,    m_shininess);
	glUniform3fv(*m_unif_lightPos, 1, m_lightPos.data());
	if (m_with_eyepos)
		glUniform3fv(*m_unif_eyePos, 1, m_eyePos.data());
	unbind();
}

void ShaderMatCustom::setAmbiant(const Geom::Vec4f& ambiant)
{
	bind();
	glUniform4fv(*m_unif_ambiant,1, ambiant.data());
	m_ambiant = ambiant;
	unbind();
}

void ShaderMatCustom::setDiffuse(const Geom::Vec4f& diffuse)
{
	bind();
	glUniform4fv(*m_unif_diffuse,1, diffuse.data());
	m_diffuse = diffuse;
	unbind();
}

void ShaderMatCustom::setSpecular(const Geom::Vec4f& specular)
{
	bind();
	glUniform4fv(*m_unif_specular,1,specular.data());
	m_specular = specular;
	unbind();
}

void ShaderMatCustom::setShininess(float shininess)
{
	bind();
	glUniform1f (*m_unif_shininess, shininess);
	m_shininess = shininess;
	unbind();
}

void ShaderMatCustom::setLightPosition(const Geom::Vec3f& lightPos)
{
	bind();
	glUniform3fv(*m_unif_lightPos,1,lightPos.data());
	m_lightPos = lightPos;
	unbind();
}

void ShaderMatCustom::setEyePosition(const Geom::Vec3f& eyePos)
{
	if (m_with_eyepos)
	{
		bind();
		glUniform3fv(*m_unif_eyePos,1,eyePos.data());
		m_eyePos = eyePos;
		unbind();
	}
}

void ShaderMatCustom::setParams(const Geom::Vec4f& ambiant, const Geom::Vec4f& diffuse, const Geom::Vec4f& specular, float shininess, const Geom::Vec3f& lightPos)
{
	m_ambiant = ambiant;
	m_diffuse = diffuse;
	m_specular = specular;
	m_shininess = shininess;
	m_lightPos = lightPos;
	sendParams();
}

unsigned int ShaderMatCustom::setAttributeColor(CGoGN::Utils::VBO* vbo)
{
	m_vboColor = vbo;
	if (!m_with_color)
	{
		m_with_color=true;
		// set the define and recompile shader
		std::string gl3vert(GLSLShader::defines_gl());
		gl3vert.append("#define WITH_COLOR 1\n");
		gl3vert.append(vertexShaderText);
		std::string gl3frag(GLSLShader::defines_gl());
		gl3frag.append("#define WITH_COLOR 1\n");
		gl3frag.append(fragmentShaderText);
		loadShadersFromMemory(gl3vert.c_str(), gl3frag.c_str());

		// and treat uniforms
		getLocations();
		sendParams();
	}
	// bind th VA with WBO
	bind();
	unsigned int id = bindVA_VBO("VertexColor", vbo);
	unbind();
	return id;
}

void ShaderMatCustom::unsetAttributeColor()
{
	m_vboColor = NULL;
	if (m_with_color)
	{
		m_with_color = false;
		// unbind the VA
		bind();
		unbindVA("VertexColor");
		unbind();
		// recompile shader
		std::string gl3vert(GLSLShader::defines_gl());
		gl3vert.append(vertexShaderText);
		std::string gl3frag(GLSLShader::defines_gl());
		gl3frag.append(fragmentShaderText);
		loadShadersFromMemory(gl3vert.c_str(), gl3frag.c_str());
		// and treat uniforms
		getLocations();
		sendParams();
	}
}

void ShaderMatCustom::restoreUniformsAttribs()
{
	getLocations();
	sendParams();

	bind();
	bindVA_VBO("VertexPosition", m_vboPos);
	bindVA_VBO("VertexNormal", m_vboNormal);
	if (m_vboColor)
		bindVA_VBO("VertexColor", m_vboColor);

	unbind();
}

unsigned int ShaderMatCustom::setAttributePosition(CGoGN::Utils::VBO* vbo)
{
	m_vboPos = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexPosition", vbo);
	unbind();
	return id;
}

unsigned int ShaderMatCustom::setAttributeNormal(CGoGN::Utils::VBO* vbo)
{
	m_vboNormal = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexNormal", vbo);
	unbind();
	return id;
}

void ShaderMatCustom::setTransformation(Geom::Matrix44f t)
{
	bind();
	CGoGNGLuint m_transf;
	*m_transf  = glGetUniformLocation(program_handler(),"TransformationMatrix");
	glUniformMatrix4fv(*m_transf, 1, false, &t(0,0));
	unbind();
}
