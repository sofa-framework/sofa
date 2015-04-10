
#define CGoGN_UTILS_DLL_EXPORT 1

#include <string.h>
#include "Utils/Shaders/shaderCustom.h"


#include "shaderCustom.vert"
#include "shaderCustom.frag"
#include "shaderCustom.geom"


ShaderCustom::ShaderCustom()
{
	m_nameVS = "ShaderCustom_vs";
	m_nameFS = "ShaderCustom_fs";
	m_nameGS = "ShaderCustom_gs";

	std::string glxvert(GLSLShader::defines_gl());
	glxvert.append(vertexShaderText);

	std::string glxgeom = GLSLShader::defines_Geom("triangles", "triangle_strip", 3);
	glxgeom.append(geometryShaderText);

	std::string glxfrag(GLSLShader::defines_gl());
	glxfrag.append(fragmentShaderText);

	loadShadersFromMemory(glxvert.c_str(), glxfrag.c_str(), glxgeom.c_str(), GL_TRIANGLES, GL_TRIANGLE_STRIP, 3);

	bind();
	getLocations();
	unbind();

	//Default values
	m_explode = 1.0f;
	m_ambiant = Geom::Vec4f(0.05f, 0.05f, 0.1f, 0.0f);
	m_diffuse = Geom::Vec4f(0.1f, 1.0f, 0.1f, 0.0f);
	m_light_pos = Geom::Vec3f(10.0f, 10.0f, 1000.0f);

	Geom::Matrix44f id;
	id.identity();
	setTransformation(id);

	setParams(m_explode, m_ambiant, m_diffuse, m_light_pos);
}

void ShaderCustom::getLocations()
{
	*m_unif_explode  = glGetUniformLocation(program_handler(), "explode");
	*m_unif_ambiant  = glGetUniformLocation(program_handler(), "ambient");
	*m_unif_diffuse  = glGetUniformLocation(program_handler(), "diffuse");
	*m_unif_lightPos = glGetUniformLocation(program_handler(), "lightPosition");
}

unsigned int ShaderCustom::setAttributePosition(Utils::VBO* vbo)
{
	m_vboPos = vbo;
	bind();
	unsigned int id = bindVA_VBO("VertexPosition", vbo);
	unbind();
	return id;
}

void ShaderCustom::setParams(float expl, const Geom::Vec4f& ambiant, const Geom::Vec4f& diffuse, const Geom::Vec3f& lightPos)
{
	m_explode = expl;
	m_ambiant = ambiant;
	m_diffuse = diffuse;
	m_light_pos = lightPos;

	bind();

	glUniform1f(*m_unif_explode, expl);
	glUniform4fv(*m_unif_ambiant, 1, ambiant.data());
	glUniform4fv(*m_unif_diffuse, 1, diffuse.data());
	glUniform3fv(*m_unif_lightPos, 1, lightPos.data());

	unbind();
}

void ShaderCustom::setExplode(float explode)
{
	m_explode = explode;
	bind();
	glUniform1f(*m_unif_explode, explode);
	unbind();
}

void ShaderCustom::setAmbiant(const Geom::Vec4f& ambiant)
{
	m_ambiant = ambiant;
	bind();
	glUniform4fv(*m_unif_ambiant,1, ambiant.data());
	unbind();
}

void ShaderCustom::setDiffuse(const Geom::Vec4f& diffuse)
{
	m_diffuse = diffuse;
	bind();
	glUniform4fv(*m_unif_diffuse,1, diffuse.data());
	unbind();
}

void ShaderCustom::setLightPosition(const Geom::Vec3f& lp)
{
	m_light_pos = lp;
	bind();
	glUniform3fv(*m_unif_lightPos,1,lp.data());
	unbind();
}

void ShaderCustom::restoreUniformsAttribs()
{
	*m_unif_explode   = glGetUniformLocation(program_handler(),"explode");
	*m_unif_ambiant   = glGetUniformLocation(program_handler(),"ambient");
	*m_unif_diffuse   = glGetUniformLocation(program_handler(),"diffuse");
	*m_unif_lightPos =  glGetUniformLocation(program_handler(),"lightPosition");

	bind();

	glUniform1f (*m_unif_explode, m_explode);
	glUniform4fv(*m_unif_ambiant,  1, m_ambiant.data());
	glUniform4fv(*m_unif_diffuse,  1, m_diffuse.data());
	glUniform3fv(*m_unif_lightPos, 1, m_light_pos.data());

	bindVA_VBO("VertexPosition", m_vboPos);

	unbind();
}

void ShaderCustom::setTransformation(Geom::Matrix44f t)
{
	bind();
	CGoGNGLuint m_transf;
	*m_transf  = glGetUniformLocation(program_handler(),"TransformationMatrix");
	glUniformMatrix4fv(*m_transf, 1, false, &t(0,0));
	unbind();
}

