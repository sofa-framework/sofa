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
#include "Utils/clippingShader.h"

namespace CGoGN
{

namespace Utils
{


/***********************************************
 *
 * 		Constructor / Destructor
 *
 ***********************************************/


ClippingShader::ClippingShader():
		// Initialize clipping shapes variables
		m_unif_clipPlanesEquations (0),
		m_unif_clipSpheresCentersAndRadiuses (0),

		// Initialize default global clipping variables
		m_hasClippingCodeBeenInserted (false),
		m_clipColorAttenuationFactor (1.0f),
		m_unif_clipColorAttenuationFactor (0),
		m_colorAttenuationMode (COLOR_ATTENUATION_MODE_LINEAR),
		m_clipMode (CLIPPING_MODE_AND)
{

}


/***********************************************
 *
 * 		Plane Clipping
 *
 ***********************************************/


unsigned int ClippingShader::addClipPlane()
{
	// Check if the clipping code has been inserted into shader
	if (errorRaiseClippingCodeNotInserted(!m_hasClippingCodeBeenInserted, "ClippingShader::addClipPlane"))
		return -1;

	// Shader name string
	std::string shaderName = m_nameVS + "/" + m_nameFS + "/" + m_nameGS;

	// Use a shader mutator
	ShaderMutator SM(shaderName, getVertexShaderSrc(), getFragmentShaderSrc());

	// Previous planes count
	int previousPlanesCount = getClipPlanesCount();

	// Modify the clip planes count constant in both shader
	if (errorRaiseShaderMutatorFailure(
			   (!SM.changeIntConstantValue(ShaderMutator::VERTEX_SHADER, "CLIP_PLANES_COUNT", previousPlanesCount + 1))
			|| (!SM.changeIntConstantValue(ShaderMutator::FRAGMENT_SHADER, "CLIP_PLANES_COUNT", previousPlanesCount + 1)),
			"ClippingShader::addClipPlane"))
		return -1;

	// Reload both shaders
	reloadVertexShaderFromMemory(SM.getModifiedVertexShaderSrc().c_str());
	reloadFragmentShaderFromMemory(SM.getModifiedFragmentShaderSrc().c_str());

	// Get new plane id
	unsigned int newPlaneId = getFreeClipPlaneId();

	// Resize planes arrays to the right size
	m_clipPlanes.resize((size_t)(previousPlanesCount + 1));
	if (newPlaneId >= m_clipPlanesIds.size())
		m_clipPlanesIds.resize((size_t)(newPlaneId + 1));
	m_clipPlanesEquations.resize(4*(size_t)(previousPlanesCount + 1), 0.0f);

	// Set new plane id
	m_clipPlanesIds[newPlaneId].used = true;
	m_clipPlanesIds[newPlaneId].index = previousPlanesCount;

	// Set default parameters values for the new plane
	Geom::Vec3f defaultNormal (0.0f, 0.0f, 1.0f);
	Geom::Vec3f defaultOrigin (0.0f, 0.0f, 0.0f);
	setClipPlaneParamsAll(newPlaneId, defaultNormal, defaultOrigin);

	// Recompile shaders (automatically calls updateClippingUniforms)
	recompile();

	return newPlaneId;
}

void ClippingShader::deleteClipPlane(unsigned int id)
{
	// Check if the given id is valid
	if (errorRaiseWrongId(id > (m_clipPlanesIds.size()), "ClippingShader::deleteClipPlane"))
		return;
	if (errorRaiseWrongId(!m_clipPlanesIds[id].used, "ClippingShader::deleteClipPlane"))
		return;

	// Check if the clipping code has been inserted into shader
	if (errorRaiseClippingCodeNotInserted(!m_hasClippingCodeBeenInserted, "ClippingShader::deleteClipPlane"))
		return;

	// Shader name string
	std::string shaderName = m_nameVS + "/" + m_nameFS + "/" + m_nameGS;

	// Use a shader mutator
	ShaderMutator SM(shaderName, getVertexShaderSrc(), getFragmentShaderSrc());

	// Previous planes count
	int previousPlanesCount = getClipPlanesCount();

	// Modify the clip planes count constant in both shader
	if (errorRaiseShaderMutatorFailure(
			   (!SM.changeIntConstantValue(ShaderMutator::VERTEX_SHADER, "CLIP_PLANES_COUNT", previousPlanesCount - 1))
			|| (!SM.changeIntConstantValue(ShaderMutator::FRAGMENT_SHADER, "CLIP_PLANES_COUNT", previousPlanesCount - 1)),
			"ClippingShader::deleteClipPlane"))
		return;

	// Reload both shaders
	reloadVertexShaderFromMemory(SM.getModifiedVertexShaderSrc().c_str());
	reloadFragmentShaderFromMemory(SM.getModifiedFragmentShaderSrc().c_str());

	// Rearrange planes arrays
	m_clipPlanes.erase(m_clipPlanes.begin() + m_clipPlanesIds[id].index);
	for (size_t i = 0; i < m_clipPlanesIds.size(); i++)
	{
		if (m_clipPlanesIds[i].index > m_clipPlanesIds[id].index)
			m_clipPlanesIds[i].index -= 1;
	}
	m_clipPlanesIds[id].used = false;
	m_clipPlanesEquations.erase(
			m_clipPlanesEquations.begin() + 4*m_clipPlanesIds[id].index,
			m_clipPlanesEquations.begin() + 4*m_clipPlanesIds[id].index + 4);

	// Recompile shaders (automatically calls updateClippingUniforms)
	recompile();
}

void ClippingShader::deleteAllClipPlanes()
{
	// Check if the clipping code has been inserted into shader
	if (errorRaiseClippingCodeNotInserted(!m_hasClippingCodeBeenInserted, "ClippingShader::deleteAllClipPlanes"))
		return;

	// Shader name string
	std::string shaderName = m_nameVS + "/" + m_nameFS + "/" + m_nameGS;

	// Use a shader mutator
	ShaderMutator SM(shaderName, getVertexShaderSrc(), getFragmentShaderSrc());

	// Modify the clip planes count constant in both shader
	if (errorRaiseShaderMutatorFailure(
			   (!SM.changeIntConstantValue(ShaderMutator::VERTEX_SHADER, "CLIP_PLANES_COUNT", 0))
			|| (!SM.changeIntConstantValue(ShaderMutator::FRAGMENT_SHADER, "CLIP_PLANES_COUNT", 0)),
			"ClippingShader::deleteAllClipPlanes"))
		return;

	// Reload both shaders
	reloadVertexShaderFromMemory(SM.getModifiedVertexShaderSrc().c_str());
	reloadFragmentShaderFromMemory(SM.getModifiedFragmentShaderSrc().c_str());

	// Rearrange planes arrays
	m_clipPlanes.resize(0);
	m_clipPlanesIds.resize(0);
	m_clipPlanesEquations.resize(0);

	// Recompile shaders (automatically calls updateClippingUniforms)
	recompile();
}

int ClippingShader::getClipPlanesCount()
{
	return (int)m_clipPlanes.size();
}

bool ClippingShader::isClipPlaneIdValid(unsigned int id)
{
	if (id >= m_clipPlanesIds.size())
		return false;
	if (!m_clipPlanesIds[id].used)
		return false;

	return true;
}

void ClippingShader::setClipPlaneParamsAll(unsigned int id, Geom::Vec3f normal, Geom::Vec3f origin)
{
	// Check if the given id is valid
	if (errorRaiseWrongId(!isClipPlaneIdValid(id), "ClippingShader::setClipPlaneParamsAll"))
		return;

	// Get the corresponding plane index
	int planeIndex = m_clipPlanesIds[id].index;

	// Normalize
	Geom::Vec3f normalNormalized = normal;
	normalNormalized.normalize();

	// Check if it is worth updating values !
	if ((normalNormalized == m_clipPlanes[planeIndex].normal)
			&& (origin == m_clipPlanes[planeIndex].origin))
		return;

	// Copy the given clipping plane parameters
	m_clipPlanes[planeIndex].normal = normalNormalized;
	m_clipPlanes[planeIndex].origin = origin;

	// Update the plane arrays
	updateClipPlaneUniformsArray(id);

	// Send again the whole planes equations array to shader
	sendClipPlanesEquationsUniform();
}

void ClippingShader::setClipPlaneParamsNormal(unsigned int id, Geom::Vec3f normal)
{
	// Check if the given id is valid
	if (errorRaiseWrongId(!isClipPlaneIdValid(id), "ClippingShader::setClipPlaneParamsFirstVec"))
		return;

	// Get the corresponding plane index
	int planeIndex = m_clipPlanesIds[id].index;

	// Normalize
	Geom::Vec3f normalNormalized = normal;
	normalNormalized.normalize();

	// Check if it is worth updating values !
	if (normalNormalized == m_clipPlanes[planeIndex].normal)
		return;

	// Copy the given clipping plane parameter
	m_clipPlanes[planeIndex].normal = normalNormalized;

	// Update the plane arrays
	updateClipPlaneUniformsArray(id);

	// Send again the whole planes equations array to shader
	sendClipPlanesEquationsUniform();
}

void ClippingShader::setClipPlaneParamsOrigin(unsigned int id, Geom::Vec3f origin)
{
	// Check if the given id is valid
	if (errorRaiseWrongId(!isClipPlaneIdValid(id), "ClippingShader::setClipPlaneParamsOrigin"))
		return;

	// Get the corresponding plane index
	int planeIndex = m_clipPlanesIds[id].index;

	// Check if it is worth updating values !
	if (origin == m_clipPlanes[planeIndex].origin)
		return;

	// Copy the given clipping plane parameter
	m_clipPlanes[planeIndex].origin = origin;

	// Update the plane arrays
	updateClipPlaneUniformsArray(id);

	// Send again the whole planes equations array to shader
	sendClipPlanesEquationsUniform();
}

Geom::Vec3f ClippingShader::getClipPlaneParamsNormal(unsigned int id)
{
	// Check if the given id is valid
	if (errorRaiseWrongId(id > (m_clipPlanesIds.size()), "ClippingShader::getClipPlaneParamsFirstVec"))
		return Geom::Vec3f(0.0f, 0.0f, 0.0f);
	if (errorRaiseWrongId(!m_clipPlanesIds[id].used, "ClippingShader::getClipPlaneParamsFirstVec"))
		return Geom::Vec3f(0.0f, 0.0f, 0.0f);

	// Get the corresponding plane index
	int planeIndex = m_clipPlanesIds[id].index;

	// Return the parameter
	return m_clipPlanes[planeIndex].normal;
}

Geom::Vec3f ClippingShader::getClipPlaneParamsOrigin(unsigned int id)
{
	// Check if the given id is valid
	if (errorRaiseWrongId(id > (m_clipPlanesIds.size()), "ClippingShader::getClipPlaneParamsOrigin"))
		return Geom::Vec3f(0.0f, 0.0f, 0.0f);
	if (errorRaiseWrongId(!m_clipPlanesIds[id].used, "ClippingShader::getClipPlaneParamsOrigin"))
		return Geom::Vec3f(0.0f, 0.0f, 0.0f);

	// Get the corresponding plane index
	int planeIndex = m_clipPlanesIds[id].index;

	// Return the parameter
	return m_clipPlanes[planeIndex].origin;
}

unsigned int ClippingShader::getFreeClipPlaneId()
{
	unsigned int freeId = 0;

	// Free id = either out of current ids vector size or no more used
	while (freeId < m_clipPlanesIds.size())
	{
		if (!m_clipPlanesIds[freeId].used)
			return freeId;
		else
			freeId++;
	}

	return freeId;
}

void ClippingShader::updateClipPlaneUniformsArray(unsigned int id)
{
	// Check if the given id is valid
	if (errorRaiseWrongId(id > (m_clipPlanesIds.size()), "ClippingShader::updateClipPlaneUniformsArray"))
		return;
	if (errorRaiseWrongId(!m_clipPlanesIds[id].used, "ClippingShader::updateClipPlaneUniformsArray"))
		return;

	// Get the corresponding plane index
	int planeIndex = m_clipPlanesIds[id].index;

	// Update the planes equations array
	float d = -(m_clipPlanes[planeIndex].normal * m_clipPlanes[planeIndex].origin);
	m_clipPlanesEquations[4*planeIndex + 0] = m_clipPlanes[planeIndex].normal[0];
	m_clipPlanesEquations[4*planeIndex + 1] = m_clipPlanes[planeIndex].normal[1];
	m_clipPlanesEquations[4*planeIndex + 2] = m_clipPlanes[planeIndex].normal[2];
	m_clipPlanesEquations[4*planeIndex + 3] = d;

}


/***********************************************
 *
 * 		Sphere Clipping
 *
 ***********************************************/


unsigned int ClippingShader::addClipSphere()
{
	// Check if the clipping code has been inserted into shader
	if (errorRaiseClippingCodeNotInserted(!m_hasClippingCodeBeenInserted, "ClippingShader::addClipSphere"))
		return -1;

	// Shader name string
	std::string shaderName = m_nameVS + "/" + m_nameFS + "/" + m_nameGS;

	// Use a shader mutator
	ShaderMutator SM(shaderName, getVertexShaderSrc(), getFragmentShaderSrc());

	// Previous spheres count
	int previousSpheresCount = getClipSpheresCount();

	// Modify the clip spheres count constant in both shader
	if (errorRaiseShaderMutatorFailure(
			   (!SM.changeIntConstantValue(ShaderMutator::VERTEX_SHADER, "CLIP_SPHERES_COUNT", previousSpheresCount + 1))
			|| (!SM.changeIntConstantValue(ShaderMutator::FRAGMENT_SHADER, "CLIP_SPHERES_COUNT", previousSpheresCount + 1)),
			"ClippingShader::addClipSphere"))
		return -1;

	// Reload both shaders
	reloadVertexShaderFromMemory(SM.getModifiedVertexShaderSrc().c_str());
	reloadFragmentShaderFromMemory(SM.getModifiedFragmentShaderSrc().c_str());

	// Get new sphere id
	unsigned int newSphereId = getFreeClipSphereId();

	// Resize spheres arrays to the right size
	m_clipSpheres.resize((size_t)(previousSpheresCount + 1));
	if (newSphereId >= m_clipSpheresIds.size())
		m_clipSpheresIds.resize((size_t)(newSphereId + 1));
	m_clipSpheresCentersAndRadiuses.resize(4*(size_t)(previousSpheresCount + 1), 0.0f);

	// Set new sphere id
	m_clipSpheresIds[newSphereId].used = true;
	m_clipSpheresIds[newSphereId].index = previousSpheresCount;

	// Set default parameters values for the new sphere
	Geom::Vec3f defaultCenter (0.0f, 0.0f, 0.0f);
	float defaultRadius = 10.0f;
	setClipSphereParamsAll(newSphereId, defaultCenter, defaultRadius);

	// Recompile shaders (automatically calls updateClippingUniforms)
	recompile();

	return newSphereId;
}

void ClippingShader::deleteClipSphere(unsigned int id)
{
	// Check if the given id is valid
	if (errorRaiseWrongId(id > (m_clipSpheresIds.size()), "ClippingShader::deleteClipSphere"))
		return;
	if (errorRaiseWrongId(!m_clipSpheresIds[id].used, "ClippingShader::deleteClipSphere"))
		return;

	// Check if the clipping code has been inserted into shader
	if (errorRaiseClippingCodeNotInserted(!m_hasClippingCodeBeenInserted, "ClippingShader::deleteClipSphere"))
		return;

	// Shader name string
	std::string shaderName = m_nameVS + "/" + m_nameFS + "/" + m_nameGS;

	// Use a shader mutator
	ShaderMutator SM(shaderName, getVertexShaderSrc(), getFragmentShaderSrc());

	// Previous spheres count
	int previousSpheresCount = getClipSpheresCount();

	// Modify the clip spheres count constant in both shader
	if (errorRaiseShaderMutatorFailure(
			   (!SM.changeIntConstantValue(ShaderMutator::VERTEX_SHADER, "CLIP_SPHERES_COUNT", previousSpheresCount - 1))
			|| (!SM.changeIntConstantValue(ShaderMutator::FRAGMENT_SHADER, "CLIP_SPHERES_COUNT", previousSpheresCount - 1)),
			"ClippingShader::deleteClipSphere"))
		return;

	// Reload both shaders
	reloadVertexShaderFromMemory(SM.getModifiedVertexShaderSrc().c_str());
	reloadFragmentShaderFromMemory(SM.getModifiedFragmentShaderSrc().c_str());

	// Rearrange spheres arrays
	m_clipSpheres.erase(m_clipSpheres.begin() + m_clipSpheresIds[id].index);
	for (size_t i = 0; i < m_clipSpheresIds.size(); i++)
	{
		if (m_clipSpheresIds[i].index > m_clipSpheresIds[id].index)
			m_clipSpheresIds[i].index -= 1;
	}
	m_clipSpheresIds[id].used = false;
	m_clipSpheresCentersAndRadiuses.erase(
			m_clipSpheresCentersAndRadiuses.begin() + 4*m_clipSpheresIds[id].index,
			m_clipSpheresCentersAndRadiuses.begin() + 4*m_clipSpheresIds[id].index + 4);

	// Recompile shaders (automatically calls updateClippingUniforms)
	recompile();
}

void ClippingShader::deleteAllClipSpheres()
{
	// Check if the clipping code has been inserted into shader
	if (errorRaiseClippingCodeNotInserted(!m_hasClippingCodeBeenInserted, "ClippingShader::deleteAllClipSpheres"))
		return;

	// Shader name string
	std::string shaderName = m_nameVS + "/" + m_nameFS + "/" + m_nameGS;

	// Use a shader mutator
	ShaderMutator SM(shaderName, getVertexShaderSrc(), getFragmentShaderSrc());

	// Modify the clip spheres count constant in both shader
	if (errorRaiseShaderMutatorFailure(
			   (!SM.changeIntConstantValue(ShaderMutator::VERTEX_SHADER, "CLIP_SPHERES_COUNT", 0))
			|| (!SM.changeIntConstantValue(ShaderMutator::FRAGMENT_SHADER, "CLIP_SPHERES_COUNT", 0)),
			"ClippingShader::deleteAllClipSpheres"))
		return;

	// Reload both shaders
	reloadVertexShaderFromMemory(SM.getModifiedVertexShaderSrc().c_str());
	reloadFragmentShaderFromMemory(SM.getModifiedFragmentShaderSrc().c_str());

	// Rearrange spheres arrays
	m_clipSpheres.resize(0);
	m_clipSpheresIds.resize(0);
	m_clipSpheresCentersAndRadiuses.resize(0);

	// Recompile shaders (automatically calls updateClippingUniforms)
	recompile();
}

int ClippingShader::getClipSpheresCount()
{
	return (int)m_clipSpheres.size();
}

bool ClippingShader::isClipSphereIdValid(unsigned int id)
{
	if (id >= m_clipSpheresIds.size())
		return false;
	if (!m_clipSpheresIds[id].used)
		return false;

	return true;
}

void ClippingShader::setClipSphereParamsAll(unsigned int id, Geom::Vec3f center, float radius)
{
	// Check if the given id is valid
	if (errorRaiseWrongId(!isClipSphereIdValid(id), "ClippingShader::setClipSphereParamsAll"))
		return;

	// Get the corresponding sphere index
	int sphereIndex = m_clipSpheresIds[id].index;

	// Check if it is worth updating values !
	if ((center == m_clipSpheres[sphereIndex].center)
			&& (radius == m_clipSpheres[sphereIndex].radius))
		return;

	// Copy the given clipping sphere parameters
	m_clipSpheres[sphereIndex].center = center;
	m_clipSpheres[sphereIndex].radius = radius;

	// Update the sphere array
	updateClipSphereUniformsArray(id);

	// Send again the whole spheres centers and radiuses array to shader
	sendClipSpheresCentersAndRadiusesUniform();
}

void ClippingShader::setClipSphereParamsCenter(unsigned int id, Geom::Vec3f center)
{
	// Check if the given id is valid
	if (errorRaiseWrongId(!isClipSphereIdValid(id), "ClippingShader::setClipSphereParamsCenter"))
		return;

	// Get the corresponding sphere index
	int sphereIndex = m_clipSpheresIds[id].index;

	// Check if it is worth updating values !
	if (center == m_clipSpheres[sphereIndex].center)
		return;

	// Copy the given clipping sphere parameter
	m_clipSpheres[sphereIndex].center = center;

	// Update the sphere array
	updateClipSphereUniformsArray(id);

	// Send again the whole spheres centers and radiuses array to shader
	sendClipSpheresCentersAndRadiusesUniform();
}

void ClippingShader::setClipSphereParamsRadius(unsigned int id, float radius)
{
	// Check if the given id is valid
	if (errorRaiseWrongId(!isClipSphereIdValid(id), "ClippingShader::setClipSphereParamsRadius"))
		return;

	// Get the corresponding sphere index
	int sphereIndex = m_clipSpheresIds[id].index;

	// Check if it is worth updating values !
	if (radius == m_clipSpheres[sphereIndex].radius)
		return;

	// Copy the given clipping sphere parameter
	m_clipSpheres[sphereIndex].radius = radius;

	// Update the sphere array
	updateClipSphereUniformsArray(id);

	// Send again the whole spheres centers and radiuses array to shader
	sendClipSpheresCentersAndRadiusesUniform();
}

Geom::Vec3f ClippingShader::getClipSphereParamsCenter(unsigned int id)
{
	// Check if the given id is valid
	if (errorRaiseWrongId(id > (m_clipSpheresIds.size()), "ClippingShader::getClipSphereParamsCenter"))
		return Geom::Vec3f(0.0f, 0.0f, 0.0f);
	if (errorRaiseWrongId(!m_clipSpheresIds[id].used, "ClippingShader::getClipSphereParamsCenter"))
		return Geom::Vec3f(0.0f, 0.0f, 0.0f);

	// Get the corresponding sphere index
	int sphereIndex = m_clipSpheresIds[id].index;

	// Return the parameter
	return m_clipSpheres[sphereIndex].center;
}

float ClippingShader::getClipSphereParamsRadius(unsigned int id)
{
	// Check if the given id is valid
	if (errorRaiseWrongId(id > (m_clipSpheresIds.size()), "ClippingShader::getClipSphereParamsRadius"))
		return 0.0f;
	if (errorRaiseWrongId(!m_clipSpheresIds[id].used, "ClippingShader::getClipSphereParamsRadius"))
		return 0.0f;

	// Get the corresponding sphere index
	int sphereIndex = m_clipSpheresIds[id].index;

	// Return the parameter
	return m_clipSpheres[sphereIndex].radius;
}

unsigned int ClippingShader::getFreeClipSphereId()
{
	unsigned int freeId = 0;

	// Free id = either out of current ids vector size or no more used
	while (freeId < m_clipSpheresIds.size())
	{
		if (!m_clipSpheresIds[freeId].used)
			return freeId;
		else
			freeId++;
	}

	return freeId;
}

void ClippingShader::updateClipSphereUniformsArray(unsigned int id)
{
	// Check if the given id is valid
	if (errorRaiseWrongId(id > (m_clipSpheresIds.size()), "ClippingShader::updateClipSphereUniformsArray"))
		return;
	if (errorRaiseWrongId(!m_clipSpheresIds[id].used, "ClippingShader::updateClipSphereUniformsArray"))
		return;

	// Get the corresponding sphere index
	int sphereIndex = m_clipSpheresIds[id].index;

	// Update the spheres centers and radiuses array
	m_clipSpheresCentersAndRadiuses[4*sphereIndex + 0] = m_clipSpheres[sphereIndex].center[0];
	m_clipSpheresCentersAndRadiuses[4*sphereIndex + 1] = m_clipSpheres[sphereIndex].center[1];
	m_clipSpheresCentersAndRadiuses[4*sphereIndex + 2] = m_clipSpheres[sphereIndex].center[2];
	m_clipSpheresCentersAndRadiuses[4*sphereIndex + 3] = m_clipSpheres[sphereIndex].radius;

}


/***********************************************
 *
 * 		Global Clipping Stuff
 *
 ***********************************************/


bool ClippingShader::insertClippingCode()
{
	// Check if the code has not already been inserted
	if (errorRaiseClippingCodeAlreadyInserted(m_hasClippingCodeBeenInserted, "ClippingShader::insertClippingCode"))
		return false;

	// Check if the vertex and fragment sources are not empty
	if (errorRaiseShaderSourceIsEmpty((getVertexShaderSrc() == NULL), "ClippingShader::insertClippingCode", ShaderMutator::VERTEX_SHADER))
		return false;
	if (errorRaiseShaderSourceIsEmpty((getFragmentShaderSrc() == NULL), "ClippingShader::insertClippingCode", ShaderMutator::FRAGMENT_SHADER))
		return false;

	// Check if the shader does not use a geometry shader
	if (errorRaiseShaderUsesGeometryShader((getGeometryShaderSrc() != NULL), "ClippingShader::insertClippingCode"))
		return false;


	// Strings to insert in shader sources

	std::string VS_headInsertion =
	"\n"
	"#define CLIP_PLANES_COUNT 0\n"
	"#define CLIP_SPHERES_COUNT 0\n"
	"\n"
	"#define PLANE_CLIPPING_ENABLED (CLIP_PLANES_COUNT > 0)\n"
	"#define SPHERE_CLIPPING_ENABLED (CLIP_SPHERES_COUNT > 0)\n"
	"\n"
	"#define CLIPPING_ENABLED (PLANE_CLIPPING_ENABLED || SPHERE_CLIPPING_ENABLED)\n"
	"\n"
	"#if CLIPPING_ENABLED\n"
	"	VARYING_VERT vec3 clip_nonTransformedPos;\n"
	"#endif\n"
	"\n";

	std::string VS_mainEndInsertion =
	"\n"
	"	#if CLIPPING_ENABLED\n"
	"		// Pass the non transformed vertex position to the fragment shader for clipping\n"
	"		clip_nonTransformedPos = VertexPosition;\n"
	"	#endif\n";

	std::string FS_headInsertion =
	"\n"
	"#define CLIP_PLANES_COUNT 0\n"
	"#define CLIP_SPHERES_COUNT 0\n"
	"\n"
	"#define PLANE_CLIPPING_ENABLED (CLIP_PLANES_COUNT > 0)\n"
	"#define SPHERE_CLIPPING_ENABLED (CLIP_SPHERES_COUNT > 0)\n"
	"\n"
	"#define CLIPPING_ENABLED (PLANE_CLIPPING_ENABLED || SPHERE_CLIPPING_ENABLED)\n"
	"\n"
	"// In following clipping modes, pixels may be deleted :\n"
	"//  - OR : only after being matched with every object\n"
	"//  - AND : on the fly as soon as one object does not match\n"
	"#define CLIPPING_MODE_AND 0\n"
	"#define CLIPPING_MODE_OR 1\n"
	"#define CLIPPING_MODE 0\n"
	"\n"
	"#define CLIPPING_COLOR_ATTENUATION_MODE_LINEAR 0\n"
	"#define CLIPPING_COLOR_ATTENUATION_MODE_QUADRATIC 1\n"
	"#define CLIPPING_COLOR_ATTENUATION_MODE 0\n"
	"\n"
	"#if CLIPPING_ENABLED\n"
	"\n"
	"	#if PLANE_CLIPPING_ENABLED\n"
	"		uniform vec4 clip_clipPlanesEquations[CLIP_PLANES_COUNT];\n"
	"	#endif\n"
	"\n"
	"	#if SPHERE_CLIPPING_ENABLED\n"
	"		uniform vec4 clip_clipSpheresCentersAndRadiuses[CLIP_SPHERES_COUNT];\n"
	"	#endif\n"
	"\n"
	"	uniform float clip_clipColorAttenuationFactor;\n"
	"\n"
	"	VARYING_FRAG vec3 clip_nonTransformedPos;\n"
	"\n"
	"#endif\n"
	"\n"
	"#if CLIPPING_ENABLED\n"
	"\n"
	"	float clip_doClippingAndGetClippingDistance()\n"
	"	{\n"
	"		// Distance to the nearest clipping object\n"
	"		float minDistanceToClipping = -1.0;\n"
	"\n"
	"		// OR clipping mode needs a boolean to know if the pixel must be clipped or not\n"
	"		// By default set to true : one clipping object matched sets it to false\n"
	"		#if (CLIPPING_MODE == CLIPPING_MODE_OR)\n"
	"			bool discardPixel = true;\n"
	"		#endif\n"
	"\n"
	"		#if PLANE_CLIPPING_ENABLED\n"
	"\n"
	"			// Do clipping for each plane\n"
	"			for (int i = 0; i < CLIP_PLANES_COUNT; i++)\n"
	"			{\n"
	"				// Get the current plane equation\n"
	"				vec4 currClipPlane = clip_clipPlanesEquations[i];\n"
	"\n"
	"				// If the plane normal is zero, use a default normal vector (0.0, 0.0, 1.0)\n"
	"				float clipPlaneNormalLength = length(currClipPlane.xyz);\n"
	"				if (clipPlaneNormalLength == 0.0)\n"
	"				{\n"
	"					currClipPlane.z = 1.0;\n"
	"					clipPlaneNormalLength = 1.0;\n"
	"				}\n"
	"\n"
	"				// Signed distance between the point and the plane\n"
	"				float distanceToPlane = dot(clip_nonTransformedPos, currClipPlane.xyz);\n"
	"				distanceToPlane += currClipPlane.w;\n"
	"				distanceToPlane /= clipPlaneNormalLength;\n"
	"\n"
	"				// AND clipping mode discards at first unmatched clipping object\n"
	"				#if (CLIPPING_MODE == CLIPPING_MODE_AND)\n"
	"					if (distanceToPlane > 0.0)\n"
	"						discard;\n"
	"				#endif\n"
	"\n"
	"				// In OR clipping mode, one match = no pixel clipping\n"
	"				#if (CLIPPING_MODE == CLIPPING_MODE_OR)\n"
	"					if (distanceToPlane < 0.0)\n"
	"						discardPixel = false;\n"
	"				#endif\n"
	"\n"
	"				// Keep the distance to the nearest plane\n"
	"				if (minDistanceToClipping < 0.0)\n"
	"					minDistanceToClipping = abs(distanceToPlane);\n"
	"				else\n"
	"					minDistanceToClipping = min(minDistanceToClipping, abs(distanceToPlane));\n"
	"			}\n"
	"\n"
	"		#endif\n"
	"\n"
	"		#if SPHERE_CLIPPING_ENABLED\n"
	"\n"
	"			// Do clipping for each sphere\n"
	"			for (int i = 0; i < CLIP_SPHERES_COUNT; i++)\n"
	"			{\n"
	"				// Get the current sphere center and radius\n"
	"				vec3 currClipSphereCenter = clip_clipSpheresCentersAndRadiuses[i].xyz;\n"
	"				float currClipSphereRadius = clip_clipSpheresCentersAndRadiuses[i].w;\n"
	"\n"
	"				// Signed distance between the point and the sphere\n"
	"				float distanceToSphere = length(clip_nonTransformedPos - currClipSphereCenter);\n"
	"				distanceToSphere -= abs(currClipSphereRadius);\n"
	"\n"
	"				// If the sphere radius is negative, this inverses the clipping effect\n"
	"				distanceToSphere *= sign(currClipSphereRadius);\n"
	"\n"
	"				// AND clipping mode discards at first unmatched clipping object\n"
	"				#if (CLIPPING_MODE == CLIPPING_MODE_AND)\n"
	"					if (distanceToSphere > 0.0)\n"
	"						discard;\n"
	"				#endif\n"
	"\n"
	"				// In OR clipping mode, one match = no pixel clipping\n"
	"				#if (CLIPPING_MODE == CLIPPING_MODE_OR)\n"
	"					if (distanceToSphere < 0.0)\n"
	"						discardPixel = false;\n"
	"				#endif\n"
	"\n"
	"				// Keep the distance to the nearest sphere\n"
	"				if (minDistanceToClipping < 0.0)\n"
	"					minDistanceToClipping = abs(distanceToSphere);\n"
	"				else\n"
	"					minDistanceToClipping = min(minDistanceToClipping, abs(distanceToSphere));\n"
	"			}\n"
	"\n"
	"		#endif\n"
	"\n"
	"		// In OR clipping mode, the pixel may be deleted only after being matched with every object\n"
	"		#if (CLIPPING_MODE == CLIPPING_MODE_OR)\n"
	"			if (discardPixel)\n"
	"				discard;\n"
	"		#endif\n"
	"\n"
	"		return minDistanceToClipping;\n"
	"	}\n"
	"\n"
	"#endif\n"
	"\n";

	std::string FS_mainBeginInsertion =
	"\n"
	"	#if CLIPPING_ENABLED\n"
	"		// Apply clipping and get the clipping distance\n"
	"		float clip_minDistanceToClipping = clip_doClippingAndGetClippingDistance();\n"
	"	#endif\n";

	std::string FS_mainEndInsertion =
	"\n"
	"	#if CLIPPING_ENABLED\n"
	"		// Attenuate the final fragment color depending on its distance to clipping objects\n"
	"		float clip_colorAttenuation = clip_minDistanceToClipping * clip_clipColorAttenuationFactor;\n"
	"		#if (CLIPPING_COLOR_ATTENUATION_MODE == CLIPPING_COLOR_ATTENUATION_MODE_QUADRATIC)\n"
	"			clip_colorAttenuation *= clip_colorAttenuation;\n"
	"		#endif\n"
	"		FRAG_OUT.rgb /= (1.0 + clip_colorAttenuation);\n"
	"	#endif;\n";

	// Shader name string
	std::string shaderName = m_nameVS + "/" + m_nameFS + "/" + m_nameGS;

	// Use a shader mutator
	ShaderMutator SM(shaderName, getVertexShaderSrc(), getFragmentShaderSrc());

	// First check if the vertex shader contains the VertexPosition attribute
	if (errorRaiseVariableNotFoundInShader(!SM.containsVariableDeclaration(ShaderMutator::VERTEX_SHADER, "VertexPosition"), "ClippingShader::insertClippingCode", ShaderMutator::VERTEX_SHADER, "VertexPosition"))
		return false;

	// Modify vertex shader source code
	if (errorRaiseShaderMutatorFailure(
			   (!SM.insertCodeBeforeMainFunction(ShaderMutator::VERTEX_SHADER, VS_headInsertion))
			|| (!SM.insertCodeAtMainFunctionBeginning(ShaderMutator::VERTEX_SHADER, VS_mainEndInsertion)),
			"ClippingShader::insertClippingCode"))
		return false;

	// Modify fragment shader source code
	if (errorRaiseShaderMutatorFailure(
			  /* (!SM.setMinShadingLanguageVersion(ShaderMutator::FRAGMENT_SHADER, 120)) // Following code insertions need at least shading language 120 (GLSL arrays)
			||*/ (!SM.insertCodeBeforeMainFunction(ShaderMutator::FRAGMENT_SHADER, FS_headInsertion))
			|| (!SM.insertCodeAtMainFunctionBeginning(ShaderMutator::FRAGMENT_SHADER, FS_mainBeginInsertion))
			|| (!SM.insertCodeAtMainFunctionEnd(ShaderMutator::FRAGMENT_SHADER, FS_mainEndInsertion)),
			"ClippingShader::insertClippingCode"))
		return false;

	// Reload both shaders
	reloadVertexShaderFromMemory(SM.getModifiedVertexShaderSrc().c_str());
	reloadFragmentShaderFromMemory(SM.getModifiedFragmentShaderSrc().c_str());

	// Recompile shaders (automatically calls updateClippingUniforms)
	recompile();

	m_hasClippingCodeBeenInserted = true;

	return true;
}

void ClippingShader::setClipColorAttenuationFactorAbsolute(float colorAttenuationFactor)
{
	// Check if it is worth updating values !
	if (colorAttenuationFactor == m_clipColorAttenuationFactor)
		return;

	// Copy the given value
	m_clipColorAttenuationFactor = colorAttenuationFactor;

	// Send again the uniform to shader
	sendClipColorAttenuationFactorUniform();
}

void ClippingShader::setClipColorAttenuationFactorRelative(float size, float factor)
{
	// Compute the relative color attenuation factor
	float colAttFact;
	if (size != 0.0f)
		colAttFact = factor / size;
	else
		colAttFact = factor;

	// Set the resulting factor in absolute
	setClipColorAttenuationFactorAbsolute(colAttFact);
}

float ClippingShader::getClipColorAttenuationFactor()
{
	return m_clipColorAttenuationFactor;
}

void ClippingShader::setClipColorAttenuationMode(colorAttenuationMode colAttMode)
{
	// Check if it is worth updating values !
	if (colAttMode == m_colorAttenuationMode)
		return;

	// Check if the clipping code has been inserted into shader
	if (errorRaiseClippingCodeNotInserted(!m_hasClippingCodeBeenInserted, "ClippingShader::setClipColorAttenuationMode"))
		return;

	// Shader name string
	std::string shaderName = m_nameVS + "/" + m_nameFS + "/" + m_nameGS;

	// Copy the given value
	m_colorAttenuationMode = colAttMode;

	// Use a shader mutator
	ShaderMutator SM(shaderName, getVertexShaderSrc(), getFragmentShaderSrc());

	// Change color attenuation mode constant
	int newConstantValue;
	switch (colAttMode)
	{
		case COLOR_ATTENUATION_MODE_LINEAR :
			newConstantValue = 0;
			break;

		case COLOR_ATTENUATION_MODE_QUADRATIC :
			newConstantValue = 1;
			break;

		default :
			newConstantValue = 0;
			break;
	}
	if (errorRaiseShaderMutatorFailure(
			(!SM.changeIntConstantValue(ShaderMutator::FRAGMENT_SHADER, "CLIPPING_COLOR_ATTENUATION_MODE", newConstantValue)),
			"ClippingShader::setClipColorAttenuationMode"))
		return;

	// Reload modified shader
	reloadFragmentShaderFromMemory(SM.getModifiedFragmentShaderSrc().c_str());

	// Recompile shaders (automatically calls updateClippingUniforms)
	recompile();
}

ClippingShader::colorAttenuationMode ClippingShader::getClipColorAttenuationMode()
{
	return m_colorAttenuationMode;
}

void ClippingShader::setClipMode(clippingMode clipMode)
{
	// Check if it is worth updating values !
	if (clipMode == m_clipMode)
		return;

	// Check if the clipping code has been inserted into shader
	if (errorRaiseClippingCodeNotInserted(!m_hasClippingCodeBeenInserted, "ClippingShader::setClipMode"))
		return;

	// Shader name string
	std::string shaderName = m_nameVS + "/" + m_nameFS + "/" + m_nameGS;

	// Copy the given value
	m_clipMode = clipMode;

	// Use a shader mutator
	ShaderMutator SM(shaderName, getVertexShaderSrc(), getFragmentShaderSrc());

	// Change clipping mode constant
	int newConstantValue;
	switch (clipMode)
	{
		case CLIPPING_MODE_AND :
			newConstantValue = 0;
			break;

		case CLIPPING_MODE_OR :
			newConstantValue = 1;
			break;

		default :
			newConstantValue = 0;
			break;
	}
	if (errorRaiseShaderMutatorFailure(
			(!SM.changeIntConstantValue(ShaderMutator::FRAGMENT_SHADER, "CLIPPING_MODE", newConstantValue)),
			"ClippingShader::setClipMode"))
		return;

	// Reload modified shader
	reloadFragmentShaderFromMemory(SM.getModifiedFragmentShaderSrc().c_str());

	// Recompile shaders (automatically calls updateClippingUniforms)
	recompile();
}

ClippingShader::clippingMode ClippingShader::getClipMode()
{
	return m_clipMode;
}


/***********************************************
 *
 * 		Clipping Uniforms Handling
 *
 ***********************************************/


void ClippingShader::updateClippingUniforms()
{
	// Plane clipping uniforms
	if (getClipPlanesCount() > 0)
	{
		// Get uniform location
		m_unif_clipPlanesEquations = glGetUniformLocation(program_handler(), "clip_clipPlanesEquations");
		errorRaiseUniformNotFoundInShader((m_unif_clipPlanesEquations == -1), "ClippingShader::updateClippingUniforms", "clip_clipPlanesEquations");

		// Send again uniform value
		sendClipPlanesEquationsUniform();
	}

	// Sphere clipping uniforms
	if (getClipSpheresCount() > 0)
	{
		// Get uniform location
		m_unif_clipSpheresCentersAndRadiuses = glGetUniformLocation(program_handler(), "clip_clipSpheresCentersAndRadiuses");
		errorRaiseUniformNotFoundInShader((m_unif_clipSpheresCentersAndRadiuses == -1), "ClippingShader::updateClippingUniforms", "clip_clipSpheresCentersAndRadiuses");

		// Send again uniform value
		sendClipSpheresCentersAndRadiusesUniform();
	}

	// Global clipping uniforms
	if ((getClipPlanesCount() > 0) || (getClipSpheresCount() > 0))
	{
		// Get uniform location
		m_unif_clipColorAttenuationFactor = glGetUniformLocation(program_handler(), "clip_clipColorAttenuationFactor");
		errorRaiseUniformNotFoundInShader((m_unif_clipColorAttenuationFactor == -1), "ClippingShader::updateClippingUniforms", "clip_clipColorAttenuationFactor");

		// Send again uniform value
		sendClipColorAttenuationFactorUniform();
	}
}

void ClippingShader::sendClipPlanesEquationsUniform()
{
	bind();
	glUniform4fv(m_unif_clipPlanesEquations, getClipPlanesCount(), &m_clipPlanesEquations.front());
}

void ClippingShader::sendClipSpheresCentersAndRadiusesUniform()
{
	bind();
	glUniform4fv(m_unif_clipSpheresCentersAndRadiuses, getClipSpheresCount(), &m_clipSpheresCentersAndRadiuses.front());
}

void ClippingShader::sendClipColorAttenuationFactorUniform()
{
	bind();
	glUniform1f(m_unif_clipColorAttenuationFactor, m_clipColorAttenuationFactor);
}


/***********************************************
 *
 * 		Error Raising
 *
 ***********************************************/


bool ClippingShader::errorRaiseShaderMutatorFailure(bool condition, const std::string& location)
{
	if (condition)
	{
		CGoGNerr
		<< "ERROR - "
		<< location
		<< " - Shader Mutator failure"
		<< CGoGNendl;
	}

	return condition;
}

bool ClippingShader::errorRaiseShaderSourceIsEmpty(bool condition, const std::string& location, ShaderMutator::shaderSrcType shaderType)
{
	if (condition)
	{
		std::string shaderName;
		switch (shaderType)
		{
			case ShaderMutator::VERTEX_SHADER :
				shaderName = m_nameVS;
				break;

			case ShaderMutator::FRAGMENT_SHADER :
				shaderName = m_nameFS;
				break;

			case ShaderMutator::GEOMETRY_SHADER :
				shaderName = m_nameGS;
				break;

			default :
				shaderName = m_nameVS;
				break;
		}

		CGoGNerr
		<< "ERROR - "
		<< location
		<< " - Could not process shader "
		<< shaderName
		<< " source code : shader source is empty"
		<< CGoGNendl;
	}

	return condition;
}

bool ClippingShader::errorRaiseShaderUsesGeometryShader(bool condition, const std::string& location)
{
	if (condition)
	{
		std::string shaderName = m_nameVS + "/" + m_nameFS + "/" + m_nameGS;

		CGoGNerr
		<< "ERROR - "
		<< location
		<< " - Could not process shader "
		<< shaderName
		<< " source code : unable to add clipping to a shader which uses a geometry shader"
		<< CGoGNendl;
	}

	return condition;
}

bool ClippingShader::errorRaiseVariableNotFoundInShader(bool condition, const std::string& location, ShaderMutator::shaderSrcType shaderType, const std::string& varName)
{
	if (condition)
	{
		std::string shaderName;
		switch (shaderType)
		{
			case ShaderMutator::VERTEX_SHADER :
				shaderName = m_nameVS;
				break;

			case ShaderMutator::FRAGMENT_SHADER :
				shaderName = m_nameFS;
				break;

			case ShaderMutator::GEOMETRY_SHADER :
				shaderName = m_nameGS;
				break;

			default :
				shaderName = m_nameVS;
				break;
		}

		CGoGNerr
		<< "ERROR - "
		<< location
		<< " - Could not process shader "
		<< shaderName
		<< " source code : "
		<< varName
		<< " not found"
		<< CGoGNendl;
	}

	return condition;
}
bool ClippingShader::errorRaiseWrongId(bool condition, const std::string& location)
{
	if (condition)
	{
		CGoGNerr
		<< "ERROR - "
		<< location
		<< " - Given ID is not valid"
		<< CGoGNendl;
	}

	return condition;
}

bool ClippingShader::errorRaiseUniformNotFoundInShader(bool condition, const std::string& location, const std::string& uniformName)
{
	if (condition)
	{
		std::string shaderName = m_nameVS + "/" + m_nameFS + "/" + m_nameGS;

		CGoGNerr
		<< "ERROR - "
		<< location
		<< " - Uniform "
		<< uniformName
		<< " not found in shader "
		<< shaderName
		<< CGoGNendl;
	}

	return condition;
}

bool ClippingShader::errorRaiseClippingCodeAlreadyInserted(bool condition, const std::string& location)
{
	if (condition)
	{
		std::string shaderName = m_nameVS + "/" + m_nameFS + "/" + m_nameGS;

		CGoGNerr
		<< "ERROR - "
		<< location
		<< " - Clipping code has already been inserted into shader "
		<< shaderName
		<< CGoGNendl;
	}

	return condition;
}

bool ClippingShader::errorRaiseClippingCodeNotInserted(bool condition, const std::string& location)
{
	if (condition)
	{
		std::string shaderName = m_nameVS + "/" + m_nameFS + "/" + m_nameGS;

		CGoGNerr
		<< "ERROR - "
		<< location
		<< " - Clipping code must be inserted into shader "
		<< shaderName
		<< " before adding clipping objects"
		<< CGoGNendl;
	}

	return condition;
}

} // namespace Utils

} // namespace CGoGN
