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
#include "Utils/clippingPresetsAnimated.h"

#include <cmath>

namespace CGoGN
{

namespace Utils
{


/***********************************************
 *
 * 		Base Class
 *
 ***********************************************/


void ClippingPresetAnimated::apply(ClippingShader* clipShader, std::vector<unsigned int> *planesIds, std::vector<unsigned int> *spheresIds)
{
	if ( (clipShader == NULL) || (planesIds == NULL) || (spheresIds == NULL) )
		return;

	// Apply preset
	ClippingPreset::apply(clipShader, planesIds, spheresIds);

	// Store the clipping shader pointer
	m_attachedClippingShader = clipShader;

	// Store ids
	m_planesIds.resize(0);
	m_planesIds.insert(m_planesIds.begin(), planesIds->begin(), planesIds->end());
	m_spheresIds.resize(0);
	m_spheresIds.insert(m_spheresIds.begin(), spheresIds->begin(), spheresIds->end());
}


/***********************************************
 *
 * 		Derived Classes
 *
 ***********************************************/


ClippingPresetAnimatedDualPlanes::ClippingPresetAnimatedDualPlanes(
		Geom::Vec3f centerStart, Geom::Vec3f centerEnd, float size, int axis, bool facing, bool zigzag)
{
	// Store animation settings
	m_dirVec = centerEnd - centerStart;
	m_dirVec.normalize();
	m_startPoint = centerStart;
	m_endPoint = centerEnd;
	m_zigzag = zigzag;
	int usedAxis = axis;
	if ((usedAxis < 0) || (usedAxis > 2))
		usedAxis = 0;

	// Axis on which planes will be aligned
	Geom::Vec3f positDir (0.0f, 0.0f, 0.0f);
	positDir[usedAxis] = 1.0f;
	Geom::Vec3f negDir (0.0f, 0.0f, 0.0f);
	negDir[usedAxis] = -1.0f;

	// Facing of planes
	float side = 1.0f;
	if (facing)
		side = -1.0f;

	// Add planes to preset
	addClipPlane(positDir, centerStart + positDir*(size / 2.0f)*(side));
	addClipPlane(negDir, centerStart + negDir*(size / 2.0f)*(side));

	// Set clipping mode
	ClippingShader::clippingMode clipMode = ClippingShader::CLIPPING_MODE_AND;
	if (facing)
		clipMode = ClippingShader::CLIPPING_MODE_OR;
	setClippingMode(clipMode);
}

void ClippingPresetAnimatedDualPlanes::step(unsigned int amount)
{
	// Check if the animation has been stopped
	if (m_animationSpeedFactor == 0.0f)
		return;

	// Check the validity of planes or spheres ids
	if ( !m_attachedClippingShader->isClipPlaneIdValid(m_planesIds[0])
		|| !m_attachedClippingShader->isClipPlaneIdValid(m_planesIds[1]) )
	{
		CGoGNerr
		<< "ERROR -"
		<< "ClippingPresetAnimatedDualPlanes::step"
		<< " - Some planes or spheres ids are not valid anymore - Animation paused"
		<< CGoGNendl;
		m_animationSpeedFactor = 0.0f;
		return;
	}

	// Store the old center
	Geom::Vec3f oldCenter = (1.0f - m_animParam)*m_startPoint + m_animParam*m_endPoint;

	// Update animation parameter value
	m_animParam += (float)amount * m_animationOneStepIncrement * m_animationSpeedFactor;
	if (!m_zigzag)
	{
		while (m_animParam < 0.0f)
			m_animParam += 1.0f;
		while (m_animParam > 1.0f)
			m_animParam -= 1.0f;
	}
	else
	{
		while ( (m_animParam < 0.0f) || (m_animParam > 1.0f) )
		{
			if (m_animParam < 0.0f)
			{
				m_animParam = -m_animParam;
				m_animationOneStepIncrement = -m_animationOneStepIncrement;
			}
			else if (m_animParam > 1.0f)
			{
				m_animParam = 1.0f - (m_animParam - 1.0f);
				m_animationOneStepIncrement = -m_animationOneStepIncrement;
			}
		}
	}

	// Calculate new center
	Geom::Vec3f newCenter = (1.0f - m_animParam)*m_startPoint + m_animParam*m_endPoint;

	// Update clipping planes
	Geom::Vec3f plane1CurrPos = m_attachedClippingShader->getClipPlaneParamsOrigin(m_planesIds[0]);
	Geom::Vec3f plane2CurrPos = m_attachedClippingShader->getClipPlaneParamsOrigin(m_planesIds[1]);
	m_attachedClippingShader->setClipPlaneParamsOrigin(m_planesIds[0], plane1CurrPos + newCenter - oldCenter);
	m_attachedClippingShader->setClipPlaneParamsOrigin(m_planesIds[1], plane2CurrPos + newCenter - oldCenter);
}


ClippingPresetAnimatedRotatingPlane::ClippingPresetAnimatedRotatingPlane(Geom::Vec3f center, int axis)
{
	// Store animation settings
	m_axis = axis;
	if ((m_axis < 0) || (m_axis))
		m_axis = 0;

	// Axis on which planes will be aligned
	Geom::Vec3f normal (0.0f, 0.0f, 0.0f);
	normal[(m_axis + 1) % 3] = 1.0f;

	// Add plane to preset
	addClipPlane(normal, center);

	// Set clipping mode
	setClippingMode(ClippingShader::CLIPPING_MODE_AND);
}

void ClippingPresetAnimatedRotatingPlane::step(unsigned int amount)
{
	// Check if the animation has been stopped
	if (m_animationSpeedFactor == 0.0f)
		return;

	// Check the validity of planes or spheres ids
	if (!m_attachedClippingShader->isClipPlaneIdValid(m_planesIds[0]))
	{
		CGoGNerr
		<< "ERROR -"
		<< "ClippingPresetAnimatedRotatingPlane::step"
		<< " - Some planes or spheres ids are not valid anymore - Animation paused"
		<< CGoGNendl;
		m_animationSpeedFactor = 0.0f;
		return;
	}

	// Update animation parameter value
	m_animParam += (float)amount * m_animationOneStepIncrement * m_animationSpeedFactor;
	while (m_animParam < 0.0f)
		m_animParam += 1.0f;
	while (m_animParam > 1.0f)
		m_animParam -= 1.0f;

	// Calculate new normal
	Geom::Vec3f planeNormal = m_attachedClippingShader->getClipPlaneParamsNormal(m_planesIds[0]);
	float angle = m_animParam*2.0*M_PI;
	planeNormal[(m_axis + 1) % 3] = cos(angle);
	planeNormal[(m_axis + 2) % 3] = sin(angle);
	m_attachedClippingShader->setClipPlaneParamsNormal(m_planesIds[0], planeNormal);
}

ClippingPresetAnimatedScaledSphere::ClippingPresetAnimatedScaledSphere(Geom::Vec3f center, float radiusStart, float radiusEnd, bool zigzag)
{
	// Store animation settings
	m_startRadius = radiusStart;
	m_endRadius = radiusEnd;
	std::cout << "Given Start Radius : " << radiusStart << std::endl;
	std::cout << "Actual Start Radius : " << m_startRadius << std::endl;
	std::cout << "Given End Radius : " << radiusEnd << std::endl;
	std::cout << "Actual End Radius : " << m_endRadius << std::endl;
	m_zigzag = zigzag;

	// Add sphere to preset
	addClipSphere(center, m_startRadius);

	// Set clipping mode
	setClippingMode(ClippingShader::CLIPPING_MODE_AND);
}

void ClippingPresetAnimatedScaledSphere::step(unsigned int amount)
{
	// Check if the animation has been stopped
	if (m_animationSpeedFactor == 0.0f)
		return;

	// Check the validity of planes or spheres ids
	if (!m_attachedClippingShader->isClipSphereIdValid(m_spheresIds[0]))
	{
		CGoGNerr
		<< "ERROR -"
		<< "ClippingPresetAnimatedScaledSphere::step"
		<< " - Some planes or spheres ids are not valid anymore - Animation paused"
		<< CGoGNendl;
		m_animationSpeedFactor = 0.0f;
		return;
	}

	// Update animation parameter value
	m_animParam += (float)amount * m_animationOneStepIncrement * m_animationSpeedFactor;
	if (!m_zigzag)
	{
		while (m_animParam < 0.0f)
			m_animParam += 1.0f;
		while (m_animParam > 1.0f)
			m_animParam -= 1.0f;
	}
	else
	{
		while ( (m_animParam < 0.0f) || (m_animParam > 1.0f) )
		{
			if (m_animParam < 0.0f)
			{
				m_animParam = -m_animParam;
				m_animationOneStepIncrement = -m_animationOneStepIncrement;
			}
			else if (m_animParam > 1.0f)
			{
				m_animParam = 1.0f - (m_animParam - 1.0f);
				m_animationOneStepIncrement = -m_animationOneStepIncrement;
			}
		}
	}

	// Calculate new radius
	float radius = (1.0f - m_animParam)*m_startRadius + m_animParam*m_endRadius;
	m_attachedClippingShader->setClipSphereParamsRadius(m_spheresIds[0], radius);
}

ClippingPresetAnimatedSpheresCubeCollision::ClippingPresetAnimatedSpheresCubeCollision(Geom::Vec3f center, float size, int spheresCount, float radius)
{
	// Store animation settings
	m_cubeCenter = center;
	m_cubeSize = size;
	int usedSpheresCount = spheresCount;
	if (usedSpheresCount < 1)
		usedSpheresCount = 1;

	// Add spheres to preset
	for (int i = 0; i < usedSpheresCount; i++)
		addClipSphere(m_cubeCenter, radius);

	// Store spheres random directions
	m_spheresDirections.resize(usedSpheresCount);
    srand(time(NULL));
	for (size_t i = 0; i < m_spheresDirections.size(); i++)
	{
		Geom::Vec3f dir ((rand() % 1000) - 500.0f, (rand() % 1000) - 500.0f, (rand() % 1000) - 500.0f);
		dir.normalize();
		m_spheresDirections[i] = dir;
	}

	// Set clipping mode
	setClippingMode(ClippingShader::CLIPPING_MODE_AND);
}

void ClippingPresetAnimatedSpheresCubeCollision::step(unsigned int amount)
{
	// Check if the animation has been stopped
	if (m_animationSpeedFactor == 0.0f)
		return;

	// Check the validity of planes or spheres ids
	for (size_t i = 0; i < m_spheresIds.size(); i++)
	{
		if (!m_attachedClippingShader->isClipSphereIdValid(m_spheresIds[i]))
		{
			CGoGNerr
			<< "ERROR -"
			<< "ClippingPresetAnimatedSpheresCubeCollision::step"
			<< " - Some planes or spheres ids are not valid anymore - Animation paused"
			<< CGoGNendl;
			m_animationSpeedFactor = 0.0f;
			return;
		}
	}

	// Update animation parameter value
	float dParam = (float)amount * m_animationOneStepIncrement * m_animationSpeedFactor;
	m_animParam += dParam;
	while (m_animParam < 0.0f)
		m_animParam += 1.0f;
	while (m_animParam > 1.0f)
		m_animParam -= 1.0f;

	// Calculate new center and detect collisions with cube faces
	for (size_t i = 0; i < m_spheresIds.size(); i++)
	{
		Geom::Vec3f oldCenter = m_attachedClippingShader->getClipSphereParamsCenter(m_spheresIds[i]);
		Geom::Vec3f newCenter = oldCenter + dParam*m_spheresDirections[i];
		m_attachedClippingShader->setClipSphereParamsCenter(m_spheresIds[i], newCenter);
		Geom::Vec3f posToCube = newCenter - m_cubeCenter;
		for (int j = 0; j < 3; j++)
		{
			if ( (posToCube[j] < -m_cubeSize*0.5f) || (posToCube[j] > m_cubeSize*0.5f) )
			{
				Geom::Vec3f cubeNormal (0.0f, 0.0f, 0.0f);
				if (posToCube[j] < 0.0f)
					cubeNormal[j] = 1.0f;
				else
					cubeNormal[j] = -1.0f;
				// Reflect
				if ( (m_spheresDirections[i] * cubeNormal) < 0.0f)
					m_spheresDirections[i] = (2.0f * ((cubeNormal*m_spheresDirections[i])*cubeNormal) - m_spheresDirections[i])*-1.0f;
			}
		}
	}
}

} // namespace Utils

} // namespace CGoGN
