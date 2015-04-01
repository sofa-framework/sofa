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

#ifndef _CGoGN_CLIPPINGPRESETSANIMATED_H_
#define _CGoGN_CLIPPINGPRESETSANIMATED_H_

#include "Utils/clippingPresets.h"
#include "Utils/clippingShader.h"
#include "Geometry/vector_gen.h"
#include <cmath>
#include <vector>
#include <cstdlib>
#include <time.h>

#include "Utils/dll.h"

namespace CGoGN
{

namespace Utils
{


/***********************************************
 *
 * 		Base Class
 *
 ***********************************************/


class CGoGN_UTILS_API ClippingPresetAnimated : public ClippingPreset
{

public :

	/// constructor
	ClippingPresetAnimated():
		m_attachedClippingShader (NULL),
		m_animationOneStepIncrement (0.01f),
		m_animationSpeedFactor (1.0f),
		m_animParam (0.0f)
		{}

public :

	/*
	 * sets the animation speed factor (can be negative for reverse animation)
	 * @param speedFactor speed factor
	 */
	inline void setAnimationSpeedFactor(float speedFactor) { m_animationSpeedFactor = speedFactor; }

	/// gets the animation speed factor
	inline float getAnimationSpeedFactor() { return m_animationSpeedFactor; }

protected :

	/// attached Clipping Shader, used for updating clipping shapes params
	ClippingShader *m_attachedClippingShader;

	/// stored ids used for updating clipping shapes params
	std::vector<unsigned int> m_planesIds;

	/// stored ids used for updating clipping shapes params
	std::vector<unsigned int> m_spheresIds;

	/// one step increment of animation parameter
	float m_animationOneStepIncrement;

	/// speed factor of the animation
	float m_animationSpeedFactor;

	/// animation parameter value (usually between 0.0 and 1.0)
	float m_animParam;

public :

	/**
	 * applies the preset on a clipping shader and store its pointer
	 * @param clipShader pointer to the clipping shader object
	 * @param planesIds returns the new added planes ids
	 * @param spheresIds returns the new added spheres ids
	 * @warning clipShader, planesIds and spheresIds must not be NULL, otherwise the function does nothing
	 * @warning clipShader must remain valid as long as step is being used and apply has not been called again
	 */
	void apply(ClippingShader* clipShader, std::vector<unsigned int> *planesIds, std::vector<unsigned int> *spheresIds);

	/**
	 * runs the animation some steps further
	 * @param amount amount of steps the animation should do
	 */
	virtual void step(unsigned int amount) = 0;

};


/***********************************************
 *
 * 		Derived Classes
 *
 ***********************************************/


class CGoGN_UTILS_API ClippingPresetAnimatedDualPlanes : public ClippingPresetAnimated
{

public :

	/**
	 * constructor
	 * @param centerStart center between planes at the start of the animation
	 * @param centerEnd center between planes at the end of the animation
	 * @param size distance between planes
	 * @param axis axis on which planes are aligned (0 for x, 1 for y, 2 for z)
	 * @param facing true means having facing planes
	 * @param zigzag true to enable zigzag mode
	 */
	ClippingPresetAnimatedDualPlanes(
			Geom::Vec3f centerStart, Geom::Vec3f centerEnd, float size, int axis, bool facing, bool zigzag);

	/**
	 * runs the animation some steps further
	 * @param amount amount of steps the animation should do
	 */
	void step(unsigned int amount);

private :

	/// direction vector
	Geom::Vec3f m_dirVec;

	/// start point of the animation
	Geom::Vec3f m_startPoint;

	/// end point of the animation
	Geom::Vec3f m_endPoint;

	/// zigzag mode ?
	bool m_zigzag;

};

class CGoGN_UTILS_API ClippingPresetAnimatedRotatingPlane : public ClippingPresetAnimated
{

public :

	/**
	 * constructor
	 * @param center center of plane
	 * @param axis axis on which plane is aligned (0 for x, 1 for y, 2 for z)
	 */
	ClippingPresetAnimatedRotatingPlane(Geom::Vec3f center, int axis);

	/**
	 * runs the animation some steps further
	 * @param amount amount of steps the animation should do
	 */
	void step(unsigned int amount);

private :

	/// axis of plane
	int m_axis;

};

class CGoGN_UTILS_API ClippingPresetAnimatedScaledSphere : public ClippingPresetAnimated
{

public :

	/**
	 * constructor
	 * @param center center of sphere
	 * @param radiusStart radius of the sphere at the beginning of the animation
	 * @param radiusEnd radius of the sphere at the end of the animation
	 * @param zigzag true to enable zigzag mode
	 */
	ClippingPresetAnimatedScaledSphere(Geom::Vec3f center, float radiusStart, float radiusEnd, bool zigzag);

	/**
	 * runs the animation some steps further
	 * @param amount amount of steps the animation should do
	 */
	void step(unsigned int amount);

private :

	/// start radius of the animation
	float m_startRadius;

	/// end radius of the animation
	float m_endRadius;

	/// zigzag mode ?
	bool m_zigzag;

};


class CGoGN_UTILS_API ClippingPresetAnimatedSpheresCubeCollision : public ClippingPresetAnimated
{

public :

	/**
	 * constructor
	 * @param center center of cube
	 * @param size size of cube
	 * @param spheresCount spheres count
	 * @param radius radius of spheres
	 */
	ClippingPresetAnimatedSpheresCubeCollision(Geom::Vec3f center, float size, int spheresCount, float radius);

	/**
	 * runs the animation some steps further
	 * @param amount amount of steps the animation should do
	 */
	void step(unsigned int amount);

private :

	/// center of cube
	Geom::Vec3f m_cubeCenter;

	/// size of cube
	float m_cubeSize;

	/// direction vector of spheres
	std::vector<Geom::Vec3f> m_spheresDirections;

};


} // namespace Utils

} // namespace CGoGN

#endif
