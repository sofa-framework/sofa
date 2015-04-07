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

#ifndef _CGoGN_CLIPPINGPRESETS_H_
#define _CGoGN_CLIPPINGPRESETS_H_

#include "Utils/clippingShader.h"
#include "Geometry/vector_gen.h"
#include <vector>
#include <cmath>

#include "Utils/dll.h"

namespace CGoGN
{

namespace Utils
{

class CGoGN_UTILS_API ClippingPreset
{


	/***********************************************
	 *
	 * 		Constructors
	 *
	 ***********************************************/

public :

	virtual ~ClippingPreset() {}

	/// public static constructor
	static ClippingPreset* CreateEmptyPreset();

	/**
	 * public static constructor
	 * @param center center between planes
	 * @param size distance between planes
	 * @param axis axis on which planes are aligned (0 for x, 1 for y, 2 for z)
	 * @param facing true means having facing planes
	 */
	static ClippingPreset* CreateDualPlanesPreset(Geom::Vec3f center, float size, int axis, bool facing);

	/**
	 * public static constructor
	 * @param center center between planes
	 * @param size distance between planes
	 * @param facing true means having facing planes
	 */
	static ClippingPreset* CreateCubePreset(Geom::Vec3f center, float size, bool facing);

	/**
	 * public static constructor
	 * @param center center of the tube
	 * @param size tube diameter
	 * @param axis axis of the tube (0 for x, 1 for y, 2 for z)
	 * @param precision planes count used to build tube
	 * @param facing true means an outer tube, false an inner tube
	 */
	static ClippingPreset* CreateTubePreset(Geom::Vec3f center, float size, int axis, int precision, bool facing);

	/**
	 * public static constructor
	 * @param center center of molecule
	 * @param size molecule size
	 * @param atomsRadiuses radiuses of atoms
	 * @param orClipping set it to true for OR clipping mode
	 */
	static ClippingPreset* CreateMoleculePreset(Geom::Vec3f center, float size, float atomsRadiuses, bool orClipping);

protected :

	/// protected constructor (used by public static constructors or child class)
	ClippingPreset();

	/***********************************************
	 *
	 * 		Preset settings
	 *
	 ***********************************************/

public :

	/**
	 * adds a clip plane to the preset
	 * @param normal clip plane normal
	 * @param origin clip plane origin
	 */
	void addClipPlane(Geom::Vec3f normal, Geom::Vec3f origin);

	/**
	 * adds a clip sphere to the preset
	 * @param center clip sphere center
	 * @param radius clip sphere radius
	 */
	void addClipSphere(Geom::Vec3f center, float radius);

	/**
	 * sets the clipping mode
	 * @param clipMode clipping mode
	 */
	void setClippingMode(ClippingShader::clippingMode clipMode);

private :

	/// clip planes structure
	struct clipPlane
	{
		Geom::Vec3f normal, origin;
	};

	/// clip planes array
	std::vector<clipPlane> m_clipPlanes;

	/// clip spheres structure
	struct clipSphere
	{
		Geom::Vec3f center;
		float radius;
	};

	/// clip spheres array
	std::vector<clipSphere> m_clipSpheres;

	/// clipping mode
	ClippingShader::clippingMode m_clipMode;


	/***********************************************
	 *
	 * 		Preset application
	 *
	 ***********************************************/

public :

	/**
	 * applies the preset on a clipping shader
	 * @param clipShader pointer to the clipping shader object
	 * @param planesIds returns the new added planes ids
	 * @param spheresIds returns the new added spheres ids
	 * @warning clipShader, planesIds and spheresIds must not be NULL, otherwise the function does nothing
	 */
	virtual void apply(ClippingShader* clipShader, std::vector<unsigned int> *planesIds, std::vector<unsigned int> *spheresIds);


};

} // namespace Utils

} // namespace CGoGN

#endif
