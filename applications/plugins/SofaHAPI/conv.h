/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFAHAPI_CONV_H
#define SOFAHAPI_CONV_H

#include <SofaHAPI/config.h>

//HAPI include
#include <HAPI/HAPITypes.h>

#include <sofa/defaulttype/SolidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

	namespace component
	{

		using sofa::defaulttype::Vec3d;
		using sofa::defaulttype::Quat;

		extern inline sofa::defaulttype::Vec3f conv(const H3DUtil::Vec3f& v)
		{
			return sofa::defaulttype::Vec3f(v.x,v.y,v.z);
		}

		extern inline sofa::defaulttype::Vec3d conv(const H3DUtil::Vec3d& v)
		{
			return sofa::defaulttype::Vec3d(v.x,v.y,v.z);
		}

		extern inline H3DUtil::Vec3f conv(const sofa::defaulttype::Vec3f& v)
		{
			return H3DUtil::Vec3f(v[0],v[1],v[2]);
		}

		extern inline H3DUtil::Vec3d conv(const sofa::defaulttype::Vec3d& v)
		{
			return H3DUtil::Vec3d(v[0],v[1],v[2]);
		}

		extern inline sofa::defaulttype::Quat conv(const HAPI::Rotation& r)
		{
			return sofa::defaulttype::Quat(conv(r.axis), r.angle);
		}

	} // namespace SofaHAPI
}
#endif // SOFAHAPI_CONV_H
