/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
/*****************************************************************************
 * Contributors:
 *     - damien.marchal@univ-lille1.fr
 ****************************************************************************/
#ifndef SOFAGEOMETRY_CONSTANTS_H
#define SOFAGEOMETRY_CONSTANTS_H

#include <SofaGeometry/config.h>
#include <sofa/defaulttype/Vec.h>

namespace sofageometry
{
    /// within the sofageometry namespace our default vector type is Vec3d.
    using sofa::defaulttype::Vec3d ;
    using sofa::defaulttype::Vec2d ;
    using sofa::defaulttype::Vec1d ;

    class SOFAGEOMETRY_API Constants
    {
    public:
        static Vec3d Origin ;
        static Vec3d XAxis ;
        static Vec3d YAxis ;
        static Vec3d ZAxis ;
    };
}

#endif // SOFAGEOMETRY_CONSTANTS_H
