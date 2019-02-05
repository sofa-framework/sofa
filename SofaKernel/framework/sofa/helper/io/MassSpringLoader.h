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
#ifndef SOFA_HELPER_IO_XSPLOADER_H
#define SOFA_HELPER_IO_XSPLOADER_H

#include <cstddef>                  /// For size_t
#include <string>                   /// For std::string
#include <sofa/helper/helper.h>     /// For SOFA_HELPER_API

namespace sofa {
namespace core {
namespace objectmodel {
class Base;
}
}
}


namespace sofa
{

namespace helper
{

namespace io
{

/// @brief Inherit this class to load data from a Xsp file.
class SOFA_HELPER_API XspLoaderDataHook
{
public:
    virtual void setNumMasses(size_t /*n*/) {}
    virtual void setNumSprings(size_t /*n*/) {}
    virtual void addMass(SReal /*px*/, SReal /*py*/, SReal /*pz*/, SReal /*vx*/, SReal /*vy*/, SReal /*vz*/, SReal /*mass*/, SReal /*elastic*/, bool /*fixed*/, bool /*surface*/) {}
    virtual void addSpring(int /*m1*/, int /*m2*/, SReal /*ks*/, SReal /*kd*/, SReal /*initpos*/) {}
    virtual void addVectorSpring(int m1, int m2, SReal ks, SReal kd, SReal initpos, SReal /*restx*/, SReal /*resty*/, SReal /*restz*/) { addSpring(m1, m2, ks, kd, initpos); }
    virtual void setGravity(SReal /*gx*/, SReal /*gy*/, SReal /*gz*/) {}
    virtual void setViscosity(SReal /*visc*/) {}
};

class SOFA_HELPER_API XspLoader
{
public:
    static bool Load(const std::string& filename,
                     XspLoaderDataHook& data,
                     sofa::core::objectmodel::Base* context);
};

} // namespace io

} // namespace helper

} // namespace sofa

#endif
