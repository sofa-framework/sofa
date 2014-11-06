/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_IO_MASSSPRINGLOADER_H
#define SOFA_HELPER_IO_MASSSPRINGLOADER_H
#include <sofa/defaulttype/Vec.h>
#include <sofa/SofaFramework.h>
namespace sofa
{

namespace helper
{

namespace io
{

class SOFA_HELPER_API MassSpringLoader
{
public:
    virtual ~MassSpringLoader() {}
    bool load(const char *filename);
    virtual void setNumMasses(int /*n*/) {}
    virtual void setNumSprings(int /*n*/) {}
    virtual void addMass(SReal /*px*/, SReal /*py*/, SReal /*pz*/, SReal /*vx*/, SReal /*vy*/, SReal /*vz*/, SReal /*mass*/, SReal /*elastic*/, bool /*fixed*/, bool /*surface*/) {}
    virtual void addSpring(int /*m1*/, int /*m2*/, SReal /*ks*/, SReal /*kd*/, SReal /*initpos*/) {}
    virtual void addVectorSpring(int m1, int m2, SReal ks, SReal kd, SReal initpos, SReal /*restx*/, SReal /*resty*/, SReal /*restz*/) { addSpring(m1, m2, ks, kd, initpos); }
    virtual void setGravity(SReal /*gx*/, SReal /*gy*/, SReal /*gz*/) {}
    virtual void setViscosity(SReal /*visc*/) {}
};

} // namespace io

} // namespace helper

} // namespace sofa

#endif
