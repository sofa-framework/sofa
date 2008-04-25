/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
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
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_HELPER_IO_SPHERELOADER_H
#define SOFA_HELPER_IO_SPHERELOADER_H
#include <sofa/defaulttype/Vec.h>
namespace sofa
{

namespace helper
{

namespace io
{

class SphereLoader
{
public:
    typedef sofa::defaulttype::Vector3::value_type Real_Sofa;
    virtual ~SphereLoader() {}
    bool load(const char *filename);
    virtual void setNumSpheres(int /*n*/) {}
    virtual void addSphere(Real_Sofa /*px*/, Real_Sofa /*py*/, Real_Sofa /*pz*/, Real_Sofa /*r*/) {}
};

} // namespace io

} // namespace helper

} // namespace sofa

#endif
