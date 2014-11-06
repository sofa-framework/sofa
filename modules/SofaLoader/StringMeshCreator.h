/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_LOADER_StringMeshCreator_H
#define SOFA_COMPONENT_LOADER_StringMeshCreator_H

#include <sofa/core/loader/MeshLoader.h>
#include <sofa/SofaCommon.h>

#include <sofa/helper/SVector.h>
namespace sofa
{

namespace component
{

namespace loader
{


/** Procedurally creates a string.
  The coordinates range from (0,0,0) to (1,0,0). They can be translated, rotated and scaled using the corresponding attributes of the parent class.

  @author François Faure, 2012
  */
class SOFA_LOADER_API StringMeshCreator : public sofa::core::loader::MeshLoader
{
public:

    SOFA_CLASS(StringMeshCreator,sofa::core::loader::MeshLoader);
    virtual std::string type() { return "This object is procedurally created"; }
    virtual bool canLoad() { return true; }
    virtual bool load(); ///< create the string

    template <class T>
    static bool canCreate ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg ) { return BaseLoader::canCreate (obj, context, arg); }

    Data< unsigned > resolution;  ///< Number of vertices (more than 1)

protected:
    StringMeshCreator();
};




} // namespace loader

} // namespace component

} // namespace sofa

#endif
