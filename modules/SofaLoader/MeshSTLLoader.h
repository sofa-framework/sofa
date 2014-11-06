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
#ifndef SOFA_COMPONENT_LOADER_MESHSTLLOADER_H
#define SOFA_COMPONENT_LOADER_MESHSTLLOADER_H

#include <sofa/core/loader/MeshLoader.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/SofaCommon.h>

namespace sofa
{

namespace component
{

namespace loader
{

#ifdef WIN32
#define strcasecmp stricmp
#endif

// Format doc: http://en.wikipedia.org/wiki/STL_(file_format)
class SOFA_LOADER_API MeshSTLLoader : public sofa::core::loader::MeshLoader
{
public:
    SOFA_CLASS(MeshSTLLoader,sofa::core::loader::MeshLoader);
protected:
    MeshSTLLoader();
public:
    virtual bool load();

    template <class T>
    static bool canCreate ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
    {
        return BaseLoader::canCreate (obj, context, arg);
    }

protected:

    bool readSTL(const char* filename);

    bool readBinarySTL(const char* filename);

public:
    //Add Data here
    Data <unsigned int> _headerSize;
    Data <bool> _forceBinary;

};




} // namespace loader

} // namespace component

} // namespace sofa

#endif
