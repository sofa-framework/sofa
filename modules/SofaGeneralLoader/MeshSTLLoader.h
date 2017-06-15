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
#ifndef SOFA_COMPONENT_LOADER_MESHSTLLOADER_H
#define SOFA_COMPONENT_LOADER_MESHSTLLOADER_H
#include "config.h"

#include <sofa/core/loader/MeshLoader.h>
#include <sofa/core/objectmodel/BaseData.h>

namespace sofa
{

namespace component
{

namespace loader
{

// Format doc: http://en.wikipedia.org/wiki/STL_(file_format)
class SOFA_GENERAL_LOADER_API MeshSTLLoader : public sofa::core::loader::MeshLoader
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

    // ascii
    bool readSTL(std::ifstream& file);

    // binary
    bool readBinarySTL(const char* filename);

public:
    //Add Data here
    Data <unsigned int> _headerSize;
    Data <bool> _forceBinary;
    Data <bool> d_mergePositionUsingMap;

};




} // namespace loader

} // namespace component

} // namespace sofa

#endif
