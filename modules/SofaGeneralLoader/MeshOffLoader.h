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
#ifndef SOFA_COMPONENT_LOADER_MESHOFFLOADER_H
#define SOFA_COMPONENT_LOADER_MESHOFFLOADER_H
#include "config.h"

#include <sofa/core/loader/MeshLoader.h>

namespace sofa
{

namespace component
{

namespace loader
{

class SOFA_GENERAL_LOADER_API MeshOffLoader : public sofa::core::loader::MeshLoader
{
public:
    SOFA_CLASS(MeshOffLoader,sofa::core::loader::MeshLoader);

    virtual bool load() override;

    template <class T>
    static bool canCreate ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
    {
        return BaseLoader::canCreate (obj, context, arg);
    }


protected:

    bool readOFF(std::ifstream &file, const char* filename);


public:
    // Add specific Data here:


};




} // namespace loader

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_LOADER_MESHOFFLOADER_H
