/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/io/mesh/config.h>

#include <sofa/core/loader/MeshLoader.h>

namespace sofa::component::io::mesh
{

class SOFA_COMPONENT_IO_MESH_API MeshGmshLoader : public sofa::core::loader::MeshLoader
{
public:
    SOFA_CLASS(MeshGmshLoader,sofa::core::loader::MeshLoader);

    bool doLoad() override;

protected:

    void doClearBuffers() override;

    bool readGmsh(std::ifstream &file, const unsigned int gmshFormat);

    void addInGroup(type::vector< sofa::core::loader::PrimitiveGroup>& group,int tag,int eid);

    void normalizeGroup(type::vector< sofa::core::loader::PrimitiveGroup>& group);

public:

};




} //namespace sofa::component::io::mesh
