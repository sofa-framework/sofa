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
#include <sofa/helper/system/config.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/Factory.inl>

namespace sofa
{

namespace helper
{

template class Factory<std::string, io::Mesh, std::string>;

namespace io
{

SOFA_LINK_CLASS(MeshOBJ)
SOFA_LINK_CLASS(MeshTrian)
SOFA_LINK_CLASS(MeshSTL)

Mesh* Mesh::Create(const std::string& filename)
{
    std::string loader="default";
    std::string::size_type p = filename.rfind('.');
    if (p!=std::string::npos)
        loader = std::string(filename, p+1);
    return FactoryMesh::CreateObject(loader, filename);
}

Mesh* Mesh::Create(const std::string& loader, const std::string& filename)
{
    return FactoryMesh::CreateObject(loader, filename);
}

} // namespace io

} // namespace helper

} // namespace sofa

