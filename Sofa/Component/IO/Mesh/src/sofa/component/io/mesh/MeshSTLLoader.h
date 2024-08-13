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

// Format doc: http://en.wikipedia.org/wiki/STL_(file_format)
class SOFA_COMPONENT_IO_MESH_API MeshSTLLoader : public sofa::core::loader::MeshLoader
{
public:
    SOFA_CLASS(MeshSTLLoader,sofa::core::loader::MeshLoader);
protected:
    MeshSTLLoader();

protected:

    // ascii
    bool readSTL(std::ifstream& file);

    // binary
    bool readBinarySTL(const char* filename);

private:
    void doClearBuffers() override;
    bool doLoad() override;

public:
    //Add Data here
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data <bool> _forceBinary;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data <unsigned int> _headerSize;



    Data <unsigned int> d_headerSize; ///< Size of the header binary file (just before the number of facet).
    Data <bool> d_forceBinary; ///< Force reading in binary mode. Even in first keyword of the file is solid.
    Data <bool> d_mergePositionUsingMap; ///< Since positions are duplicated in a STL, they have to be merged. Using a map to do so will temporarily duplicate memory but should be more efficient. Disable it if memory is really an issue.

};

} //namespace sofa::component::io::mesh
