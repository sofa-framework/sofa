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

/// Cette classe permet la fabrication d'un visuel pour un fichier de type trian
/// ces fichiers se presentent de la maniere suivante
/// nombre de sommets
///liste des coordonnees des sommets ex 1.45 1.25 6.85
/// nombre de faces
///liste de toutes les faces ex 1 2 3 0 0 0 les 3 derniers chiffres ne sont pas utilises pour le moment

class SOFA_COMPONENT_IO_MESH_API MeshTrianLoader : public sofa::core::loader::MeshLoader
{
public:
    SOFA_CLASS(MeshTrianLoader,sofa::core::loader::MeshLoader);
protected:
    MeshTrianLoader();
public:
    bool doLoad() override;

protected:

    void doClearBuffers() override;

    bool readTrian(const char* filename);

    bool readTrian2(const char* filename);

public:
    //Add specific Data here:
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data <bool> p_trian2;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data <type::vector< type::fixed_array <int,3> > > neighborTable;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data <type::vector< type::vector<unsigned int> > > edgesOnBorder;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data <type::vector<unsigned int> > trianglesOnBorderList;

    Data <bool> d_trian2; ///< Set to true if the mesh is a trian2 format.
    Data <type::vector< type::fixed_array <int,3> > > d_neighborTable; ///< Table of neighborhood triangle indices for each triangle.
    Data <type::vector< type::vector<unsigned int> > > d_edgesOnBorder; ///< List of edges which are on the border of the mesh loaded.
    Data <type::vector<unsigned int> > d_trianglesOnBorderList; ///< List of triangle indices which are on the border of the mesh loaded.
};


} //namespace sofa::component::io::mesh
