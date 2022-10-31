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

class SOFA_COMPONENT_IO_MESH_API GIDMeshLoader : public sofa::core::loader::MeshLoader
{
public :
	SOFA_CLASS(GIDMeshLoader, sofa::core::loader::MeshLoader);

    typedef sofa::core::topology::Topology::Edge Edge;
    typedef sofa::core::topology::Topology::Triangle Triangle;
    typedef sofa::core::topology::Topology::Quad Quad;
    typedef sofa::core::topology::Topology::Tetrahedron Tetrahedron;
    typedef sofa::core::topology::Topology::Hexahedron Hexahedron;
    typedef sofa::type::Vec3 Coord;


public :
    bool doLoad() override;

protected :
	enum ElementType{ LINEAR, TRIANGLE, QUADRILATERAL, TETRAHEDRA, HEXAHEDRA, PRISM, PYRAMID, SPHERE, CIRCLE };

	GIDMeshLoader();
	~GIDMeshLoader() override;

	bool readGID(std::ifstream& file);

    void doClearBuffers() override;

private :

	bool readLinearElements(std::ifstream& file);
	bool readTriangleElements(std::ifstream& file);
	bool readQuadrilateralElements(std::ifstream& file);
	bool readTetrahedralElements(std::ifstream& file);
	bool readHexahedralElements(std::ifstream& file);

private :
	unsigned short m_dimensions;
	ElementType m_eltType;
	unsigned short m_nNode;

};

} //namespace sofa::component::io::mesh
