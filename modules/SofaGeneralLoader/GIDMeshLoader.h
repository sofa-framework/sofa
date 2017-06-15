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
#ifndef GIDMESHLOADER_H
#define GIDMESHLOADER_H
#include "config.h"

#include <sofa/core/loader/MeshLoader.h>
#include <sofa/core/topology/Topology.h>


namespace sofa
{

namespace component
{

namespace loader
{

class SOFA_GENERAL_LOADER_API GIDMeshLoader : public sofa::core::loader::MeshLoader
{
public :
	SOFA_CLASS(GIDMeshLoader, sofa::core::loader::MeshLoader);

    typedef sofa::core::topology::Topology::Edge Edge;
    typedef sofa::core::topology::Topology::Triangle Triangle;
    typedef sofa::core::topology::Topology::Quad Quad;
    typedef sofa::core::topology::Topology::Tetrahedron Tetrahedron;
    typedef sofa::core::topology::Topology::Hexahedron Hexahedron;
    typedef sofa::defaulttype::Vector3 Coord;


public :
	virtual bool load();

	template <class T>
	static bool canCreate ( T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg )
	{
		return BaseLoader::canCreate (obj, context, arg);
	}

protected :
	enum ElementType{ LINEAR, TRIANGLE, QUADRILATERAL, TETRAHEDRA, HEXAHEDRA, PRISM, PYRAMID, SPHERE, CIRCLE };

	GIDMeshLoader();
	virtual ~GIDMeshLoader();

	bool readGID(std::ifstream& file);

private :

	bool readLinearElements(std::ifstream& file);
	bool readTriangleElements(std::ifstream& file);
	bool readQuadrilateralElements(std::ifstream& file);
	bool readTetrahedralElements(std::ifstream& file);
	bool readHexahedralElements(std::ifstream& file);


protected :
//	Data<helper::vector<Coord> > m_vertices;
//	Data<helper::vector<Edge> > m_edges;
//	Data<helper::vector<Triangle> > m_triangles;
//	Data<helper::vector<Quad> > m_quads;
//	Data<helper::vector<Tetrahedron> > m_tetrahedra;
//	Data<helper::vector<Hexahedron> > m_hexahedra;

private :
	unsigned short m_dimensions;
	ElementType m_eltType;
	unsigned short m_nNode;


};

} //namespace loader

} // namespace component

} // namespace sofa

#endif // GIDMESHLOADER_H
