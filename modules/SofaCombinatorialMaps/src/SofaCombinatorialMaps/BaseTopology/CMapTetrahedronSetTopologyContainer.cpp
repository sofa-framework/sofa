/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaCombinatorialMaps/BaseTopology/CMapTetrahedronSetTopologyContainer.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

SOFA_DECL_CLASS(CMapTetrahedronSetTopologyContainer)
int CMapTetrahedronSetTopologyContainerClass = core::RegisterObject("Tetrahedron set topology container backward compatibility")
		.add< CMapTetrahedronSetTopologyContainer >()
		;

CMapTetrahedronSetTopologyContainer::CMapTetrahedronSetTopologyContainer():
	Inherit1(),
	map_(nullptr)
{}

CMapTetrahedronSetTopologyContainer::~CMapTetrahedronSetTopologyContainer()
{

}

int CMapTetrahedronSetTopologyContainer::getNbPoints() const
{
	return map_->nb_cells<Vertex::ORBIT>();
}

void CMapTetrahedronSetTopologyContainer::init()
{
	this->getContext()->get(map_);
	if (!map_)
		return;

	Inherit1::init();
}

void CMapTetrahedronSetTopologyContainer::bwdInit()
{
	Inherit1::bwdInit();
}

void CMapTetrahedronSetTopologyContainer::reinit()
{
	Inherit1::reinit();
}

void CMapTetrahedronSetTopologyContainer::reset()
{
	Inherit1::reset();
}

void CMapTetrahedronSetTopologyContainer::cleanup()
{
	Inherit1::cleanup();
}

void CMapTetrahedronSetTopologyContainer::draw(const sofa::core::visual::VisualParams*)
{
}

bool CMapTetrahedronSetTopologyContainer::load(const char* filename)
{
}

const EdgeSetTopologyContainer::SeqEdges&CMapTetrahedronSetTopologyContainer::getEdges()
{
	return map_->getEdges();
}

const TriangleSetTopologyContainer::SeqTriangles&CMapTetrahedronSetTopologyContainer::getTriangles()
{
	return map_->getTriangles();
}

const TetrahedronSetTopologyContainer::SeqTetrahedra&CMapTetrahedronSetTopologyContainer::getTetrahedra()
{
	return map_->getTetrahedra();
}

size_t CMapTetrahedronSetTopologyContainer::getNbEdges()
{
	return map_->nb_cells<Edge::ORBIT>();
}

size_t CMapTetrahedronSetTopologyContainer::getNbTriangles()
{
	return map_->nb_cells<Face::ORBIT>();
}

size_t CMapTetrahedronSetTopologyContainer::getNbTetrahedra()
{
	return map_->template nb_cells<Volume::ORBIT>();
}

size_t CMapTetrahedronSetTopologyContainer::getNbTetras()
{
	return getNbTetrahedra();
}

const EdgeSetTopologyContainer::EdgesAroundVertex&CMapTetrahedronSetTopologyContainer::getEdgesAroundVertex(TetrahedronSetTopologyContainer::PointID i)
{
	return map_->getEdgesAroundVertex(i);
}

const TriangleSetTopologyContainer::EdgesInTriangle&CMapTetrahedronSetTopologyContainer::getEdgesInTriangle(TetrahedronSetTopologyContainer::TriangleID i)
{
	return map_->getEdgesInTriangle(i);
}

const TetrahedronSetTopologyContainer::EdgesInTetrahedron&CMapTetrahedronSetTopologyContainer::getEdgesInTetrahedron(TetrahedronSetTopologyContainer::TetraID i)
{
	return map_->getEdgesInTetrahedron(i);
}

const TriangleSetTopologyContainer::TrianglesAroundVertex&CMapTetrahedronSetTopologyContainer::getTrianglesAroundVertex(TetrahedronSetTopologyContainer::PointID i)
{
}

const TriangleSetTopologyContainer::TrianglesAroundEdge&CMapTetrahedronSetTopologyContainer::getTrianglesAroundEdge(TetrahedronSetTopologyContainer::EdgeID i)
{
}

const TetrahedronSetTopologyContainer::TrianglesInTetrahedron&CMapTetrahedronSetTopologyContainer::getTrianglesInTetrahedron(TetrahedronSetTopologyContainer::TetraID i)
{
	return map_->getTrianglesInTetrahedron(i);
}

const TetrahedronSetTopologyContainer::TetrahedraAroundVertex&CMapTetrahedronSetTopologyContainer::getTetrahedraAroundVertex(TetrahedronSetTopologyContainer::PointID i)
{
}

const TetrahedronSetTopologyContainer::TetrahedraAroundEdge&CMapTetrahedronSetTopologyContainer::getTetrahedraAroundEdge(TetrahedronSetTopologyContainer::EdgeID i)
{
}

const TetrahedronSetTopologyContainer::TetrahedraAroundTriangle&CMapTetrahedronSetTopologyContainer::getTetrahedraAroundTriangle(TetrahedronSetTopologyContainer::TriangleID i)
{
	return map_->getTetrahedraAroundTriangle(i);
}

const sofa::core::topology::BaseMeshTopology::VerticesAroundVertex CMapTetrahedronSetTopologyContainer::getVerticesAroundVertex(TetrahedronSetTopologyContainer::PointID i)
{
}

const sofa::helper::vector<sofa::core::topology::Topology::index_type> CMapTetrahedronSetTopologyContainer::getElementAroundElement(sofa::core::topology::Topology::index_type elem)
{
}

const sofa::helper::vector<sofa::core::topology::Topology::index_type> CMapTetrahedronSetTopologyContainer::getElementAroundElements(sofa::helper::vector<sofa::core::topology::Topology::index_type> elems)
{
}

void CMapTetrahedronSetTopologyContainer::clear()
{
}

void CMapTetrahedronSetTopologyContainer::addPoint(SReal px, SReal py, SReal pz)
{
}

void CMapTetrahedronSetTopologyContainer::addEdge(int a, int b)
{
}

void CMapTetrahedronSetTopologyContainer::addTriangle(int a, int b, int c)
{
}

void CMapTetrahedronSetTopologyContainer::addTetra(int a, int b, int c, int d)
{
}

bool CMapTetrahedronSetTopologyContainer::checkConnexity()
{
	return true;
}

size_t CMapTetrahedronSetTopologyContainer::getNumberOfConnectedComponent()
{
	return 0u;
}

const sofa::helper::vector<sofa::core::topology::Topology::index_type> CMapTetrahedronSetTopologyContainer::getConnectedElement(sofa::core::topology::Topology::index_type elem)
{
	return sofa::helper::vector<sofa::core::topology::Topology::index_type>();
}

void CMapTetrahedronSetTopologyContainer::reOrientateTriangle(TetrahedronSetTopologyContainer::TriangleID id)
{
}

const sofa::helper::vector<TetrahedronSetTopologyContainer::TriangleID>&CMapTetrahedronSetTopologyContainer::getTrianglesOnBorder()
{
	static const sofa::helper::vector<TetrahedronSetTopologyContainer::TriangleID> empty;
	return empty;
}

const sofa::helper::vector<TetrahedronSetTopologyContainer::EdgeID>&CMapTetrahedronSetTopologyContainer::getEdgesOnBorder()
{
	static const sofa::helper::vector<TetrahedronSetTopologyContainer::EdgeID> empty;
	return empty;
}

const sofa::helper::vector<TetrahedronSetTopologyContainer::PointID>&CMapTetrahedronSetTopologyContainer::getPointsOnBorder()
{
	static const sofa::helper::vector<TetrahedronSetTopologyContainer::PointID> empty;
	return empty;
}

size_t CMapTetrahedronSetTopologyContainer::getNumberOfElements() const
{
	return map_->template nb_cells<Volume::ORBIT>();
}

bool CMapTetrahedronSetTopologyContainer::checkTopology() const
{
	return true;
}


} // namespace topology
} // namespace component
} // namespace sofa

