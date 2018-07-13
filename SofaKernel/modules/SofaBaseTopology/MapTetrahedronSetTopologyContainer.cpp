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

#include <SofaBaseTopology/MapTetrahedronSetTopologyContainer.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

SOFA_DECL_CLASS(MapTetrahedronSetTopologyContainer)
int MapTetrahedronSetTopologyContainerClass = core::RegisterObject("Tetrahedron set topology container backward compatibility")
		.add< MapTetrahedronSetTopologyContainer >()
		;

MapTetrahedronSetTopologyContainer::MapTetrahedronSetTopologyContainer():
	Inherit1(),
	map_(nullptr)
{}

MapTetrahedronSetTopologyContainer::~MapTetrahedronSetTopologyContainer()
{

}

int MapTetrahedronSetTopologyContainer::getNbPoints() const
{
	return map_->nb_cells<Vertex::ORBIT>();
}

void MapTetrahedronSetTopologyContainer::init()
{
	this->getContext()->get(map_);
	if (!map_)
		return;

	Inherit1::init();
}

void MapTetrahedronSetTopologyContainer::bwdInit()
{
	Inherit1::bwdInit();
}

void MapTetrahedronSetTopologyContainer::reinit()
{
	Inherit1::reinit();
}

void MapTetrahedronSetTopologyContainer::reset()
{
	Inherit1::reset();
}

void MapTetrahedronSetTopologyContainer::cleanup()
{
	Inherit1::cleanup();
}

void MapTetrahedronSetTopologyContainer::draw(const sofa::core::visual::VisualParams*)
{
}

bool MapTetrahedronSetTopologyContainer::load(const char* filename)
{
}

const EdgeSetTopologyContainer::SeqEdges&MapTetrahedronSetTopologyContainer::getEdges()
{
	return map_->getEdges();
}

const TriangleSetTopologyContainer::SeqTriangles&MapTetrahedronSetTopologyContainer::getTriangles()
{
	return map_->getTriangles();
}

const TetrahedronSetTopologyContainer::SeqTetrahedra&MapTetrahedronSetTopologyContainer::getTetrahedra()
{
	return map_->getTetrahedra();
}

int MapTetrahedronSetTopologyContainer::getNbEdges()
{
	return map_->nb_cells<Edge::ORBIT>();
}

int MapTetrahedronSetTopologyContainer::getNbTriangles()
{
	return map_->nb_cells<Face::ORBIT>();
}

int MapTetrahedronSetTopologyContainer::getNbTetrahedra()
{
	return map_->template nb_cells<Volume::ORBIT>();
}

int MapTetrahedronSetTopologyContainer::getNbTetras()
{
	return getNbTetrahedra();
}

const EdgeSetTopologyContainer::EdgesAroundVertex&MapTetrahedronSetTopologyContainer::getEdgesAroundVertex(TetrahedronSetTopologyContainer::PointID i)
{
	return map_->getEdgesAroundVertex(i);
}

const TriangleSetTopologyContainer::EdgesInTriangle&MapTetrahedronSetTopologyContainer::getEdgesInTriangle(TetrahedronSetTopologyContainer::TriangleID i)
{
	return map_->getEdgesInTriangle(i);
}

const TetrahedronSetTopologyContainer::EdgesInTetrahedron&MapTetrahedronSetTopologyContainer::getEdgesInTetrahedron(TetrahedronSetTopologyContainer::TetraID i)
{
	return map_->getEdgesInTetrahedron(i);
}

const TriangleSetTopologyContainer::TrianglesAroundVertex&MapTetrahedronSetTopologyContainer::getTrianglesAroundVertex(TetrahedronSetTopologyContainer::PointID i)
{
}

const TriangleSetTopologyContainer::TrianglesAroundEdge&MapTetrahedronSetTopologyContainer::getTrianglesAroundEdge(TetrahedronSetTopologyContainer::EdgeID i)
{
}

const TetrahedronSetTopologyContainer::TrianglesInTetrahedron&MapTetrahedronSetTopologyContainer::getTrianglesInTetrahedron(TetrahedronSetTopologyContainer::TetraID i)
{
	return map_->getTrianglesInTetrahedron(i);
}

const TetrahedronSetTopologyContainer::TetrahedraAroundVertex&MapTetrahedronSetTopologyContainer::getTetrahedraAroundVertex(TetrahedronSetTopologyContainer::PointID i)
{
}

const TetrahedronSetTopologyContainer::TetrahedraAroundEdge&MapTetrahedronSetTopologyContainer::getTetrahedraAroundEdge(TetrahedronSetTopologyContainer::EdgeID i)
{
}

const TetrahedronSetTopologyContainer::TetrahedraAroundTriangle&MapTetrahedronSetTopologyContainer::getTetrahedraAroundTriangle(TetrahedronSetTopologyContainer::TriangleID i)
{
	return map_->getTetrahedraAroundTriangle(i);
}

const sofa::core::topology::BaseMeshTopology::VerticesAroundVertex MapTetrahedronSetTopologyContainer::getVerticesAroundVertex(TetrahedronSetTopologyContainer::PointID i)
{
}

const sofa::helper::vector<sofa::core::topology::Topology::index_type> MapTetrahedronSetTopologyContainer::getElementAroundElement(sofa::core::topology::Topology::index_type elem)
{
}

const sofa::helper::vector<sofa::core::topology::Topology::index_type> MapTetrahedronSetTopologyContainer::getElementAroundElements(sofa::helper::vector<sofa::core::topology::Topology::index_type> elems)
{
}

void MapTetrahedronSetTopologyContainer::clear()
{
}

void MapTetrahedronSetTopologyContainer::addPoint(SReal px, SReal py, SReal pz)
{
}

void MapTetrahedronSetTopologyContainer::addEdge(int a, int b)
{
}

void MapTetrahedronSetTopologyContainer::addTriangle(int a, int b, int c)
{
}

void MapTetrahedronSetTopologyContainer::addTetra(int a, int b, int c, int d)
{
}

bool MapTetrahedronSetTopologyContainer::checkConnexity()
{
	return true;
}

unsigned int MapTetrahedronSetTopologyContainer::getNumberOfConnectedComponent()
{
	return 0u;
}

const sofa::helper::vector<sofa::core::topology::Topology::index_type> MapTetrahedronSetTopologyContainer::getConnectedElement(sofa::core::topology::Topology::index_type elem)
{
	return sofa::helper::vector<sofa::core::topology::Topology::index_type>();
}

void MapTetrahedronSetTopologyContainer::reOrientateTriangle(TetrahedronSetTopologyContainer::TriangleID id)
{
}

const sofa::helper::vector<TetrahedronSetTopologyContainer::TriangleID>&MapTetrahedronSetTopologyContainer::getTrianglesOnBorder()
{
	static const sofa::helper::vector<TetrahedronSetTopologyContainer::TriangleID> empty;
	return empty;
}

const sofa::helper::vector<TetrahedronSetTopologyContainer::EdgeID>&MapTetrahedronSetTopologyContainer::getEdgesOnBorder()
{
	static const sofa::helper::vector<TetrahedronSetTopologyContainer::EdgeID> empty;
	return empty;
}

const sofa::helper::vector<TetrahedronSetTopologyContainer::PointID>&MapTetrahedronSetTopologyContainer::getPointsOnBorder()
{
	static const sofa::helper::vector<TetrahedronSetTopologyContainer::PointID> empty;
	return empty;
}

unsigned int MapTetrahedronSetTopologyContainer::getNumberOfElements() const
{
	return map_->template nb_cells<Volume::ORBIT>();
}

bool MapTetrahedronSetTopologyContainer::checkTopology() const
{
	return true;
}


} // namespace topology
} // namespace component
} // namespace sofa

