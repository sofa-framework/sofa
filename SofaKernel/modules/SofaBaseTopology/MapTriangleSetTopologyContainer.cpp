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

#include <SofaBaseTopology/MapTriangleSetTopologyContainer.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

SOFA_DECL_CLASS(MapTriangleSetTopologyContainer)
int MapTriangleSetTopologyContainerClass = core::RegisterObject("Triangle set topology container backward compatibility")
        .add< MapTriangleSetTopologyContainer >()
        ;


MapTriangleSetTopologyContainer::MapTriangleSetTopologyContainer() :
	Inherit1(),
	map_(nullptr)
{
}

MapTriangleSetTopologyContainer::~MapTriangleSetTopologyContainer()
{
}

void MapTriangleSetTopologyContainer::init()
{
	Inherit1::init();
	this->getContext()->get(map_);
	if (!map_)
		return;
	this->setNbPoints(map_->nb_cells<Vertex::ORBIT>());
}

void MapTriangleSetTopologyContainer::bwdInit()
{
    Inherit1::bwdInit();
}

void MapTriangleSetTopologyContainer::reinit()
{
    Inherit1::reinit();
}

void MapTriangleSetTopologyContainer::reset()
{
    Inherit1::reset();
}

void MapTriangleSetTopologyContainer::cleanup()
{
    Inherit1::cleanup();
}

const EdgeSetTopologyContainer::SeqEdges&MapTriangleSetTopologyContainer::getEdges()
{
    return map_->getEdges();
}

const TriangleSetTopologyContainer::SeqTriangles&MapTriangleSetTopologyContainer::getTriangles()
{
	return map_->getTriangles();
}

int MapTriangleSetTopologyContainer::getNbEdges()
{
	return map_->nb_cells<Edge::ORBIT>();
}

int MapTriangleSetTopologyContainer::getNbTriangles()
{
	return map_->nb_cells<Face::ORBIT>();
}


const TriangleSetTopologyContainer::Edge MapTriangleSetTopologyContainer::getEdge(TriangleSetTopologyContainer::EdgeID i)
{
	return map_->getEdges()[i];
}

const TriangleSetTopologyContainer::Triangle MapTriangleSetTopologyContainer::getTriangle(TriangleSetTopologyContainer::TriangleID i)
{
	return map_->getTriangles()[i];
}


const EdgeSetTopologyContainer::EdgesAroundVertex&MapTriangleSetTopologyContainer::getEdgesAroundVertex(TriangleSetTopologyContainer::PointID i)
{
	return map_->getEdgesAroundVertex(i);
}

const TriangleSetTopologyContainer::EdgesInTriangle&MapTriangleSetTopologyContainer::getEdgesInTriangle(TriangleSetTopologyContainer::TriangleID i)
{
	return map_->getEdgesInTriangle(i);
}


const TriangleSetTopologyContainer::TrianglesAroundVertex&MapTriangleSetTopologyContainer::getTrianglesAroundVertex(TriangleSetTopologyContainer::PointID i)
{
	return map_->getTrianglesAroundVertex(i);
}

const TriangleSetTopologyContainer::TrianglesAroundEdge&MapTriangleSetTopologyContainer::getTrianglesAroundEdge(TriangleSetTopologyContainer::EdgeID i)
{
	return map_->getTrianglesAroundEdge(i);
}

//const core::topology::BaseMeshTopology::VerticesAroundVertex MapTriangleSetTopologyContainer::getVerticesAroundVertex(TriangleSetTopologyContainer::PointID i)
//{
//	return Inherit1::getVerticesAroundVertex(i);
//}

//int MapTriangleSetTopologyContainer::getEdgeIndex(TriangleSetTopologyContainer::PointID v1, TriangleSetTopologyContainer::PointID v2)
//{
//	return map_->getEdgeIndex(v1, v2);
//}

//int MapTriangleSetTopologyContainer::getTriangleIndex(TriangleSetTopologyContainer::PointID v1, TriangleSetTopologyContainer::PointID v2, TriangleSetTopologyContainer::PointID v3)
//{
//	return map_->getTriangleIndex(v1, v2, v3);
//}

//int MapTriangleSetTopologyContainer::getVertexIndexInTriangle(const TriangleSetTopologyContainer::Triangle& t, TriangleSetTopologyContainer::PointID vertexIndex) const
//{
//	return map_->getVertexIndexInTriangle(t, vertexIndex);
//}

//int MapTriangleSetTopologyContainer::getEdgeIndexInTriangle(const TriangleSetTopologyContainer::EdgesInTriangle& t, TriangleSetTopologyContainer::EdgeID edgeIndex) const
//{
//	return map_->getEdgeIndexInTriangle(t,edgeIndex);
//}



void MapTriangleSetTopologyContainer::clear()
{
	Inherit1::clear();
}

void MapTriangleSetTopologyContainer::addEdge(int, int)
{
//	map_->addEdge(a,b);
}

void MapTriangleSetTopologyContainer::addTriangle(int, int, int)
{
//	map_->addTriangle(a,b,c);
}

bool MapTriangleSetTopologyContainer::checkConnexity()
{
//	return map_->checkConnexity();
}

unsigned int MapTriangleSetTopologyContainer::getNumberOfConnectedComponent()
{
	return map_->getNumberOfConnectedComponent();
}

int MapTriangleSetTopologyContainer::getRevision() const
{
//	return map_->getRevision();
	return -1;
}

void MapTriangleSetTopologyContainer::reOrientateTriangle(TriangleSetTopologyContainer::TriangleID id)
{
	return map_->reOrientateTriangle(id);
}

const sofa::helper::vector<TriangleSetTopologyContainer::TriangleID>&MapTriangleSetTopologyContainer::getTrianglesOnBorder()
{
	static const sofa::helper::vector<TriangleSetTopologyContainer::TriangleID> empty;
	return empty;
}

const sofa::helper::vector<TriangleSetTopologyContainer::EdgeID>&MapTriangleSetTopologyContainer::getEdgesOnBorder()
{
	static const sofa::helper::vector<TriangleSetTopologyContainer::EdgeID> empty;
	return empty;
}

const sofa::helper::vector<TriangleSetTopologyContainer::PointID>&MapTriangleSetTopologyContainer::getPointsOnBorder()
{
	static const sofa::helper::vector<TriangleSetTopologyContainer::PointID> empty;
	return empty;
}

void MapTriangleSetTopologyContainer::updateTopologyEngineGraph()
{
	Inherit1::updateTopologyEngineGraph();
//	map_->updateTopologyEngineGraph();
}

unsigned int MapTriangleSetTopologyContainer::getNumberOfElements() const
{
	return map_->getNumberOfConnectedComponent();
}

bool MapTriangleSetTopologyContainer::checkTopology() const
{
	// TODO : use map_->checkTopology();
	return Inherit1::checkTopology();
//	return map_->checkTopology();
}

void MapTriangleSetTopologyContainer::createEdgeSetArray()
{
//	map_->
}

const TriangleSetTopologyContainer::VecTriangleID MapTriangleSetTopologyContainer::getConnectedElement(TriangleSetTopologyContainer::TriangleID elem)
{
	return map_->getConnectedElement(elem);
}

const TriangleSetTopologyContainer::VecTriangleID MapTriangleSetTopologyContainer::getElementAroundElement(TriangleSetTopologyContainer::TriangleID elem)
{
	return map_->getElementAroundElement(elem);
}

const TriangleSetTopologyContainer::VecTriangleID MapTriangleSetTopologyContainer::getElementAroundElements(TriangleSetTopologyContainer::VecTriangleID elems)
{
	return map_->getElementAroundElements(elems);
}

void MapTriangleSetTopologyContainer::createTriangleSetArray()
{
	 // TODO
}

void MapTriangleSetTopologyContainer::createEdgesInTriangleArray()
{
	// TODO
}

void MapTriangleSetTopologyContainer::createTrianglesAroundVertexArray()
{
		// NOTHING TODO
}

void MapTriangleSetTopologyContainer::createTrianglesAroundEdgeArray()
{
		// NOTHING TODO
}

TriangleSetTopologyContainer::TrianglesAroundVertex&MapTriangleSetTopologyContainer::getTrianglesAroundVertexForModification(const TriangleSetTopologyContainer::PointID vertexIndex)
{
	return m_trianglesAroundVertex[vertexIndex];
}

TriangleSetTopologyContainer::TrianglesAroundEdge&MapTriangleSetTopologyContainer::getTrianglesAroundEdgeForModification(const TriangleSetTopologyContainer::EdgeID edgeIndex)
{
	return m_trianglesAroundEdge[edgeIndex];
}


} // namespace topology
} // namespace component
} // namespace sofa


