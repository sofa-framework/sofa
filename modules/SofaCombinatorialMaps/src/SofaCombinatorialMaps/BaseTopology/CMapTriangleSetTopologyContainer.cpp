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

#include <SofaCombinatorialMaps/BaseTopology/CMapTriangleSetTopologyContainer.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

SOFA_DECL_CLASS(CMapTriangleSetTopologyContainer)
int CMapTriangleSetTopologyContainerClass = core::RegisterObject("Triangle set topology container backward compatibility")
		.add< CMapTriangleSetTopologyContainer >()
        ;


CMapTriangleSetTopologyContainer::CMapTriangleSetTopologyContainer() :
	Inherit1(),
	map_(nullptr)
{
}

CMapTriangleSetTopologyContainer::~CMapTriangleSetTopologyContainer()
{
}

void CMapTriangleSetTopologyContainer::init()
{
	Inherit1::init();
	this->getContext()->get(map_);
	if (!map_)
		return;
	this->setNbPoints(map_->nb_cells<Vertex::ORBIT>());
}

void CMapTriangleSetTopologyContainer::bwdInit()
{
    Inherit1::bwdInit();
}

void CMapTriangleSetTopologyContainer::reinit()
{
    Inherit1::reinit();
}

void CMapTriangleSetTopologyContainer::reset()
{
    Inherit1::reset();
}

void CMapTriangleSetTopologyContainer::cleanup()
{
    Inherit1::cleanup();
}

const EdgeSetTopologyContainer::SeqEdges& CMapTriangleSetTopologyContainer::getEdges()
{
    return map_->getEdges();
}

const TriangleSetTopologyContainer::SeqTriangles& CMapTriangleSetTopologyContainer::getTriangles()
{
	return map_->getTriangles();
}

size_t CMapTriangleSetTopologyContainer::getNbEdges()
{
	return map_->nb_cells<Edge::ORBIT>();
}

size_t CMapTriangleSetTopologyContainer::getNbTriangles()
{
	return map_->nb_cells<Face::ORBIT>();
}


const TriangleSetTopologyContainer::Edge CMapTriangleSetTopologyContainer::getEdge(TriangleSetTopologyContainer::EdgeID i)
{
	return map_->getEdges()[i];
}

const TriangleSetTopologyContainer::Triangle CMapTriangleSetTopologyContainer::getTriangle(TriangleSetTopologyContainer::TriangleID i)
{
	return map_->getTriangles()[i];
}


const EdgeSetTopologyContainer::EdgesAroundVertex& CMapTriangleSetTopologyContainer::getEdgesAroundVertex(TriangleSetTopologyContainer::PointID i)
{
	return map_->getEdgesAroundVertex(i);
}

const TriangleSetTopologyContainer::EdgesInTriangle& CMapTriangleSetTopologyContainer::getEdgesInTriangle(TriangleSetTopologyContainer::TriangleID i)
{
	return map_->getEdgesInTriangle(i);
}


const TriangleSetTopologyContainer::TrianglesAroundVertex& CMapTriangleSetTopologyContainer::getTrianglesAroundVertex(TriangleSetTopologyContainer::PointID i)
{
	return map_->getTrianglesAroundVertex(i);
}

const TriangleSetTopologyContainer::TrianglesAroundEdge& CMapTriangleSetTopologyContainer::getTrianglesAroundEdge(TriangleSetTopologyContainer::EdgeID i)
{
	return map_->getTrianglesAroundEdge(i);
}


void CMapTriangleSetTopologyContainer::clear()
{
	Inherit1::clear();
}

void CMapTriangleSetTopologyContainer::addEdge(int, int)
{
//	map_->addEdge(a,b);
}

void CMapTriangleSetTopologyContainer::addTriangle(int, int, int)
{
//	map_->addTriangle(a,b,c);
}

bool CMapTriangleSetTopologyContainer::checkConnexity()
{
//	return map_->checkConnexity();
}

size_t CMapTriangleSetTopologyContainer::getNumberOfConnectedComponent()
{
	return map_->getNumberOfConnectedComponent();
}

int CMapTriangleSetTopologyContainer::getRevision() const
{
//	return map_->getRevision();
	return -1;
}

void CMapTriangleSetTopologyContainer::reOrientateTriangle(TriangleSetTopologyContainer::TriangleID id)
{
	return map_->reOrientateTriangle(id);
}

const sofa::helper::vector<TriangleSetTopologyContainer::TriangleID>& CMapTriangleSetTopologyContainer::getTrianglesOnBorder()
{
	static const sofa::helper::vector<TriangleSetTopologyContainer::TriangleID> empty;
	return empty;
}

const sofa::helper::vector<TriangleSetTopologyContainer::EdgeID>& CMapTriangleSetTopologyContainer::getEdgesOnBorder()
{
	static const sofa::helper::vector<TriangleSetTopologyContainer::EdgeID> empty;
	return empty;
}

const sofa::helper::vector<TriangleSetTopologyContainer::PointID>& CMapTriangleSetTopologyContainer::getPointsOnBorder()
{
	static const sofa::helper::vector<TriangleSetTopologyContainer::PointID> empty;
	return empty;
}

void CMapTriangleSetTopologyContainer::updateTopologyEngineGraph()
{
	Inherit1::updateTopologyEngineGraph();
//	map_->updateTopologyEngineGraph();
}

size_t CMapTriangleSetTopologyContainer::getNumberOfElements() const
{
	return map_->getNumberOfConnectedComponent();
}

bool CMapTriangleSetTopologyContainer::checkTopology() const
{
	// TODO : use map_->checkTopology();
	return Inherit1::checkTopology();
//	return map_->checkTopology();
}

void CMapTriangleSetTopologyContainer::createEdgeSetArray()
{
//	map_->
}

const TriangleSetTopologyContainer::VecTriangleID CMapTriangleSetTopologyContainer::getConnectedElement(TriangleSetTopologyContainer::TriangleID elem)
{
	return map_->getConnectedElement(elem);
}

const TriangleSetTopologyContainer::VecTriangleID CMapTriangleSetTopologyContainer::getElementAroundElement(TriangleSetTopologyContainer::TriangleID elem)
{
	return map_->getElementAroundElement(elem);
}

const TriangleSetTopologyContainer::VecTriangleID CMapTriangleSetTopologyContainer::getElementAroundElements(TriangleSetTopologyContainer::VecTriangleID elems)
{
	return map_->getElementAroundElements(elems);
}

void CMapTriangleSetTopologyContainer::createTriangleSetArray()
{
	 // TODO
}

void CMapTriangleSetTopologyContainer::createEdgesInTriangleArray()
{
	// TODO
}

void CMapTriangleSetTopologyContainer::createTrianglesAroundVertexArray()
{
		// NOTHING TODO
}

void CMapTriangleSetTopologyContainer::createTrianglesAroundEdgeArray()
{
		// NOTHING TODO
}

TriangleSetTopologyContainer::TrianglesAroundVertex& CMapTriangleSetTopologyContainer::getTrianglesAroundVertexForModification(const TriangleSetTopologyContainer::PointID vertexIndex)
{
	return m_trianglesAroundVertex[vertexIndex];
}

TriangleSetTopologyContainer::TrianglesAroundEdge& CMapTriangleSetTopologyContainer::getTrianglesAroundEdgeForModification(const TriangleSetTopologyContainer::EdgeID edgeIndex)
{
	return m_trianglesAroundEdge[edgeIndex];
}


} // namespace topology
} // namespace component
} // namespace sofa


