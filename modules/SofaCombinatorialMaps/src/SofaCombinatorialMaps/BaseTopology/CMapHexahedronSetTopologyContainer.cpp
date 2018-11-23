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

#include <SofaCombinatorialMaps/BaseTopology/CMapHexahedronSetTopologyContainer.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

SOFA_DECL_CLASS(CMapHexahedronSetTopologyContainer)
int CMapHexahedronSetTopologyContainerClass = core::RegisterObject("Hexahedron set topology container backward compatibility")
		.add< CMapHexahedronSetTopologyContainer >()
		;

CMapHexahedronSetTopologyContainer::CMapHexahedronSetTopologyContainer():
	Inherit1(),
	map_(nullptr)
{}

CMapHexahedronSetTopologyContainer::~CMapHexahedronSetTopologyContainer()
{

}

int CMapHexahedronSetTopologyContainer::getNbPoints() const
{
	return map_->nb_cells<Vertex::ORBIT>();
}

void CMapHexahedronSetTopologyContainer::init()
{
	this->getContext()->get(map_);
	if (!map_)
		return;

	Inherit1::init();
}

void CMapHexahedronSetTopologyContainer::bwdInit()
{
	Inherit1::bwdInit();
}

void CMapHexahedronSetTopologyContainer::reinit()
{
	Inherit1::reinit();
}

void CMapHexahedronSetTopologyContainer::reset()
{
	Inherit1::reset();
}

void CMapHexahedronSetTopologyContainer::cleanup()
{
	Inherit1::cleanup();
}

void CMapHexahedronSetTopologyContainer::draw(const sofa::core::visual::VisualParams*)
{
}

bool CMapHexahedronSetTopologyContainer::load(const char* filename)
{
}

const EdgeSetTopologyContainer::SeqEdges&CMapHexahedronSetTopologyContainer::getEdges()
{
	return map_->getEdges();
}

const QuadSetTopologyContainer::SeqQuads&CMapHexahedronSetTopologyContainer::getQuads()
{
	return map_->getQuads();
}

const HexahedronSetTopologyContainer::SeqHexahedra&CMapHexahedronSetTopologyContainer::getHexahedra()
{
	return map_->getHexahedra();
}

size_t CMapHexahedronSetTopologyContainer::getNbEdges()
{
	return map_->nb_cells<Edge::ORBIT>();
}

size_t CMapHexahedronSetTopologyContainer::getNbQuads()
{
	return map_->nb_cells<Face::ORBIT>();
}

size_t CMapHexahedronSetTopologyContainer::getNbHexahedra()
{
	return map_->template nb_cells<Volume::ORBIT>();
}

size_t CMapHexahedronSetTopologyContainer::getNbHexas()
{
	return getNbHexahedra();
}

const EdgeSetTopologyContainer::EdgesAroundVertex&CMapHexahedronSetTopologyContainer::getEdgesAroundVertex(HexahedronSetTopologyContainer::PointID i)
{
	return map_->getEdgesAroundVertex(i);
}

const QuadSetTopologyContainer::EdgesInQuad&CMapHexahedronSetTopologyContainer::getEdgesInQuad(HexahedronSetTopologyContainer::QuadID i)
{
	return map_->getEdgesInQuad(i);
}

const HexahedronSetTopologyContainer::EdgesInHexahedron&CMapHexahedronSetTopologyContainer::getEdgesInHexahedron(HexahedronSetTopologyContainer::HexaID i)
{
	return map_->getEdgesInHexahedron(i);
}

const QuadSetTopologyContainer::QuadsAroundVertex&CMapHexahedronSetTopologyContainer::getQuadsAroundVertex(HexahedronSetTopologyContainer::PointID i)
{
}

const QuadSetTopologyContainer::QuadsAroundEdge&CMapHexahedronSetTopologyContainer::getQuadsAroundEdge(HexahedronSetTopologyContainer::EdgeID i)
{
}

const HexahedronSetTopologyContainer::QuadsInHexahedron&CMapHexahedronSetTopologyContainer::getQuadsInHexahedron(HexahedronSetTopologyContainer::HexaID i)
{
	return map_->getQuadsInHexahedron(i);
}

const HexahedronSetTopologyContainer::HexahedraAroundVertex&CMapHexahedronSetTopologyContainer::getHexahedraAroundVertex(HexahedronSetTopologyContainer::PointID i)
{
}

const HexahedronSetTopologyContainer::HexahedraAroundEdge&CMapHexahedronSetTopologyContainer::getHexahedraAroundEdge(HexahedronSetTopologyContainer::EdgeID i)
{
}

const HexahedronSetTopologyContainer::HexahedraAroundQuad&CMapHexahedronSetTopologyContainer::getHexahedraAroundQuad(HexahedronSetTopologyContainer::QuadID i)
{
	return map_->getHexahedraAroundQuad(i);
}

const sofa::core::topology::BaseMeshTopology::VerticesAroundVertex CMapHexahedronSetTopologyContainer::getVerticesAroundVertex(HexahedronSetTopologyContainer::PointID i)
{
}

const sofa::helper::vector<sofa::core::topology::Topology::index_type> CMapHexahedronSetTopologyContainer::getElementAroundElement(sofa::core::topology::Topology::index_type elem)
{
}

const sofa::helper::vector<sofa::core::topology::Topology::index_type> CMapHexahedronSetTopologyContainer::getElementAroundElements(sofa::helper::vector<sofa::core::topology::Topology::index_type> elems)
{
}

void CMapHexahedronSetTopologyContainer::clear()
{
}

void CMapHexahedronSetTopologyContainer::addPoint(SReal px, SReal py, SReal pz)
{
}

void CMapHexahedronSetTopologyContainer::addEdge(int a, int b)
{
}

void CMapHexahedronSetTopologyContainer::addQuad(int a, int b, int c, int d)
{
}

void CMapHexahedronSetTopologyContainer::addHexa(int a, int b, int c, int d, int e, int f, int g, int h)
{
}

bool CMapHexahedronSetTopologyContainer::checkConnexity()
{
	return true;
}

size_t CMapHexahedronSetTopologyContainer::getNumberOfConnectedComponent()
{
	return 0u;
}

const sofa::helper::vector<sofa::core::topology::Topology::index_type> CMapHexahedronSetTopologyContainer::getConnectedElement(sofa::core::topology::Topology::index_type elem)
{
	return sofa::helper::vector<sofa::core::topology::Topology::index_type>();
}

//void CMapHexahedronSetTopologyContainer::reOrientateQuad(HexahedronSetTopologyContainer::QuadID id)
//{
//}


const sofa::helper::vector<HexahedronSetTopologyContainer::EdgeID>&CMapHexahedronSetTopologyContainer::getEdgesOnBorder()
{
	static const sofa::helper::vector<HexahedronSetTopologyContainer::EdgeID> empty;
	return empty;
}

const sofa::helper::vector<HexahedronSetTopologyContainer::PointID>&CMapHexahedronSetTopologyContainer::getPointsOnBorder()
{
	static const sofa::helper::vector<HexahedronSetTopologyContainer::PointID> empty;
	return empty;
}

size_t CMapHexahedronSetTopologyContainer::getNumberOfElements() const
{
	return map_->template nb_cells<Volume::ORBIT>();
}

bool CMapHexahedronSetTopologyContainer::checkTopology() const
{
	return true;
}


} // namespace topology
} // namespace component
} // namespace sofa

