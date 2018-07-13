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

#include <SofaBaseTopology/MapHexahedronSetTopologyContainer.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace topology
{

SOFA_DECL_CLASS(MapHexahedronSetTopologyContainer)
int MapHexahedronSetTopologyContainerClass = core::RegisterObject("Hexahedron set topology container backward compatibility")
		.add< MaHexahedronSetTopologyContainer >()
		;

MaHexahedronSetTopologyContainer::MaHexahedronSetTopologyContainer():
	Inherit1(),
	map_(nullptr)
{}

MaHexahedronSetTopologyContainer::~MaHexahedronSetTopologyContainer()
{

}

int MaHexahedronSetTopologyContainer::getNbPoints() const
{
	return map_->nb_cells<Vertex::ORBIT>();
}

void MaHexahedronSetTopologyContainer::init()
{
	this->getContext()->get(map_);
	if (!map_)
		return;

	Inherit1::init();
}

void MaHexahedronSetTopologyContainer::bwdInit()
{
	Inherit1::bwdInit();
}

void MaHexahedronSetTopologyContainer::reinit()
{
	Inherit1::reinit();
}

void MaHexahedronSetTopologyContainer::reset()
{
	Inherit1::reset();
}

void MaHexahedronSetTopologyContainer::cleanup()
{
	Inherit1::cleanup();
}

void MaHexahedronSetTopologyContainer::draw(const sofa::core::visual::VisualParams*)
{
}

bool MaHexahedronSetTopologyContainer::load(const char* filename)
{
}

const EdgeSetTopologyContainer::SeqEdges&MaHexahedronSetTopologyContainer::getEdges()
{
	return map_->getEdges();
}

const QuadSetTopologyContainer::SeqQuads&MaHexahedronSetTopologyContainer::getQuads()
{
	return map_->getQuads();
}

const HexahedronSetTopologyContainer::SeqHexahedra&MaHexahedronSetTopologyContainer::getHexahedra()
{
	return map_->getHexahedra();
}

int MaHexahedronSetTopologyContainer::getNbEdges()
{
	return map_->nb_cells<Edge::ORBIT>();
}

int MaHexahedronSetTopologyContainer::getNbQuads()
{
	return map_->nb_cells<Face::ORBIT>();
}

int MaHexahedronSetTopologyContainer::getNbHexahedra()
{
	return map_->template nb_cells<Volume::ORBIT>();
}

int MaHexahedronSetTopologyContainer::getNbHexas()
{
	return getNbHexahedra();
}

const EdgeSetTopologyContainer::EdgesAroundVertex&MaHexahedronSetTopologyContainer::getEdgesAroundVertex(HexahedronSetTopologyContainer::PointID i)
{
	return map_->getEdgesAroundVertex(i);
}

const QuadSetTopologyContainer::EdgesInQuad&MaHexahedronSetTopologyContainer::getEdgesInQuad(HexahedronSetTopologyContainer::QuadID i)
{
	return map_->getEdgesInQuad(i);
}

const HexahedronSetTopologyContainer::EdgesInHexahedron&MaHexahedronSetTopologyContainer::getEdgesInHexahedron(HexahedronSetTopologyContainer::HexaID i)
{
	return map_->getEdgesInHexahedron(i);
}

const QuadSetTopologyContainer::QuadsAroundVertex&MaHexahedronSetTopologyContainer::getQuadsAroundVertex(HexahedronSetTopologyContainer::PointID i)
{
}

const QuadSetTopologyContainer::QuadsAroundEdge&MaHexahedronSetTopologyContainer::getQuadsAroundEdge(HexahedronSetTopologyContainer::EdgeID i)
{
}

const HexahedronSetTopologyContainer::QuadsInHexahedron&MaHexahedronSetTopologyContainer::getQuadsInHexahedron(HexahedronSetTopologyContainer::HexaID i)
{
	return map_->getQuadsInHexahedron(i);
}

const HexahedronSetTopologyContainer::HexahedraAroundVertex&MaHexahedronSetTopologyContainer::getHexahedraAroundVertex(HexahedronSetTopologyContainer::PointID i)
{
}

const HexahedronSetTopologyContainer::HexahedraAroundEdge&MaHexahedronSetTopologyContainer::getHexahedraAroundEdge(HexahedronSetTopologyContainer::EdgeID i)
{
}

const HexahedronSetTopologyContainer::HexahedraAroundQuad&MaHexahedronSetTopologyContainer::getHexahedraAroundQuad(HexahedronSetTopologyContainer::QuadID i)
{
	return map_->getHexahedraAroundQuad(i);
}

const sofa::core::topology::BaseMeshTopology::VerticesAroundVertex MaHexahedronSetTopologyContainer::getVerticesAroundVertex(HexahedronSetTopologyContainer::PointID i)
{
}

const sofa::helper::vector<sofa::core::topology::Topology::index_type> MaHexahedronSetTopologyContainer::getElementAroundElement(sofa::core::topology::Topology::index_type elem)
{
}

const sofa::helper::vector<sofa::core::topology::Topology::index_type> MaHexahedronSetTopologyContainer::getElementAroundElements(sofa::helper::vector<sofa::core::topology::Topology::index_type> elems)
{
}

void MaHexahedronSetTopologyContainer::clear()
{
}

void MaHexahedronSetTopologyContainer::addPoint(SReal px, SReal py, SReal pz)
{
}

void MaHexahedronSetTopologyContainer::addEdge(int a, int b)
{
}

void MaHexahedronSetTopologyContainer::addQuad(int a, int b, int c, int d)
{
}

void MaHexahedronSetTopologyContainer::addHexa(int a, int b, int c, int d, int e, int f, int g, int h)
{
}

bool MaHexahedronSetTopologyContainer::checkConnexity()
{
	return true;
}

unsigned int MaHexahedronSetTopologyContainer::getNumberOfConnectedComponent()
{
	return 0u;
}

const sofa::helper::vector<sofa::core::topology::Topology::index_type> MaHexahedronSetTopologyContainer::getConnectedElement(sofa::core::topology::Topology::index_type elem)
{
	return sofa::helper::vector<sofa::core::topology::Topology::index_type>();
}

//void MaHexahedronSetTopologyContainer::reOrientateQuad(HexahedronSetTopologyContainer::QuadID id)
//{
//}


const sofa::helper::vector<HexahedronSetTopologyContainer::EdgeID>&MaHexahedronSetTopologyContainer::getEdgesOnBorder()
{
	static const sofa::helper::vector<HexahedronSetTopologyContainer::EdgeID> empty;
	return empty;
}

const sofa::helper::vector<HexahedronSetTopologyContainer::PointID>&MaHexahedronSetTopologyContainer::getPointsOnBorder()
{
	static const sofa::helper::vector<HexahedronSetTopologyContainer::PointID> empty;
	return empty;
}

unsigned int MaHexahedronSetTopologyContainer::getNumberOfElements() const
{
	return map_->template nb_cells<Volume::ORBIT>();
}

bool MaHexahedronSetTopologyContainer::checkTopology() const
{
	return true;
}


} // namespace topology
} // namespace component
} // namespace sofa

