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
#include <sofa/core/topology/MapTopology.h>
#include <sofa/core/topology/CMBaseTopologyEngine.h>
namespace sofa
{

namespace core
{

namespace topology
{

void MapTopology::cleanup()
{
	Inherit1::cleanup();
}

void MapTopology::createHexahedraAroundVertexArray()
{
	serr << "createHexahedraAroundVertexArray not implemented" << sendl;
}

void MapTopology::createHexahedraAroundQuadArray()
{
	serr << "createHexahedraAroundQuadArray not implemented" << sendl;
}

void MapTopology::createEdgesInHexahedronArray()
{
	serr << "createEdgesInHexahedronArray not implemented" << sendl;
}

void MapTopology::createQuadsInHexahedronArray()
{
	serr << "createQuadsInHexahedronArray not implemented" << sendl;
}

const BaseMeshTopology::SeqEdges& MapTopology::getEdges()
{
	return d_edge.getValue();
}

const BaseMeshTopology::SeqTriangles& MapTopology::getTriangles()
{
	if (d_triangle.getValue().empty())
		createTriangleSetArray();
	return d_triangle.getValue();
}

const BaseMeshTopology::SeqQuads& MapTopology::getQuads()
{
	return d_quad.getValue();
}

const BaseMeshTopology::SeqTetrahedra& MapTopology::getTetrahedra()
{
	return d_tetra.getValue();
}

const BaseMeshTopology::SeqHexahedra& MapTopology::getHexahedra() const
{
	return d_hexa.getValue();
}

unsigned int MapTopology::getNbHexahedra() const
{
	return getHexahedra().size();
}

const BaseMeshTopology::EdgesAroundVertex& MapTopology::getEdgesAroundVertex(Topology::PointID i)
{
	if (!m_edgesAroundVertex.is_valid())
		createEdgesAroundVertexArray();
	return m_edgesAroundVertex[i];
}

const BaseMeshTopology::EdgesInTriangle& MapTopology::getEdgesInTriangle(Topology::TriangleID i)
{
	const static EdgesInTriangle empty;
	if (!m_edgesInTriangle.is_valid())
		createEdgesInTriangleArray();
	return m_edgesInTriangle.is_valid() ? m_edgesInTriangle[i] : empty;

}

const BaseMeshTopology::EdgesInQuad& MapTopology::getEdgesInQuad(Topology::QuadID i)
{
	const static EdgesInQuad empty;
	if (!m_edgesInQuad.is_valid())
		createEdgesInQuadArray();
	return m_edgesInQuad.is_valid() ? m_edgesInQuad[i] : empty;
}

const BaseMeshTopology::TrianglesAroundVertex& MapTopology::getTrianglesAroundVertex(Topology::PointID i)
{
    return m_trianglesAroundVertex[i];
}

const BaseMeshTopology::TrianglesAroundEdge& MapTopology::getTrianglesAroundEdge(Topology::EdgeID i)
{
    return m_trianglesAroundEdge[i];
}

const BaseMeshTopology::QuadsAroundVertex& MapTopology::getQuadsAroundVertex(Topology::PointID i)
{
	return m_quadsAroundVertex[i];
}

const BaseMeshTopology::QuadsAroundEdge& MapTopology::getQuadsAroundEdge(Topology::EdgeID i)
{
	return m_quadsAroundEdge[i];
}

const BaseMeshTopology::VerticesAroundVertex MapTopology::getVerticesAroundVertex(Topology::PointID i)
{
//	return m_ver[i];
	return VerticesAroundVertex();
}

const MapTopology::TrianglesInTetrahedron& MapTopology::getTrianglesInTetrahedron(MapTopology::TetrahedronID i)
{
	const static TrianglesInTetrahedron empty;
	if (!m_trianglesInTetrahedron.is_valid())
		createTrianglesInTetrahedronArray();
	return m_trianglesInTetrahedron.is_valid()? m_trianglesInTetrahedron[i] : empty;
}

const MapTopology::EdgesInTetrahedron& MapTopology::getEdgesInTetrahedron(MapTopology::TetrahedronID i)
{
	const static EdgesInTetrahedron empty;
	if (!m_edgesInTetrahedron.is_valid())
		createEdgesInTetrahedronArray();
	return m_edgesInTetrahedron.is_valid()? m_edgesInTetrahedron[i] : empty;
}

const MapTopology::TetrahedraAroundTriangle& MapTopology::getTetrahedraAroundTriangle(MapTopology::TetrahedronID i)
{
	const static TetrahedraAroundTriangle empty;
	if (!m_tetrahedraAroundTriangle.is_valid())
		createTetrahedraAroundTriangleArray();
	return m_tetrahedraAroundTriangle.is_valid()? m_tetrahedraAroundTriangle[i] : empty;
}

const MapTopology::HexahedraAroundVertex& MapTopology::getHexahedraAroundVertex(MapTopology::PointID i)
{
	const static HexahedraAroundVertex empty;
	if (!m_hexahedraAroundVertex.is_valid())
		createHexahedraAroundVertexArray();
	return m_hexahedraAroundVertex.is_valid()? m_hexahedraAroundVertex[i] : empty;
}

const MapTopology::HexahedraAroundQuad& MapTopology::getHexahedraAroundQuad(MapTopology::QuadID i)
{
	const static HexahedraAroundQuad empty;
	if (!m_hexahedraAroundQuad.is_valid())
		createHexahedraAroundQuadArray();
	return m_hexahedraAroundQuad.is_valid()? m_hexahedraAroundQuad[i] : empty;
}

const MapTopology::EdgesInHexahedron& MapTopology::getEdgesInHexahedron(MapTopology::HexahedronID i)
{
	const static EdgesInHexahedron empty;
	if (!m_edgesInHexahedron.is_valid())
		createEdgesInHexahedronArray();
	return m_edgesInHexahedron.is_valid()? m_edgesInHexahedron[i] : empty;
}

const MapTopology::QuadsInHexahedron& MapTopology::getQuadsInHexahedron(MapTopology::HexahedronID i)
{
	const static QuadsInHexahedron empty;
	if (!m_quadsInHexahedron.is_valid())
		createQuadsInHexahedronArray();
	return m_quadsInHexahedron.is_valid()? m_quadsInHexahedron[i] : empty;
}

const sofa::helper::vector<Topology::index_type> MapTopology::getElementAroundElement(Topology::index_type elem)
{
	return sofa::helper::vector<Topology::index_type>();
}

const sofa::helper::vector<Topology::index_type> MapTopology::getElementAroundElements(sofa::helper::vector<Topology::index_type> elems)
{
	return sofa::helper::vector<Topology::index_type>();
}

int MapTopology::getEdgeIndex(Topology::PointID v1, Topology::PointID v2)
{
	return -1;
}

int MapTopology::getTriangleIndex(Topology::PointID v1, Topology::PointID v2, Topology::PointID v3)
{
	return -1;
}

int MapTopology::getQuadIndex(Topology::PointID v1, Topology::PointID v2, Topology::PointID v3, Topology::PointID v4)
{
	return -1;
}

int MapTopology::getVertexIndexInTriangle(const TriangleIds& t, Topology::PointID vertexIndex) const
{
    return -1;
}

int MapTopology::getEdgeIndexInTriangle(const BaseMeshTopology::EdgesInTriangle& t, Topology::EdgeID edgeIndex) const
{
	return -1;
}

int MapTopology::getVertexIndexInQuad(const Topology::Quad& t, Topology::PointID vertexIndex) const
{
	return -1;
}

int MapTopology::getEdgeIndexInQuad(const BaseMeshTopology::EdgesInQuad& t, Topology::EdgeID edgeIndex) const
{
	return -1;
}

const sofa::helper::vector<Topology::index_type> MapTopology::getConnectedElement(Topology::index_type elem)
{
	return sofa::helper::vector<Topology::index_type> ();
}

void MapTopology::reOrientateTriangle(Topology::TriangleID id)
{

}

void MapTopology::addTopologyChange(const cm_topology::TopologyChange* topologyChange)
{
	auto& my_changeList = *(m_changeList.beginEdit());
	my_changeList.push_back(topologyChange);
	m_changeList.endEdit();
}

void MapTopology::addStateChange(const cm_topology::TopologyChange* topologyChange)
{
	auto& my_stateChangeList = *(m_stateChangeList.beginEdit());
	my_stateChangeList.push_back(topologyChange);
	m_stateChangeList.endEdit();
}

void MapTopology::addTopologyEngine(cm_topology::TopologyEngine* _topologyEngine)
{
	m_topologyEngineList.push_back(_topologyEngine);
	m_topologyEngineList.back()->m_changeList.setParent(&this->m_changeList);
	this->updateTopologyEngineGraph();
}

void MapTopology::updateTopologyEngineGraph()
{
	this->updateDataEngineGraph(this->d_hexa, this->m_enginesList);
}

void MapTopology::updateDataEngineGraph(objectmodel::BaseData& my_Data, sofa::helper::list<cm_topology::TopologyEngine*>& my_enginesList)
{
	//TODO
}

MapTopology::MapTopology() :
	Inherit1(),
	d_initPoints(initData(&d_initPoints, "position", "Initial position of points")),
	d_edge(initData(&d_edge, "edges", "List of edge indices")),
	d_triangle(initData(&d_triangle, "triangles", "List of triangle indices")),
	d_quad(initData(&d_quad, "quads", "List of quad indices")),
	d_tetra(initData(&d_tetra, "tetrahedra", "List of tetrahedron indices")),
	d_hexa(initData(&d_hexa, "hexahedra", "List of hexahedron indices")),

	d_use_vertex_qt_(initData(&d_use_vertex_qt_, false, "vertices_quick_traversal", "toggle vertices quick traversal")),
	d_use_edge_qt_(initData(&d_use_edge_qt_, false, "edges_quick_traversal", "toggle edges quick traversal")),
	d_use_face_qt_(initData(&d_use_face_qt_, false, "faces_quick_traversal", "toggle faces quick traversal")),
	d_use_volume_qt_(initData(&d_use_volume_qt_, false, "volumes_quick_traversal", "toggle volumes quick traversal")),

	mech_state_(initLink("mstate", "mechanical state linked to the topology"))
{

}

MapTopology::~MapTopology()
{

}

void MapTopology::init()
{
	Inherit1::init();
}

void MapTopology::reinit()
{
	Inherit1::reinit();
}

void MapTopology::reset()
{
	Inherit1::reset();
}

void MapTopology::bwdInit()
{
	Inherit1::bwdInit();
}

} // namespace topology

} // namespace core

} // namespace sofa
