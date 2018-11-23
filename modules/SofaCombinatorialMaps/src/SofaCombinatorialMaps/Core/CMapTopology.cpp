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
#include <SofaCombinatorialMaps/Core/CMapTopology.h>
#include <SofaCombinatorialMaps/Core/CMBaseTopologyEngine.h>

namespace sofa
{

namespace core
{

namespace topology
{

void CMapTopology::cleanup()
{
	Inherit1::cleanup();
}

void CMapTopology::createHexahedraAroundVertexArray()
{
	serr << "createHexahedraAroundVertexArray not implemented" << sendl;
}

void CMapTopology::createHexahedraAroundQuadArray()
{
	serr << "createHexahedraAroundQuadArray not implemented" << sendl;
}

void CMapTopology::createEdgesInHexahedronArray()
{
	serr << "createEdgesInHexahedronArray not implemented" << sendl;
}

void CMapTopology::createQuadsInHexahedronArray()
{
	serr << "createQuadsInHexahedronArray not implemented" << sendl;
}

const BaseMeshTopology::SeqEdges& CMapTopology::getEdges()
{
	return d_edge.getValue();
}

const BaseMeshTopology::SeqTriangles& CMapTopology::getTriangles()
{
	if (d_triangle.getValue().empty())
		createTriangleSetArray();
	return d_triangle.getValue();
}

const BaseMeshTopology::SeqQuads& CMapTopology::getQuads()
{
	return d_quad.getValue();
}

const BaseMeshTopology::SeqTetrahedra& CMapTopology::getTetrahedra()
{
	return d_tetra.getValue();
}

const BaseMeshTopology::SeqHexahedra& CMapTopology::getHexahedra() const
{
	return d_hexa.getValue();
}

unsigned int CMapTopology::getNbHexahedra() const
{
	return getHexahedra().size();
}

const BaseMeshTopology::EdgesAroundVertex& CMapTopology::getEdgesAroundVertex(Topology::PointID i)
{
	if (!m_edgesAroundVertex.is_valid())
		createEdgesAroundVertexArray();
	return m_edgesAroundVertex[i];
}

const BaseMeshTopology::EdgesInTriangle& CMapTopology::getEdgesInTriangle(Topology::TriangleID i)
{
	const static EdgesInTriangle empty;
	if (!m_edgesInTriangle.is_valid())
		createEdgesInTriangleArray();
	return m_edgesInTriangle.is_valid() ? m_edgesInTriangle[i] : empty;

}

const BaseMeshTopology::EdgesInQuad& CMapTopology::getEdgesInQuad(Topology::QuadID i)
{
	const static EdgesInQuad empty;
	if (!m_edgesInQuad.is_valid())
		createEdgesInQuadArray();
	return m_edgesInQuad.is_valid() ? m_edgesInQuad[i] : empty;
}

const BaseMeshTopology::TrianglesAroundVertex& CMapTopology::getTrianglesAroundVertex(Topology::PointID i)
{
    return m_trianglesAroundVertex[i];
}

const BaseMeshTopology::TrianglesAroundEdge& CMapTopology::getTrianglesAroundEdge(Topology::EdgeID i)
{
    return m_trianglesAroundEdge[i];
}

const BaseMeshTopology::QuadsAroundVertex& CMapTopology::getQuadsAroundVertex(Topology::PointID i)
{
	return m_quadsAroundVertex[i];
}

const BaseMeshTopology::QuadsAroundEdge& CMapTopology::getQuadsAroundEdge(Topology::EdgeID i)
{
	return m_quadsAroundEdge[i];
}

const BaseMeshTopology::VerticesAroundVertex CMapTopology::getVerticesAroundVertex(Topology::PointID i)
{
//	return m_ver[i];
	return VerticesAroundVertex();
}

const CMapTopology::TrianglesInTetrahedron& CMapTopology::getTrianglesInTetrahedron(CMapTopology::TetrahedronID i)
{
	const static TrianglesInTetrahedron empty;
	if (!m_trianglesInTetrahedron.is_valid())
		createTrianglesInTetrahedronArray();
	return m_trianglesInTetrahedron.is_valid()? m_trianglesInTetrahedron[i] : empty;
}

const CMapTopology::EdgesInTetrahedron& CMapTopology::getEdgesInTetrahedron(CMapTopology::TetrahedronID i)
{
	const static EdgesInTetrahedron empty;
	if (!m_edgesInTetrahedron.is_valid())
		createEdgesInTetrahedronArray();
	return m_edgesInTetrahedron.is_valid()? m_edgesInTetrahedron[i] : empty;
}

const CMapTopology::TetrahedraAroundTriangle& CMapTopology::getTetrahedraAroundTriangle(CMapTopology::TetrahedronID i)
{
	const static TetrahedraAroundTriangle empty;
	if (!m_tetrahedraAroundTriangle.is_valid())
		createTetrahedraAroundTriangleArray();
	return m_tetrahedraAroundTriangle.is_valid()? m_tetrahedraAroundTriangle[i] : empty;
}

const CMapTopology::HexahedraAroundVertex& CMapTopology::getHexahedraAroundVertex(CMapTopology::PointID i)
{
	const static HexahedraAroundVertex empty;
	if (!m_hexahedraAroundVertex.is_valid())
		createHexahedraAroundVertexArray();
	return m_hexahedraAroundVertex.is_valid()? m_hexahedraAroundVertex[i] : empty;
}

const CMapTopology::HexahedraAroundQuad& CMapTopology::getHexahedraAroundQuad(CMapTopology::QuadID i)
{
	const static HexahedraAroundQuad empty;
	if (!m_hexahedraAroundQuad.is_valid())
		createHexahedraAroundQuadArray();
	return m_hexahedraAroundQuad.is_valid()? m_hexahedraAroundQuad[i] : empty;
}

const CMapTopology::EdgesInHexahedron& CMapTopology::getEdgesInHexahedron(CMapTopology::HexahedronID i)
{
	const static EdgesInHexahedron empty;
	if (!m_edgesInHexahedron.is_valid())
		createEdgesInHexahedronArray();
	return m_edgesInHexahedron.is_valid()? m_edgesInHexahedron[i] : empty;
}

const CMapTopology::QuadsInHexahedron& CMapTopology::getQuadsInHexahedron(CMapTopology::HexahedronID i)
{
	const static QuadsInHexahedron empty;
	if (!m_quadsInHexahedron.is_valid())
		createQuadsInHexahedronArray();
	return m_quadsInHexahedron.is_valid()? m_quadsInHexahedron[i] : empty;
}

const sofa::helper::vector<Topology::index_type> CMapTopology::getElementAroundElement(Topology::index_type elem)
{
	return sofa::helper::vector<Topology::index_type>();
}

const sofa::helper::vector<Topology::index_type> CMapTopology::getElementAroundElements(sofa::helper::vector<Topology::index_type> elems)
{
	return sofa::helper::vector<Topology::index_type>();
}

int CMapTopology::getEdgeIndex(Topology::PointID v1, Topology::PointID v2)
{
	return -1;
}

int CMapTopology::getTriangleIndex(Topology::PointID v1, Topology::PointID v2, Topology::PointID v3)
{
	return -1;
}

int CMapTopology::getQuadIndex(Topology::PointID v1, Topology::PointID v2, Topology::PointID v3, Topology::PointID v4)
{
	return -1;
}

int CMapTopology::getVertexIndexInTriangle(const TriangleIds& t, Topology::PointID vertexIndex) const
{
    return -1;
}

int CMapTopology::getEdgeIndexInTriangle(const BaseMeshTopology::EdgesInTriangle& t, Topology::EdgeID edgeIndex) const
{
	return -1;
}

int CMapTopology::getVertexIndexInQuad(const Topology::Quad& t, Topology::PointID vertexIndex) const
{
	return -1;
}

int CMapTopology::getEdgeIndexInQuad(const BaseMeshTopology::EdgesInQuad& t, Topology::EdgeID edgeIndex) const
{
	return -1;
}

const sofa::helper::vector<Topology::index_type> CMapTopology::getConnectedElement(Topology::index_type elem)
{
	return sofa::helper::vector<Topology::index_type> ();
}

void CMapTopology::reOrientateTriangle(Topology::TriangleID id)
{

}

void CMapTopology::addTopologyChange(const cm_topology::TopologyChange* topologyChange)
{
	auto& my_changeList = *(m_changeList.beginEdit());
	my_changeList.push_back(topologyChange);
	m_changeList.endEdit();
}

void CMapTopology::addStateChange(const cm_topology::TopologyChange* topologyChange)
{
	auto& my_stateChangeList = *(m_stateChangeList.beginEdit());
	my_stateChangeList.push_back(topologyChange);
	m_stateChangeList.endEdit();
}

void CMapTopology::addTopologyEngine(cm_topology::TopologyEngine* _topologyEngine)
{
	m_topologyEngineList.push_back(_topologyEngine);
	m_topologyEngineList.back()->m_changeList.setParent(&this->m_changeList);
	this->updateTopologyEngineGraph();
}

void CMapTopology::updateTopologyEngineGraph()
{
	this->updateDataEngineGraph(this->d_hexa, this->m_enginesList);
}

void CMapTopology::updateDataEngineGraph(objectmodel::BaseData& my_Data, std::list<cm_topology::TopologyEngine*>& my_enginesList)
{
	//TODO
}

CMapTopology::CMapTopology() :
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

CMapTopology::~CMapTopology()
{

}

void CMapTopology::init()
{
	Inherit1::init();
}

void CMapTopology::reinit()
{
	Inherit1::reinit();
}

void CMapTopology::reset()
{
	Inherit1::reset();
}

void CMapTopology::bwdInit()
{
	Inherit1::bwdInit();
}

} // namespace topology

} // namespace core

} // namespace sofa
