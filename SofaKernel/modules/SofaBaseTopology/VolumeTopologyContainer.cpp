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
#include <SofaBaseTopology/VolumeTopologyContainer.h>
#include <sofa/core/ObjectFactory.h>
#include <cgogn/io/volume_import.h>
#include <cgogn/geometry/types/eigen.h>

namespace sofa
{

namespace component
{

namespace topology
{

SOFA_DECL_CLASS(VolumeTopologyContainer)
int VolumeTopologyContainerClass = core::RegisterObject("Volume topology container")
		.add< VolumeTopologyContainer >()
		;

VolumeTopologyContainer::VolumeTopologyContainer() :
	cellCache_(topology_)
{

}

VolumeTopologyContainer::~VolumeTopologyContainer()
{

}

void VolumeTopologyContainer::initFromMeshLoader()
{
	helper::ReadAccessor< Data< VecCoord > > m_position = d_initPoints;
	helper::ReadAccessor< Data< helper::vector< TetraIds > > > m_tetra = d_tetra;
	helper::ReadAccessor< Data< helper::vector< HexaIds > > > m_hexa = d_hexa;

	cgogn::io::VolumeImport<Topology/*, Eigen::Vector3d*/> volume_import(topology_);
	volume_import.reserve(m_tetra.size() + m_hexa.size());

	//auto* pos_att = volume_import.position_attribute();
	auto* pos_att = volume_import.vertex_container().template add_chunk_array<Eigen::Vector3d>("position");

	for(std::size_t i = 0ul, end = m_position.size(); i < end ; ++i)
	{
		const auto id = volume_import.insert_line_vertex_container();
		const auto& src = m_position[i];
		auto& dest = pos_att->operator [](id);
		dest[0] = src[0];
		dest[1] = src[1];
		dest[2] = src[2];
	}

	for(const TetraIds& t : m_tetra.ref())
		volume_import.add_tetra(t[0], t[1], t[2], t[3]);
	for(const HexaIds& h : m_hexa.ref())
		volume_import.add_hexa(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);

	volume_import.create_map();
}

void VolumeTopologyContainer::createTrianglesAroundEdgeArray()
{
	if (!this->m_trianglesAroundEdge.is_valid())
		this->m_trianglesAroundEdge = this->template check_attribute<TrianglesAroundEdge, Edge>("TrianglesAroundEdgeArray");
	assert(this->m_trianglesAroundEdge.is_valid());
	this->parallel_foreach_cell([&](Edge e)
	{
		auto & tris = this->m_trianglesAroundEdge[e.dart];
		foreach_incident_face(e, [this,&tris](Face f)
		{
			tris.push_back(topology_.embedding(f));
		});
	});
}

void VolumeTopologyContainer::createEdgesInQuadArray()
{
	if (!this->m_edgesInQuad.is_valid())
		this->m_edgesInQuad = this->template check_attribute<EdgesInQuad, Face>("EdgesInQuadArray");
	assert(this->m_edgesInQuad.is_valid());
	this->parallel_foreach_cell([&](Face f)
	{
		auto & edges = this->m_edgesInQuad[f.dart];
		unsigned int i = 0;
		this->foreach_incident_edge(f, [&](Edge e)
		{
			edges[i++] = topology_.embedding(e);
		});
	});
}

void VolumeTopologyContainer::createEdgesAroundVertexArray()
{
	if (!this->m_edgesAroundVertex.is_valid())
		this->m_edgesAroundVertex = this->template check_attribute<EdgesAroundVertex, Vertex>("EdgesAroundVertexArray");
	assert(this->m_edgesAroundVertex.is_valid());
	this->parallel_foreach_cell([&](Vertex v)
	{
		auto & edges = this->m_edgesAroundVertex[v.dart];
		foreach_incident_edge(v, [this,&edges](Edge e)
		{
			edges.push_back(topology_.embedding(e));
		});
	});
}

void VolumeTopologyContainer::createEdgesInTetrahedronArray()
{
	if (!this->m_edgesInTetrahedron.is_valid())
		this->m_edgesInTetrahedron = this->template check_attribute<EdgesInTetrahedron, Volume>("EdgesInTetrahedronArray");
	assert(this->m_edgesInTetrahedron.is_valid());
	this->parallel_foreach_cell([&](Volume w)
	{
		auto & edges = this->m_edgesInTetrahedron[w.dart];
		unsigned int i = 0u;
		foreach_incident_edge(w, [this,&edges,&i](Edge e)
		{
			edges[i++] = topology_.embedding(e);
		});
	});
}

void VolumeTopologyContainer::createTrianglesInTetrahedronArray()
{
	if (!this->m_trianglesInTetrahedron.is_valid())
		this->m_trianglesInTetrahedron = this->template check_attribute<TrianglesInTetrahedron, Volume>("TrianglesInTetrahedronArray");
	assert(this->m_trianglesInTetrahedron.is_valid());
	this->parallel_foreach_cell([&](Volume w)
	{
		auto & faces = this->m_trianglesInTetrahedron[w.dart];
		unsigned int i = 0u;
		foreach_incident_face(w, [this,&faces,&i](Face f)
		{
			faces[i++] = topology_.embedding(f);
		});
	});
}

void VolumeTopologyContainer::createTetrahedraAroundTriangleArray()
{
	if (!this->m_tetrahedraAroundTriangle.is_valid())
		this->m_tetrahedraAroundTriangle = this->template check_attribute<TetrahedraAroundTriangle, Face>("TetrahedraAroundTriangleArray");
	assert(this->m_tetrahedraAroundTriangle.is_valid());
	this->parallel_foreach_cell([&](Face f)
	{
		auto & tetras = this->m_tetrahedraAroundTriangle[f.dart];
		foreach_incident_volume(f, [this,&tetras](Volume w)
		{
			tetras.push_back(topology_.embedding(w));
		});
	});
}

void VolumeTopologyContainer::createHexahedraAroundVertexArray()
{
	if (!this->m_hexahedraAroundVertex.is_valid())
		this->m_hexahedraAroundVertex = this->template check_attribute<HexahedraAroundVertex, Vertex>("HexahedraAroundVertexArray");
	assert(this->m_hexahedraAroundVertex.is_valid());
	this->parallel_foreach_cell([&](Vertex v)
	{
		auto & hexahedra = this->m_hexahedraAroundVertex[v.dart];
		foreach_incident_volume(v, [this,&hexahedra](Volume w)
		{
			hexahedra.push_back(topology_.embedding(w));
		});
	});
}

void VolumeTopologyContainer::createTriangleSetArray()
{
	helper::WriteAccessor< Data< helper::vector< TriangleIds > > > m_tri = d_triangle;
	m_tri.clear();
	m_tri.reserve(this->template nb_cells<Face::ORBIT>());
	this->foreach_cell([&](Face f)
	{
		const auto& dofs = this->get_dofs(f);
		m_tri.push_back(TriangleIds(dofs[0], dofs[1], dofs[2]));
	});
}

void VolumeTopologyContainer::init()
{
	//	topology_.clear_and_remove_attributes();
	Inherit1::init();
	initFromMeshLoader();

	face_dofs_ = this->template check_attribute<helper::vector<unsigned int>, Face>("Face_dofs");
	volume_dofs_ = this->template check_attribute<helper::vector<unsigned int>, Volume>("Volume_dofs");
	assert(face_dofs_.is_valid());

	if (d_use_vertex_qt_.getValue() || d_use_edge_qt_.getValue() || d_use_face_qt_.getValue() || d_use_volume_qt_.getValue())
		qt_ = cgogn::make_unique<QuickTraversor>(topology_);

	cellCache_.build<Vertex>();
	cellCache_.build<Edge>();
	cellCache_.build<Face>();
	cellCache_.build<Volume>();

	if (d_use_vertex_qt_.getValue())
		qt_->build<Vertex>();
	if (d_use_edge_qt_.getValue())
		qt_->build<Edge>();
	if (d_use_face_qt_.getValue())
		qt_->build<Face>();
	if (d_use_volume_qt_.getValue())
		qt_->build<Volume>();

	this->parallel_foreach_cell([&](Face f)
	{
		auto& dofs = this->face_dofs_[f.dart];
		dofs.reserve(4u);
		this->foreach_incident_vertex(f,[&](Vertex v)
		{
			dofs.push_back(this->embedding(v));
		});
	});

	this->parallel_foreach_cell([&](Volume w)
	{
		auto& dofs = this->volume_dofs_[w.dart];
		dofs.reserve(8u);
		this->foreach_incident_vertex(w,[&](Vertex v)
		{
			dofs.push_back(this->embedding(v));
		});
	});
}

void VolumeTopologyContainer::bwdInit()
{
	Inherit1::bwdInit();
}

void VolumeTopologyContainer::reinit()
{
	Inherit1::reinit();
}

void VolumeTopologyContainer::reset()
{
	//	topology_.clear_and_remove_attributes();
	Inherit1::reset();
}

void VolumeTopologyContainer::cleanup()
{
	Inherit1::cleanup();
}

unsigned int VolumeTopologyContainer::getNumberOfConnectedComponent()
{
    return topology_.nb_connected_components();
}


void VolumeTopologyContainer::draw(const core::visual::VisualParams* /*vparams*/)
{
//	TriangleSetGeometryAlgorithms<DataTypes>::draw(vparams);

//    const VecCoord& coords =mech_state_->read(core::ConstVecCoordId::position())->getValue();

//    // Draw Tetra
////    if (d_drawTetrahedra.getValue())
////    {
//	if (vparams->displayFlags().getShowWireFrame())
//		vparams->drawTool()->setPolygonMode(0, true);
//	const sofa::defaulttype::Vec4f& color_tmp = d_drawColorTetrahedra.getValue();
//	defaulttype::Vec4f color4(color_tmp[0] - 0.2f, color_tmp[1] - 0.2f, color_tmp[2] - 0.2f, 1.0);

//	const sofa::helper::vector<Tetrahedron> &tetraArray = this->m_topology->getTetrahedra();
//	std::vector<defaulttype::Vector3>   pos;
//	pos.reserve(tetraArray.size()*4u);

//	for (unsigned int i = 0; i<tetraArray.size(); ++i)
//	{
//		const Tetrahedron& tet = tetraArray[i];
//		for (unsigned int j = 0u; j<4u; ++j)
//		{
//			pos.push_back(defaulttype::Vector3(DataTypes::getCPos(coords[tet[j]])));
//		}
//	}

//	const float& scale = d_drawScaleTetrahedra.getValue();

//	if (scale >= 1.0 && scale < 0.001)
//		vparams->drawTool()->drawTetrahedra(pos, color4);
//	else
//		vparams->drawTool()->drawScaledTetrahedra(pos, color4, scale);

//	if (vparams->displayFlags().getShowWireFrame())
//		vparams->drawTool()->setPolygonMode(0, false);
////    }
}

} // namespace topology

} // namespace component

} // namespace sofa
