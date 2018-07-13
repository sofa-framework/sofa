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
#ifndef SOFA_COMPONENT_TOPOLOGY_SURFACETOPOLOGYCONTAINER_H
#define SOFA_COMPONENT_TOPOLOGY_SURFACETOPOLOGYCONTAINER_H

#include "config.h"
#include <sofa/core/topology/MapTopology.h>
#include <cgogn/io/map_import.h>
#include <cgogn/io/map_export.h>

namespace sofa
{

namespace component
{

namespace topology
{

class SOFA_BASE_TOPOLOGY_API SurfaceTopologyContainer : virtual public core::topology::MapTopology
{
public:
    SOFA_CLASS(SurfaceTopologyContainer, core::topology::MapTopology);
    using Topology = Topo_Traits::Topology2;
    using Vertex = Topo_Traits::Vertex2;
    using Edge = Topo_Traits::Edge2;
    using Face = Topo_Traits::Face2;
    using Volume = Topo_Traits::Volume2;
    using BaseVertex = Inherit1::Vertex;
    using BaseEdge   = Inherit1::Edge;
    using BaseFace   = Inherit1::Face;
    using BaseVolume = Inherit1::Volume;

    using CellCache = cgogn::CellCache<Topology>;

	using FilteredQuickTraversor = cgogn::FilteredQuickTraversor<Topology>;

	using DartMarker = cgogn::DartMarker<Topology>;
	template<Orbit ORB>
	using CellMarker = cgogn::CellMarker<Topology, ORB>;

	SurfaceTopologyContainer();
	~SurfaceTopologyContainer() override;

	virtual unsigned int getNbPoints() const override
	{
		return this->nb_cells<Vertex::ORBIT>();
	}

	template<Orbit ORBIT>
	inline unsigned int nb_cells() const
	{
		return topology_.nb_cells<ORBIT>();
	}

    virtual void foreach_vertex(const std::function<void (BaseVertex)>& func) override
    {
	topology_.foreach_cell([&](Vertex v) { func((v.dart));});
    }
    virtual void foreach_edge(const std::function<void (BaseEdge)>& func) override
    {
	topology_.foreach_cell([&](Edge e) { func((e.dart));});
    }
    virtual void foreach_face(const std::function<void (BaseFace)>& func) override
    {
	topology_.foreach_cell([&](Face f) { func((f.dart));});
    }
    virtual void foreach_volume(const std::function<void (BaseVolume)>& func) override
    {
	topology_.foreach_cell([&](Volume w) { func((w.dart));});
    }

	template<typename FUNC, typename Traversor>
	inline void foreach_cell(const FUNC& f, const Traversor& t)
    {
		topology_.foreach_cell(f, t);
    }

	template<typename FUNC>
	inline void foreach_cell(const FUNC& f)
	{
		topology_.foreach_cell(f);
	}

	template<typename FUNC>
	inline void parallel_foreach_cell(const FUNC& f)
	{
		topology_.parallel_foreach_cell(f);
	}


    virtual void foreach_incident_vertex_of_edge(BaseEdge e, const std::function<void (BaseVertex)>& func) override
    {
	topology_.foreach_incident_vertex(Edge(e.id_),[&func](Vertex v) { func(v.dart); });
    }
    template<typename FUNC>
    inline void foreach_incident_vertex(Edge e,const FUNC& f)
    {
	topology_.foreach_incident_vertex(e,f);
    }


    virtual void foreach_incident_vertex_of_face(BaseFace f, const std::function<void (BaseVertex)>& func) override
    {
	topology_.foreach_incident_vertex(Face(f.id_),[&func](Vertex v) { func(v.dart); });
    }
    template<typename FUNC>
    inline void foreach_incident_vertex(Face f,const FUNC& func)
    {
	topology_.foreach_incident_vertex(f,func);
    }


    virtual void foreach_incident_vertex_of_volume(BaseVolume w, const std::function<void (BaseVertex)>& func) override
    {
	topology_.foreach_incident_vertex(Volume(w.id_),[&func](Vertex v) { func(v.dart); });
    }
    template<typename FUNC>
    inline void foreach_incident_vertex(Volume w,const FUNC& func)
    {
	topology_.foreach_incident_vertex(w,func);
    }

    virtual void foreach_incident_edge_of_face(BaseFace f, const std::function<void (BaseEdge)>& func) override
    {
	topology_.foreach_incident_edge(Face(f.id_),[&func](Edge e) { func(e.dart); });
    }
    template<typename FUNC>
    inline void foreach_incident_edge(Face f,const FUNC& func)
    {
        topology_.foreach_incident_edge(f,func);
    }


    virtual void foreach_incident_edge_of_volume(BaseVolume w, const std::function<void (BaseEdge)>& func) override
    {
	topology_.foreach_incident_edge(Volume(w.id_),[&func](Edge e) { func(e.dart); });
    }

	template<typename FUNC>
	inline void foreach_incident_edge(Vertex v,const FUNC& func)
	{
		topology_.foreach_incident_edge(v,func);
	}


	template<typename FUNC>
	inline void foreach_incident_edge(Volume vol,const FUNC& func)
	{
		topology_.foreach_incident_edge(vol,func);
	}


    virtual void foreach_incident_face_of_volume(BaseVolume w, const std::function<void (BaseFace)>& func) override
    {
	topology_.foreach_incident_face(Volume(w.id_),[&func](Face f) { func(f.dart); });
    }
	template<typename FUNC>
	inline void foreach_incident_face(Volume vol,const FUNC& func)
	{
		topology_.foreach_incident_face(vol,func);
	}

	template<typename FUNC>
	inline void foreach_incident_face(Vertex v,const FUNC& func)
	{
		topology_.foreach_incident_face(v,func);
	}

	template<typename FUNC>
	inline void foreach_incident_face(Edge e,const FUNC& func)
	{
		topology_.foreach_incident_face(e,func);
	}

	// attributes
	template<typename T, typename CellType>
	inline Attribute<T,CellType::ORBIT> add_attribute(const std::string& attribute_name)
	{
		return topology_.add_attribute<T,CellType>(attribute_name);
	}

	template<typename T, typename CellType>
	inline Attribute<T,CellType::ORBIT> get_attribute(const std::string& attribute_name)
	{
		return topology_.get_attribute<T,CellType>(attribute_name);
	}

	template<typename T, typename CellType>
	inline Attribute<T,CellType::ORBIT> check_attribute(const std::string& attribute_name)
	{
		if (topology_.has_attribute(CellType::ORBIT,attribute_name))
			return topology_.get_attribute<T,CellType>(attribute_name);
		else
			return topology_.add_attribute<T,CellType>(attribute_name);
	}

	inline unsigned int get_dof(Vertex v)
	{
		return topology_.embedding(v);
	}

	inline const helper::fixed_array<unsigned int, 2>& get_dofs(Edge e) const
	{
		return this->edge_dofs_[e.dart];
	}

	inline const helper::vector<unsigned int>& get_dofs(Face f) const
	{
		return this->face_dofs_[f.dart];
	}

	inline Dart phi1(Dart d) const
	{
		return topology_.phi1(d);
	}

	inline Dart phi_1(Dart d) const
	{
		return topology_.phi_1(d);
	}

	inline Dart phi2(Dart d) const
	{
		return topology_.phi2(d);
	}

	inline bool isBoundaryEdge(Edge e) { return topology_.is_incident_to_boundary(e); }

public:
	virtual void init() override;
	virtual void bwdInit() override;
	virtual void reinit() override;
	virtual void reset() override;
	virtual void cleanup() override;

    unsigned int getNumberOfConnectedComponent() override;

	virtual void exportMesh(const std::string& filename) override
	{
		cgogn::io::export_surface(topology_, cgogn::io::ExportOptions::create().filename(filename).position_attribute(Vertex::ORBIT, "position"));
	}

protected:
	virtual void initFromMeshLoader() override;


	virtual void createTriangleSetArray() override;

	virtual void createEdgesInTriangleArray() override;
	virtual void createTrianglesAroundVertexArray() override;
	virtual void createTrianglesAroundEdgeArray() override;
	virtual void createEdgesAroundVertexArray() override;
	virtual void createTrianglesInTetrahedronArray() override;
	virtual void createEdgesInQuadArray() override;
	virtual void createEdgesInTetrahedronArray() override;
	virtual void createTetrahedraAroundTriangleArray() override;
	virtual void createHexahedraAroundVertexArray() override;

private:
	Topology topology_;
	std::unique_ptr<CellCache> cache_;

public:
	Topology& getMap()
	{
		return topology_;
	}
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_TOPOLOGY_SURFACETOPOLOGYCONTAINER_H
