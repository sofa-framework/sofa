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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_TOPOLOGY_MAPTOPOLOGY_H
#define SOFA_CORE_TOPOLOGY_MAPTOPOLOGY_H

#include <functional>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/State.h>

#include <cgogn/core/cmap/cmap3.h>
#include <cgogn/core/basic/dart_marker.h>
#include <cgogn/core/basic/cell_marker.h>
#include <cgogn/io/io_utils.h>


namespace sofa
{

namespace helper
{


/// ReadAccessor implementation class for cgogn attributes types
template<class T, cgogn::Orbit ORBIT>
class ReadAccessorAttribut
{
public:
    typedef T container_type;
    typedef size_t size_type;
    typedef typename container_type::value_type value_type;
    typedef value_type&				reference;
    typedef const value_type&		const_reference;
    typedef typename container_type::iterator iterator;
    typedef typename container_type::const_iterator const_iterator;

//    typedef typename T::orb_ ORBIT;

    static const cgogn::Orbit orb_ = ORBIT;

protected:
    const container_type* vref;

public:
    ReadAccessorAttribut(const container_type& container) : vref(&container) {}

    const container_type& ref() const { return *vref; }

    bool empty() const { return vref->size() == 0; }
    size_type size() const { return vref->size(); }
    const_reference operator[](cgogn::Dart c) const { return (*vref)[cgogn::Cell<ORBIT>(c)]; }

    const_iterator begin() const { return vref->begin(); }
    const_iterator end() const { return vref->end(); }

//    inline friend std::ostream& operator<< ( std::ostream& os, const ReadAccessorAttribut<T,ORBIT>& vec )
//    {
        //return os << *vec.vref;
//    }

};

template<class T, cgogn::Orbit ORBIT>
class WriteAccessorAttribut
{
public:
    typedef T container_type;
    typedef size_t size_type;
    typedef typename container_type::value_type value_type;
    typedef value_type&				reference;
    typedef const value_type&		const_reference;
    typedef typename container_type::iterator iterator;
    typedef typename container_type::const_iterator const_iterator;

    static const cgogn::Orbit orb_ = ORBIT;

protected:
    container_type* vref;

public:
    WriteAccessorAttribut(container_type& container) : vref(&container) {}

    const container_type& ref() const { return *vref; }
    container_type& wref() { return *vref; }

    bool empty() const { return vref->size() == 0; }
    size_type size() const { return vref->size(); }

    const_reference operator[](cgogn::Dart c) const { return (*vref)[ cgogn::Cell<ORBIT>(c)]; }
    reference operator[](cgogn::Dart c) { return (*vref)[cgogn::Cell<ORBIT>(c)]; }

    const_iterator begin() const { return vref->begin(); }
    iterator begin() { return vref->begin(); }
    const_iterator end() const { return vref->end(); }
    iterator end() { return vref->end(); }

    //void clear() { vref->clear(); }
    //void resize(size_type s, bool /*init*/ = true) { vref->resize(s); }
    //void reserve(size_type s) { vref->reserve(s); }
    //void push_back(const value_type& v) { vref->push_back(v); }

//    inline friend std::ostream& operator<< ( std::ostream& os, const WriteAccessorAttribut<T, ORBIT>& vec )
//    {
        //return os << *vec.vref;
//    }

//    inline friend std::istream& operator>> ( std::istream& in, WriteAccessorAttribut<T,ORBIT>& vec )
//    {
        //return in >> *vec.vref;
//    }

};

template<class T, cgogn::Orbit ORBIT>
class ReadAccessor< cgogn::Attribute<T, ORBIT>> : public ReadAccessorAttribut< cgogn::Attribute<T, ORBIT>, ORBIT >
{
public:
    using Inherit = ReadAccessorAttribut< cgogn::Attribute<T, ORBIT>, ORBIT > ;
    using container_type = typename Inherit::container_type ;
    ReadAccessor(const container_type& c) : Inherit(c) {}
};

template<class T, cgogn::Orbit ORBIT>
class WriteAccessor<cgogn::Attribute<T, ORBIT>> : public WriteAccessorAttribut< cgogn::Attribute<T, ORBIT>, ORBIT >
{
public:
    using Inherit = WriteAccessorAttribut<cgogn::Attribute<T, ORBIT>, ORBIT>;
    using container_type = typename Inherit::container_type ;
    WriteAccessor(container_type& c) : Inherit(c) {}
};

}

}









namespace sofa
{

namespace core
{

namespace cm_topology
{
// fw declaration of cm_topology::TopologyEngine
class TopologyEngine;
// fw declaration of cm_topology::TopologyChange
class TopologyChange;

template<typename T>
struct TopologyElementInfo;

}
namespace topology
{

struct CGOGN_Traits
{
	using index_type = cgogn::Dart;
	using Topology2 = cgogn::CMap2;
	using Topology3 = cgogn::CMap3;

	using Vertex2	= Topology2::Vertex;
	using Edge2		= Topology2::Edge;
	using Face2		= Topology2::Face;
	using Volume2	= Topology2::Volume;

	using Vertex3	= Topology3::Vertex;
	using Edge3		= Topology3::Edge;
	using Face3		= Topology3::Face;
	using Volume3	= Topology3::Volume;

	template<typename T>
	using Attribute_T = cgogn::Attribute_T<T>;

	struct Vertex
	{
		inline Vertex() : id_() {}
		inline Vertex(index_type id) : id_(id) {}
		index_type id_;
		inline operator index_type() const
		{
			return id_;
		}
	};

	struct Edge
	{
		inline Edge() : id_() {}
		inline Edge(index_type id) : id_(id) {}
		index_type id_;
		inline operator index_type() const
		{
			return id_;
		}
	};

	struct Face
	{
		inline Face() : id_() {}
		inline Face(index_type id) : id_(id) {}
		index_type id_;
		inline operator index_type() const
		{
			return id_;
		}
	};

	struct Volume
	{
		inline Volume() : id_() {}
		inline Volume(index_type id) : id_(id) {}
		index_type id_;
		inline operator index_type() const
		{
			return id_;
		}
	};
};


class SOFA_CORE_API MapTopology : public virtual core::objectmodel::BaseObject
{
public:
	SOFA_CLASS(MapTopology,core::objectmodel::BaseObject);

	using Topo_Traits = CGOGN_Traits;
	using Dart = cgogn::Dart;
	using Orbit = cgogn::Orbit;
	template<typename T>
	using Attribute_T = CGOGN_Traits::Attribute_T<T>;
	template <typename T, Orbit ORBIT>
	using Attribute = cgogn::Attribute<T, ORBIT>;

	using Vertex = Topo_Traits::Vertex;
	using Edge = Topo_Traits::Edge;
	using Face = Topo_Traits::Face;
	using Volume = Topo_Traits::Volume;

	using Vec3Types = sofa::defaulttype::Vec3Types;
	using VecCoord = Vec3Types::VecCoord;
	using TopologyChanger = cm_topology::TopologyChange;
	using TopologyEngine = cm_topology::TopologyEngine;
	template<typename T>
	using TopologyElementInfo = cm_topology::TopologyElementInfo<T>;


	// compatibility
	using EdgeIds = core::topology::Topology::Edge;
	using TriangleIds = core::topology::Topology::Triangle;
	using QuadIds = core::topology::Topology::Quad;
	using TetraIds = core::topology::Topology::Tetra;
	using HexaIds = core::topology::Topology::Hexa;

	using index_type = core::topology::Topology::index_type;
	using PointID = core::topology::Topology::PointID;
	using EdgeID = core::topology::Topology::EdgeID;
	using TriangleID = core::topology::Topology::TriangleID;
	using QuadID = core::topology::Topology::QuadID;
	using TetrahedronID = core::topology::Topology::TetrahedronID;
	using HexahedronID = core::topology::Topology::HexahedronID;

	using SeqEdges = core::topology::BaseMeshTopology::SeqEdges;
	using SeqTriangles = core::topology::BaseMeshTopology::SeqTriangles;
	using SeqQuads = core::topology::BaseMeshTopology::SeqQuads;
	using SeqTetrahedra = core::topology::BaseMeshTopology::SeqTetrahedra;
	using SeqHexahedra = core::topology::BaseMeshTopology::SeqHexahedra;

	using EdgesAroundVertex = core::topology::BaseMeshTopology::EdgesAroundVertex;
	using EdgesInTriangle = core::topology::BaseMeshTopology::EdgesInTriangle;
	using EdgesInQuad = core::topology::BaseMeshTopology::EdgesInQuad;
	using QuadsAroundEdge = core::topology::BaseMeshTopology::QuadsAroundEdge;
	using QuadsAroundVertex = core::topology::BaseMeshTopology::QuadsAroundVertex;
	using TrianglesInTetrahedron = core::topology::BaseMeshTopology::TrianglesInTetrahedron;
	using EdgesInHexahedron = core::topology::BaseMeshTopology::EdgesInHexahedron;
	using EdgesInTetrahedron = core::topology::BaseMeshTopology::EdgesInTetrahedron;
	using QuadsInHexahedron = core::topology::BaseMeshTopology::QuadsInHexahedron;
	using TetrahedraAroundVertex = core::topology::BaseMeshTopology::TetrahedraAroundVertex;
	using TetrahedraAroundEdge = core::topology::BaseMeshTopology::TetrahedraAroundEdge;
	using TetrahedraAroundTriangle = core::topology::BaseMeshTopology::TetrahedraAroundTriangle;
	using HexahedraAroundVertex = core::topology::BaseMeshTopology::HexahedraAroundVertex;
	using HexahedraAroundEdge = core::topology::BaseMeshTopology::HexahedraAroundEdge;
	using HexahedraAroundQuad = core::topology::BaseMeshTopology::HexahedraAroundQuad;
	using TrianglesAroundVertex = core::topology::BaseMeshTopology::TrianglesAroundVertex;
	using TrianglesAroundEdge = core::topology::BaseMeshTopology::TrianglesAroundEdge;
	using VerticesAroundVertex = core::topology::BaseMeshTopology::VerticesAroundVertex;


	using EdgeDOFs = helper::fixed_array<unsigned int, 2>;
	using FaceDOFs = helper::vector<unsigned int>;
	using VolumeDOFs = helper::vector<unsigned int>;

	MapTopology();
	~MapTopology() override;

	virtual void foreach_vertex(std::function<void(Vertex)> const &) = 0;

	virtual void foreach_edge(std::function<void(Edge)> const &) = 0;

	virtual void foreach_face(std::function<void(Face)> const &) = 0;

	virtual void foreach_volume(std::function<void(Volume)> const &) = 0;

	virtual void foreach_incident_vertex_of_edge(Edge /*edge_id*/, std::function<void(Vertex)> const & /*func*/) = 0;

	virtual void foreach_incident_vertex_of_face(Face /*face_id*/, std::function<void(Vertex)> const & /*func*/) = 0;

	virtual void foreach_incident_vertex_of_volume(Volume /*w*/, std::function<void(Vertex)> const & /*func*/) = 0;

	virtual void foreach_incident_edge_of_face(Face /*f_id*/, std::function<void(Edge)> const & /*func*/) = 0;

	virtual void foreach_incident_edge_of_volume(Volume /*vol_id*/, std::function<void(Edge)> const & /*func*/) = 0;

	virtual void foreach_incident_face_of_volume(Volume /*vol_id*/, std::function<void(Face)> const & /*func*/) = 0;


	virtual void init() override;
	virtual void bwdInit() override;
	virtual void reinit() override;
	virtual void reset() override;
	virtual void cleanup() override;

	virtual void exportMesh(const std::string& filename) = 0;

protected:
	virtual void initFromMeshLoader() = 0;

	Data< bool > d_use_vertex_qt_;
	Data< bool > d_use_edge_qt_;
	Data< bool > d_use_face_qt_;
	Data< bool > d_use_volume_qt_;

	Data< VecCoord > d_initPoints;
	Data< helper::vector< EdgeIds > > d_edge;
	Data< helper::vector< TriangleIds > > d_triangle;
	Data< helper::vector< QuadIds > > d_quad;
	Data< helper::vector< TetraIds > > d_tetra;
	Data< helper::vector< HexaIds > > d_hexa;

	SingleLink< MapTopology, core::State< Vec3Types >, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK > mech_state_;

	// compatibility
protected:
	Attribute_T<EdgesAroundVertex>			m_edgesAroundVertex;
	Attribute_T<EdgesInQuad>				m_edgesInQuad;
	Attribute_T<TrianglesAroundVertex>		m_trianglesAroundVertex;
	Attribute_T<TrianglesAroundEdge>		m_trianglesAroundEdge;
	Attribute_T<EdgesInTriangle>			m_edgesInTriangle;
	Attribute_T<QuadsAroundEdge>			m_quadsAroundEdge;
	Attribute_T<QuadsAroundVertex>			m_quadsAroundVertex;
	Attribute_T<QuadsAroundVertex>			m_orientedQuadsAroundVertex;
	Attribute_T<EdgesAroundVertex>			m_orientedEdgesAroundVertex;
	Attribute_T<TrianglesInTetrahedron>		m_trianglesInTetrahedron;
	Attribute_T<EdgesInHexahedron>			m_edgesInHexahedron;
	Attribute_T<EdgesInTetrahedron>			m_edgesInTetrahedron;
	Attribute_T<QuadsInHexahedron>			m_quadsInHexahedron;
	Attribute_T<TetrahedraAroundVertex>		m_tetrahedraAroundVertex;
	Attribute_T<TetrahedraAroundEdge>		m_tetrahedraAroundEdge;
	Attribute_T<TetrahedraAroundTriangle>	m_tetrahedraAroundTriangle;
	Attribute_T<HexahedraAroundVertex>		m_hexahedraAroundVertex;
	Attribute_T<HexahedraAroundEdge>		m_hexahedraAroundEdge;
	Attribute_T<HexahedraAroundQuad>		m_hexahedraAroundQuad;


	virtual void createTriangleSetArray() = 0;

	virtual void createEdgesAroundVertexArray() = 0;
	virtual void createEdgesInTriangleArray() = 0;
	virtual void createEdgesInQuadArray() = 0;
	virtual void createEdgesInTetrahedronArray() = 0;
	virtual void createEdgesInHexahedronArray();
	virtual void createTrianglesAroundVertexArray() = 0;
	virtual void createTrianglesAroundEdgeArray() = 0;
	virtual void createTrianglesInTetrahedronArray() = 0;
	virtual void createQuadsInHexahedronArray();
	virtual void createTetrahedraAroundTriangleArray() = 0;
	virtual void createHexahedraAroundVertexArray() = 0;
	virtual void createHexahedraAroundQuadArray();

	Attribute_T<EdgeDOFs> edge_dofs_;
	Attribute_T<FaceDOFs> face_dofs_;
	Attribute_T<VolumeDOFs> volume_dofs_;

	// compatibility
public:
//	bool hasPos() const;
//	SReal getPX(int) const;
//	SReal getPY(int) const;
//	SReal getPZ(int) const;

	virtual unsigned int getNbPoints() const = 0;
	const SeqEdges& getEdges();
	const SeqTriangles& getTriangles();
	const SeqQuads& getQuads();
	const SeqTetrahedra& getTetrahedra();
	const SeqHexahedra& getHexahedra() const;
	unsigned int getNbHexahedra() const;

	/** \brief Returns a reference to the Data of points array container. */
	inline Data<VecCoord>& getPointDataArray() {return d_initPoints;}
	/** \brief Returns a reference to the Data of edges array container. */
	inline Data< SeqEdges >& getEdgeDataArray() {return d_edge;}
	/** \brief Returns a reference to the Data of triangles array container. */
	inline Data< SeqTriangles >& getTriangleDataArray() {return d_triangle;}
	/** \brief Returns a reference to the Data of quads array container. */
	inline Data< SeqQuads >& getQuadDataArray() {return d_quad;}
	/** \brief Returns a reference to the Data of tetrahedra array container. */
	inline Data< SeqTetrahedra >& getTetrahedronDataArray() {return d_tetra;}
	/** \brief Get the Data which contains the array of hexahedra. */
	inline Data< SeqHexahedra >& getHexahedronDataArray() {return d_hexa;}

	const SeqEdges& getEdgeArray();

	const VerticesAroundVertex getVerticesAroundVertex(PointID i);
	const EdgesAroundVertex& getEdgesAroundVertex(PointID i);
	const EdgesInTriangle& getEdgesInTriangle(TriangleID i);
	const EdgesInQuad& getEdgesInQuad(QuadID i);
	const EdgesInTetrahedron& getEdgesInTetrahedron(TetrahedronID i);
	const EdgesInHexahedron& getEdgesInHexahedron(HexahedronID i);
	const TrianglesAroundVertex& getTrianglesAroundVertex(PointID i);
	const TrianglesAroundEdge& getTrianglesAroundEdge(EdgeID i);
	const TrianglesInTetrahedron& getTrianglesInTetrahedron(TetrahedronID i);
	const QuadsAroundVertex& getQuadsAroundVertex(PointID i);
	const QuadsAroundEdge& getQuadsAroundEdge(EdgeID i);
	const QuadsInHexahedron& getQuadsInHexahedron(HexahedronID i);
	const TetrahedraAroundTriangle& getTetrahedraAroundTriangle(TetrahedronID i);
	const HexahedraAroundVertex& getHexahedraAroundVertex(PointID i);
	const HexahedraAroundQuad& getHexahedraAroundQuad(QuadID i);

	const sofa::helper::vector<index_type> getElementAroundElement(index_type elem);
	const sofa::helper::vector<index_type> getElementAroundElements(sofa::helper::vector<index_type> elems);
	int getEdgeIndex(PointID v1, PointID v2);
	int getTriangleIndex(PointID v1, PointID v2, PointID v3);
	int getQuadIndex(PointID v1, PointID v2, PointID v3, PointID v4);
	int getVertexIndexInTriangle(const TriangleIds& t, PointID vertexIndex) const;
	int getEdgeIndexInTriangle(const EdgesInTriangle& t, EdgeID edgeIndex) const;
	int getVertexIndexInQuad(const QuadIds& q, PointID vertexIndex) const;
	int getEdgeIndexInQuad(const EdgesInQuad& t, EdgeID edgeIndex) const;
//	virtual void clear() override;
//	virtual void addPoint(SReal px, SReal py, SReal pz);
//	virtual void addEdge(int a, int b);
//	virtual void addTriangle(int a, int b, int c);
//	virtual void addQuad(int a, int b, int c, int d);
//	virtual bool checkConnexity();
    virtual unsigned int getNumberOfConnectedComponent() { return 0; }
	const sofa::helper::vector<index_type> getConnectedElement(index_type elem);
	void reOrientateTriangle(TriangleID id);

	// TOPOLOGY CONTAINER
public:
	virtual void addTopologyChange(const cm_topology::TopologyChange *topologyChange);
	virtual void addStateChange(const cm_topology::TopologyChange *topologyChange);
	void addTopologyEngine(cm_topology::TopologyEngine* _topologyEngine);

protected:
	void updateTopologyEngineGraph();
	/// \brief functions to really update the graph of Data/DataEngines linked to the different Data array, using member variable.
	virtual void updateDataEngineGraph(sofa::core::objectmodel::BaseData& my_Data, sofa::helper::list <sofa::core::cm_topology::TopologyEngine *>& my_enginesList);

protected:
	/// Array of topology modifications that have already occured (addition) or will occur next (deletion).
    Data <sofa::helper::list<const cm_topology::TopologyChange *> >m_changeList;

	/// Array of state modifications that have already occured (addition) or will occur next (deletion).
	Data <sofa::helper::list<const cm_topology::TopologyChange *> >m_stateChangeList;

	/// List of topology engines which will interact on all topological Data.
    sofa::helper::list<cm_topology::TopologyEngine *> m_topologyEngineList;

	sofa::helper::list <cm_topology::TopologyEngine *> m_enginesList;

};


} // namespace topology

namespace cm_topology
{

/// The enumeration used to give unique identifiers to Topological objects.
enum TopologyObjectType
{
	VERTEX,
	EDGE,
	FACE,
	VOLUME
};

template<class TopologyElement>
struct TopologyElementInfo;

template<>
struct TopologyElementInfo<topology::CGOGN_Traits::Vertex>
{
	static TopologyObjectType type() { return VERTEX; }
	static const char* name() { return "Vertex"; }
};

template<>
struct TopologyElementInfo<topology::CGOGN_Traits::Edge>
{
	static TopologyObjectType type() { return EDGE; }
	static const char* name() { return "Edge"; }
};

template<>
struct TopologyElementInfo<topology::CGOGN_Traits::Face>
{
	static TopologyObjectType type() { return FACE; }
	static const char* name() { return "Face"; }
};

template<>
struct TopologyElementInfo<topology::CGOGN_Traits::Volume>
{
	static TopologyObjectType type() { return VOLUME; }
	static const char* name() { return "Volume"; }
};

} // namespace cm_topology


} // namespace core

} // namespace sofa

#endif // SOFA_CORE_TOPOLOGY_MAPTOPOLOGY_H
