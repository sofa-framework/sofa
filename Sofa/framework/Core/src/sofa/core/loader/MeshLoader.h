/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/core/config.h>
#include <sofa/core/loader/BaseLoader.h>
#include <sofa/core/objectmodel/lifecycle/RenamedData.h>
#include <sofa/topology/Edge.h>
#include <sofa/topology/Hexahedron.h>
#include <sofa/topology/Point.h>
#include <sofa/topology/Prism.h>
#include <sofa/topology/Pyramid.h>
#include <sofa/topology/Quad.h>
#include <sofa/topology/Tetrahedron.h>
#include <sofa/topology/Triangle.h>
#include <sofa/type/PrimitiveGroup.h>

namespace sofa::helper::io {
    class Mesh;
}

namespace sofa::core::loader
{

using sofa::type::PrimitiveGroup;
using topology::Topology;

using sofa::topology::Edge;
using sofa::topology::Triangle;
using sofa::topology::Quad;
using sofa::topology::Tetrahedron;
using sofa::topology::Hexahedron;
using sofa::topology::Prism;
using sofa::topology::Pyramid;

class SOFA_CORE_API MeshLoader : public BaseLoader
{
public:
    using Vec3 = sofa::type::Vec3;

    SOFA_ABSTRACT_CLASS(MeshLoader, BaseLoader);

    typedef sofa::Index PointID;

    /* specify for each control point lying on an edge : the control point index, the index of the  edge,
     the 2 integers specifying the position within this edge (i.e. 11 for a quadratic edge, 13 within a quartic edge).. */
    typedef sofa::type::fixed_array<PointID, 4> HighOrderEdgePosition;
    /* specify for each control point lying on a triangle  : the control point index, the index of the  triangle,
     the 3 integers specifying the position within this triangle (i.e. 111 for a cubic triangle , 121 within a quartic triangle).. */
    typedef sofa::type::fixed_array<PointID, 5> HighOrderTrianglePosition;
    /* specify for each control point lying on a Quad  : the control point index, the index of the  quad,
     the 2 integers specifying the degree of the element in the x and y directions, the 2 integers specifying the position within this quad (i.e. 12 for a cubic triangle ).. */
    typedef sofa::type::fixed_array<PointID, 6> HighOrderQuadPosition;
    /* specify for each control point lying on a tetrahedron  : the control point index, the index of the  tetrahedron,
     the 3 integers specifying the position within this tetrahedron (i.e. 1111 for a quartic tetrahedron , 1211 within a quintic tetrahedron).. */
    typedef sofa::type::fixed_array<PointID, 6> HighOrderTetrahedronPosition;
    /* specify for each control point lying on a Hexahedron  : the control point index, the index of the  Hexahedron,
     the 3 integers specifying the degree of the element in the x, y and z directions, the 3 integers specifying the position within this hexahedron (i.e. 121  ).. */
    typedef sofa::type::fixed_array<PointID, 8> HighOrderHexahedronPosition;

    typedef sofa::type::vector<PointID> Polyline;


protected:
    MeshLoader();

protected:
    virtual void clearBuffers() final;

private:
    virtual bool doLoad() = 0;

    virtual void doClearBuffers() = 0;

public:
    bool canLoad() override;

    //virtual void init();
    void parse ( sofa::core::objectmodel::BaseObjectDescription* arg ) override;

    void init() override;

    void reinit() override;

    virtual bool load() final;

    /// Apply Homogeneous transformation to the positions
    virtual void applyTransformation (sofa::type::Matrix4 const& T);

    /// @name Initial transformations accessors.
    /// @{
    void setTranslation(SReal dx, SReal dy, SReal dz)
    {
        d_translation.setValue(Vec3(dx, dy, dz));
    }
    void setRotation(SReal rx, SReal ry, SReal rz)
    {
        d_rotation.setValue(Vec3(rx, ry, rz));
    }
    void setScale(SReal sx, SReal sy, SReal sz)
    {
        d_scale.setValue(Vec3(sx, sy, sz));
    }
    void setTransformation(const sofa::type::Matrix4& t)
    {
        d_transformation.setValue(t);
    }

    virtual Vec3 getTranslation() const
    {
        return d_translation.getValue();
    }
    virtual Vec3 getRotation() const
    {
        return d_rotation.getValue();
    }
    virtual Vec3 getScale() const
    {
        return d_scale.getValue();
    }
    virtual sofa::type::Matrix4 getTransformation() const
    {
        return d_transformation.getValue();
    }
    /// @}

    // Point coordinates in 3D in double.
    Data< type::vector< Vec3 > > d_positions; ///< Vertices of the mesh loaded

    //Tab of 1D elements
    Data< type::vector< Polyline > > d_polylines; ///< Polylines of the mesh loaded

    // Tab of 2D elements composition
    Data< type::vector< Edge > > d_edges; ///< Edges of the mesh loaded
    Data< type::vector< Triangle > > d_triangles; ///< Triangles of the mesh loaded
    Data< type::vector< Quad > > d_quads; ///< Quads of the mesh loaded
    Data< type::vector< type::vector<sofa::Index> > > d_polygons; ///< Polygons of the mesh loaded
    Data< type::vector< HighOrderEdgePosition > > d_highOrderEdgePositions; ///< High order edge points of the mesh loaded
    Data< type::vector< HighOrderTrianglePosition > > d_highOrderTrianglePositions; ///< High order triangle points of the mesh loaded
    Data< type::vector< HighOrderQuadPosition > > d_highOrderQuadPositions; ///< High order quad points of the mesh loaded

    // Tab of 3D elements composition
    Data< type::vector< Tetrahedron > > d_tetrahedra; ///< Tetrahedra of the mesh loaded
    Data< type::vector< Hexahedron > > d_hexahedra; ///< Hexahedra of the mesh loaded
    Data< type::vector< Prism > > d_prisms;
    SOFA_ATTRIBUTE_DEPRECATED("v25.12", "v26.06", "Pentahedron is renamed to Prism")
    objectmodel::lifecycle::RenamedData<type::vector< Prism >> d_pentahedra; ///< Pentahedra of the mesh loaded
    Data< type::vector< HighOrderTetrahedronPosition > > d_highOrderTetrahedronPositions; ///< High order tetrahedron points of the mesh loaded
    Data< type::vector< HighOrderHexahedronPosition > > d_highOrderHexahedronPositions; ///< High order hexahedron points of the mesh loaded
    Data< type::vector< Pyramid > > d_pyramids; ///< Pyramids of the mesh loaded

    // polygons in 3D ?

    //Misc
    Data< type::vector<sofa::type::Vec3 > > d_normals; ///< Normals of the mesh loaded

    // Groups
    Data< type::vector< PrimitiveGroup > > d_edgesGroups; ///< Groups of Edges
    Data< type::vector< PrimitiveGroup > > d_trianglesGroups; ///< Groups of Triangles
    Data< type::vector< PrimitiveGroup > > d_quadsGroups; ///< Groups of Quads
    Data< type::vector< PrimitiveGroup > > d_polygonsGroups; ///< Groups of Polygons
    Data< type::vector< PrimitiveGroup > > d_tetrahedraGroups; ///< Groups of Tetrahedra
    Data< type::vector< PrimitiveGroup > > d_hexahedraGroups; ///< Groups of Hexahedra
    Data< type::vector< PrimitiveGroup > > d_prismsGroups; ///< Groups of Prisms
    Data< type::vector< PrimitiveGroup > > d_pyramidsGroups; ///< Groups of Pyramids

    objectmodel::lifecycle::RenamedData<type::vector< PrimitiveGroup >> d_pentahedraGroups;

    Data< bool > d_flipNormals; ///< Flip Normals
    Data< bool > d_triangulate; ///< Divide all polygons into triangles
    Data< bool > d_createSubelements; ///< Divide all n-D elements into their (n-1)-D boundary elements (e.g. tetrahedra to triangles)
    Data< bool > d_onlyAttachedPoints; ///< Only keep points attached to elements of the mesh

    Data< Vec3 > d_translation; ///< Translation of the DOFs
    Data< Vec3 > d_rotation; ///< Rotation of the DOFs
    Data< Vec3 > d_scale; ///< Scale of the DOFs in 3 dimensions
    Data< type::Matrix4 > d_transformation; ///< 4x4 Homogeneous matrix to transform the DOFs (when present replace any)


    virtual void updateMesh();
    virtual void updateElements();
    virtual void updatePoints();
    virtual void updateNormals();

protected:

    /// to be able to call reinit w/o applying several time the same transform
    type::Matrix4 d_previousTransformation;


    void addPosition(type::vector< sofa::type::Vec3 >& pPositions, const sofa::type::Vec3& p);
    void addPosition(type::vector<sofa::type::Vec3 >& pPositions,  SReal x, SReal y, SReal z);

    void addPolyline(type::vector<Polyline>& pPolylines, Polyline p);

    void addEdge(type::vector<Edge>& pEdges, const Edge& p);
    void addEdge(type::vector<Edge>& pEdges, sofa::Index p0, sofa::Index p1);

    void addTriangle(type::vector<Triangle>& pTriangles, const Triangle& p);
    void addTriangle(type::vector<Triangle>& pTriangles, sofa::Index p0, sofa::Index p1, sofa::Index p2);

    void addQuad(type::vector<Quad>& pQuads, const Quad& p);
    void addQuad(type::vector<Quad>& pQuads, sofa::Index p0, sofa::Index p1, sofa::Index p2, sofa::Index p3);

    void addPolygon(type::vector< type::vector<sofa::Index> >& pPolygons, const type::vector<sofa::Index>& p);

    void addTetrahedron(type::vector<Tetrahedron>& pTetrahedra, const Tetrahedron& p);
    void addTetrahedron(type::vector<Tetrahedron>& pTetrahedra, sofa::Index p0, sofa::Index p1, sofa::Index p2, sofa::Index p3);

    void addHexahedron(type::vector< Hexahedron>& pHexahedra, const Hexahedron& p);
    void addHexahedron(type::vector< Hexahedron>& pHexahedra,
                       sofa::Index p0, sofa::Index p1, sofa::Index p2, sofa::Index p3,
                       sofa::Index p4, sofa::Index p5, sofa::Index p6, sofa::Index p7);

    void addPrism(type::vector< Prism>& pPrism, const Prism& p);
    void addPrism(type::vector< Prism>& pPrism,
                        sofa::Index p0, sofa::Index p1, sofa::Index p2, sofa::Index p3,
                        sofa::Index p4, sofa::Index p5);

    void addPyramid(type::vector< Pyramid>& pPyramids, const Pyramid& p);
    void addPyramid(type::vector< Pyramid>& pPyramids,
                    sofa::Index p0, sofa::Index p1, sofa::Index p2, sofa::Index p3, sofa::Index p4);

    /// Temporary method that will copy all buffers from a io::Mesh into the corresponding Data. Will be removed as soon as work on unifying meshloader is finished
    void copyMeshToData(helper::io::Mesh& _mesh);
};

} // namespace sofa::core::loader
