/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_LOADER_MESHLOADER_H
#define SOFA_CORE_LOADER_MESHLOADER_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/core/loader/BaseLoader.h>
#include <sofa/core/loader/PrimitiveGroup.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/helper/fixed_array.h>


namespace sofa
{

namespace core
{

namespace loader
{

using sofa::defaulttype::Vector3;


class SOFA_CORE_API MeshLoader : public BaseLoader
{
public:
    SOFA_ABSTRACT_CLASS(MeshLoader, BaseLoader);

    typedef topology::Topology::Edge Edge;
    typedef topology::Topology::Triangle Triangle;
    typedef topology::Topology::Quad Quad;
    typedef topology::Topology::Tetrahedron Tetrahedron;
    typedef topology::Topology::Hexahedron Hexahedron;
    typedef topology::Topology::Pentahedron Pentahedron;
    typedef topology::Topology::Pyramid Pyramid;
    typedef topology::Topology::PointID PointID;
    /* specify for each control point lying on an edge : the control point index, the index of the  edge, 
     the 2 integers specifying the position within this edge (i.e. 11 for a quadratic edge, 13 within a quartic edge).. */
    typedef sofa::helper::fixed_array<PointID,4> HighOrderEdgePosition;
    /* specify for each control point lying on a triangle  : the control point index, the index of the  triangle, 
     the 3 integers specifying the position within this triangle (i.e. 111 for a cubic triangle , 121 within a quartic triangle).. */
    typedef sofa::helper::fixed_array<PointID,5> HighOrderTrianglePosition;
    /* specify for each control point lying on a Quad  : the control point index, the index of the  quad, 
     the 2 integers specifying the degree of the element in the x and y directions, the 2 integers specifying the position within this quad (i.e. 12 for a cubic triangle ).. */
    typedef sofa::helper::fixed_array<PointID,6> HighOrderQuadPosition;
    /* specify for each control point lying on a tetrahedron  : the control point index, the index of the  tetrahedron, 
     the 3 integers specifying the position within this tetrahedron (i.e. 1111 for a quartic tetrahedron , 1211 within a quintic tetrahedron).. */
    typedef sofa::helper::fixed_array<PointID,6> HighOrderTetrahedronPosition;
    /* specify for each control point lying on a Hexahedron  : the control point index, the index of the  Hexahedron, 
     the 3 integers specifying the degree of the element in the x, y and z directions, the 3 integers specifying the position within this hexahedron (i.e. 121  ).. */
    typedef sofa::helper::fixed_array<PointID,8> HighOrderHexahedronPosition;



protected:
    MeshLoader();
public:
    virtual bool canLoad();

    //virtual void init();
    virtual void parse ( sofa::core::objectmodel::BaseObjectDescription* arg );

    virtual void init();

    virtual void reinit();


    /// Apply Homogeneous transformation to the positions
    virtual void applyTransformation (sofa::defaulttype::Matrix4 const& T);

    /// @name Initial transformations accessors.
    /// @{
    void setTranslation(SReal dx, SReal dy, SReal dz) {d_translation.setValue(Vector3(dx,dy,dz));}
    void setRotation(SReal rx, SReal ry, SReal rz) {d_rotation.setValue(Vector3(rx,ry,rz));}
    void setScale(SReal sx, SReal sy, SReal sz) {d_scale.setValue(Vector3(sx,sy,sz));}
    void setTransformation(const sofa::defaulttype::Matrix4& t) {d_transformation.setValue(t);}

    virtual Vector3 getTranslation() const {return d_translation.getValue();}
    virtual Vector3 getRotation() const {return d_rotation.getValue();}
    virtual Vector3 getScale() const {return d_scale.getValue();}
    virtual sofa::defaulttype::Matrix4 getTransformation() const {return d_transformation.getValue();}
    /// @}

    // Point coordinates in 3D in double.
    Data< helper::vector<sofa::defaulttype::Vec<3,SReal> > > d_positions;

    // Tab of 2D elements composition
    Data< helper::vector< Edge > > d_edges;
    Data< helper::vector< Triangle > > d_triangles;
    Data< helper::vector< Quad > > d_quads;
    Data< helper::vector< helper::vector <unsigned int> > > d_polygons;
    Data< helper::vector< HighOrderEdgePosition > > d_highOrderEdgePositions;
    Data< helper::vector< HighOrderTrianglePosition > > d_highOrderTrianglePositions;
    Data< helper::vector< HighOrderQuadPosition > > d_highOrderQuadPositions;

    // Tab of 3D elements composition
    Data< helper::vector< Tetrahedron > > d_tetrahedra;
    Data< helper::vector< Hexahedron > > d_hexahedra;
    Data< helper::vector< Pentahedron > > d_pentahedra;
    Data< helper::vector< Pyramid > > d_pyramids;
    Data< helper::vector< HighOrderTetrahedronPosition > > d_highOrderTetrahedronPositions;
    Data< helper::vector< HighOrderHexahedronPosition > > d_highOrderHexahedronPositions;

    // polygons in 3D ?

    //Misc
    Data< helper::vector<sofa::defaulttype::Vec<3,SReal> > > d_normals; /// Normals per vertex

    // Groups
    Data< helper::vector< PrimitiveGroup > > d_edgesGroups;
    Data< helper::vector< PrimitiveGroup > > d_trianglesGroups;
    Data< helper::vector< PrimitiveGroup > > d_quadsGroups;
    Data< helper::vector< PrimitiveGroup > > d_polygonsGroups;
    Data< helper::vector< PrimitiveGroup > > d_tetrahedraGroups;
    Data< helper::vector< PrimitiveGroup > > d_hexahedraGroups;
    Data< helper::vector< PrimitiveGroup > > d_pentahedraGroups;
    Data< helper::vector< PrimitiveGroup > > d_pyramidsGroups;

    Data< bool > d_flipNormals;
    Data< bool > d_triangulate;
    Data< bool > d_createSubelements;
    Data< bool > d_onlyAttachedPoints;

    Data< Vector3 > d_translation;
    Data< Vector3 > d_rotation;
    Data< Vector3 > d_scale;
    Data< defaulttype::Matrix4 > d_transformation;


   virtual void updateMesh();
   virtual void updateElements();
   virtual void updatePoints();
   virtual void updateNormals();

protected:

    /// to be able to call reinit w/o applying several time the same transform
    defaulttype::Matrix4 d_previousTransformation;


    void addPosition(helper::vector< sofa::defaulttype::Vec<3,SReal> >* pPositions, const sofa::defaulttype::Vec<3,SReal> &p);
    void addPosition(helper::vector<sofa::defaulttype::Vec<3,SReal> >* pPositions,  SReal x, SReal y, SReal z);

    void addEdge(helper::vector<Edge>* pEdges, const Edge &p);
    void addEdge(helper::vector<Edge>* pEdges, unsigned int p0, unsigned int p1);

    void addTriangle(helper::vector<Triangle>* pTriangles, const Triangle &p);
    void addTriangle(helper::vector<Triangle>* pTriangles, unsigned int p0, unsigned int p1, unsigned int p2);

    void addQuad(helper::vector<Quad>* pQuads, const Quad &p);
    void addQuad(helper::vector<Quad>* pQuads, unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3);

    void addPolygon(helper::vector< helper::vector <unsigned int> >* pPolygons, const helper::vector<unsigned int> &p);

    void addTetrahedron(helper::vector<Tetrahedron>* pTetrahedra, const Tetrahedron &p);
    void addTetrahedron(helper::vector<Tetrahedron>* pTetrahedra, unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3);

    void addHexahedron(helper::vector< Hexahedron>* pHexahedra, const Hexahedron &p);
    void addHexahedron(helper::vector< Hexahedron>* pHexahedra,
            unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3,
            unsigned int p4, unsigned int p5, unsigned int p6, unsigned int p7);

    void addPentahedron(helper::vector< Pentahedron>* pPentahedra, const Pentahedron &p);
    void addPentahedron(helper::vector< Pentahedron>* pPentahedra,
            unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3,
            unsigned int p4, unsigned int p5);

    void addPyramid(helper::vector< Pyramid>* pPyramids, const Pyramid &p);
    void addPyramid(helper::vector< Pyramid>* pPyramids,
            unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3, unsigned int p4);
};


} // namespace loader

} // namespace core

} // namespace sofa

#endif
