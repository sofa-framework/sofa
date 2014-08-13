/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_CORE_LOADER_MESHLOADER_H
#define SOFA_CORE_LOADER_MESHLOADER_H

#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Mat.h>
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


class SOFA_CORE_API MeshLoader : public virtual BaseLoader
{
public:
    SOFA_ABSTRACT_CLASS(MeshLoader, BaseLoader);

    typedef topology::Topology::Edge Edge;
    typedef topology::Topology::Triangle Triangle;
    typedef topology::Topology::Quad Quad;
    typedef topology::Topology::Tetrahedron Tetrahedron;
    typedef topology::Topology::Hexahedron Hexahedron;

protected:
    MeshLoader();
public:
    virtual bool canLoad();

    //virtual void init();
    virtual void parse ( sofa::core::objectmodel::BaseObjectDescription* arg );

    virtual void init();

    virtual void reinit();

    /// Apply translation vector to the position.
    virtual void applyTranslation (const SReal dx, const SReal dy, const SReal dz);

    /// Apply rotation using Euler Angles in degree.
    virtual void applyRotation (const SReal rx, const SReal ry, const SReal rz);

    /// Apply rotation using quaternion.
    virtual void applyRotation (const defaulttype::Quat q);

    /// Apply Scale to the positions
    virtual void applyScale (const SReal sx, const SReal sy, const SReal sz);

    /// Apply Homogeneous transformation to the positions
    virtual void applyTransformation (sofa::defaulttype::Matrix4 const& T);

    /// @name Initial transformations accessors.
    /// @{
    void setTranslation(SReal dx, SReal dy, SReal dz) {translation.setValue(Vector3(dx,dy,dz));}
    void setRotation(SReal rx, SReal ry, SReal rz) {rotation.setValue(Vector3(rx,ry,rz));}
    void setScale(SReal sx, SReal sy, SReal sz) {scale.setValue(Vector3(sx,sy,sz));}
    void setTransformation(const sofa::defaulttype::Matrix4& t) {d_transformation.setValue(t);}

    virtual Vector3 getTranslation() const {return translation.getValue();}
    virtual Vector3 getRotation() const {return rotation.getValue();}
    virtual Vector3 getScale() const {return scale.getValue();}
    virtual sofa::defaulttype::Matrix4 getTransformation() const {return d_transformation.getValue();}
    /// @}

    // Point coordinates in 3D in double.
    Data< helper::vector<sofa::defaulttype::Vec<3,SReal> > > positions;

    // Tab of 2D elements composition
    Data< helper::vector< Edge > > edges;
    Data< helper::vector< Triangle > > triangles;
    Data< helper::vector< Quad > > quads;
    Data< helper::vector< helper::vector <unsigned int> > > polygons;

    // Tab of 3D elements composition
    Data< helper::vector< Tetrahedron > > tetrahedra;
    Data< helper::vector< Hexahedron > > hexahedra;
    // polygons in 3D ?

    //Misc
    Data< helper::vector<sofa::defaulttype::Vec<3,SReal> > > normals; /// Normals per vertex

    // Groups
    Data< helper::vector< PrimitiveGroup > > edgesGroups;
    Data< helper::vector< PrimitiveGroup > > trianglesGroups;
    Data< helper::vector< PrimitiveGroup > > quadsGroups;
    Data< helper::vector< PrimitiveGroup > > polygonsGroups;
    Data< helper::vector< PrimitiveGroup > > tetrahedraGroups;
    Data< helper::vector< PrimitiveGroup > > hexahedraGroups;

    Data< bool > flipNormals;
    Data< bool > triangulate;
    Data< bool > createSubelements;
    Data< bool > onlyAttachedPoints;

    Data< Vector3 > translation;
    Data< Vector3 > rotation;
    Data< Vector3 > scale;
    Data< sofa::defaulttype::Matrix4 > d_transformation;

protected:
    void updateMesh();
private:
    void updateElements();
    void updatePoints();
    void updateNormals();

protected:



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

};


} // namespace loader

} // namespace core

} // namespace sofa

#endif
