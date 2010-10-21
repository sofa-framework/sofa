/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_CORE_LOADER_MESHLOADER_H
#define SOFA_CORE_LOADER_MESHLOADER_H

#include <sofa/core/loader/BaseLoader.h>
#include <sofa/core/loader/PrimitiveGroup.h>
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
    SOFA_CLASS(MeshLoader, BaseLoader);

    MeshLoader();

    virtual bool canLoad();

    //virtual void init();
    virtual void parse ( sofa::core::objectmodel::BaseObjectDescription* arg );

    virtual void reinit();

    /// Apply translation vector to the position.
    virtual void applyTranslation (const SReal dx, const SReal dy, const SReal dz);

    /// Apply rotation using Euler Angles in degree.
    virtual void applyRotation (const SReal rx, const SReal ry, const SReal rz);

    /// Apply rotation using quaternion.
    virtual void applyRotation (const defaulttype::Quat q);

protected:
    void updateMesh();
private:
    void updateElements();
    void updatePoints();
    void updateNormals();

protected:

    // Point coordinates in 3D in double.
    Data< helper::vector<sofa::defaulttype::Vec<3,SReal> > > positions;

    // Tab of 2D elements composition
    Data< helper::vector< helper::fixed_array <unsigned int,2> > > edges;
    Data< helper::vector< helper::fixed_array <unsigned int,3> > > triangles;
    Data< helper::vector< helper::fixed_array <unsigned int,4> > > quads;
    Data< helper::vector< helper::vector <unsigned int> > > polygons;

    // Tab of 3D elements composition
    Data< helper::vector< helper::fixed_array<unsigned int,4> > > tetrahedra;
    Data< helper::vector< helper::fixed_array<unsigned int,8> > > hexahedra;
    // polygons in 3D ?

    //Misc
    Data< helper::vector<sofa::defaulttype::Vec<3,SReal> > > normals;

    // Groups
    Data< helper::vector< PrimitiveGroup > > edgesGroups;
    Data< helper::vector< PrimitiveGroup > > trianglesGroups;
    Data< helper::vector< PrimitiveGroup > > quadsGroups;
    Data< helper::vector< PrimitiveGroup > > polygonsGroups;
    Data< helper::vector< PrimitiveGroup > > tetrahedraGroups;
    Data< helper::vector< PrimitiveGroup > > hexahedraGroups;

    Data< bool > flipNormals;
    Data< bool > triangulate;
    Data< bool > onlyAttachedPoints;

    Data< Vector3 > translation;
    Data< Vector3 > rotation;

    void addPosition(helper::vector<sofa::defaulttype::Vec<3,SReal> >* pPositions, const sofa::defaulttype::Vec<3,SReal> &p);
    void addPosition(helper::vector<sofa::defaulttype::Vec<3,SReal> >* pPositions,  SReal x, SReal y, SReal z);

    void addEdge(helper::vector<helper::fixed_array <unsigned int,2> >* pEdges, const helper::fixed_array <unsigned int,2> &p);
    void addEdge(helper::vector<helper::fixed_array <unsigned int,2> >* pEdges, unsigned int p0, unsigned int p1);

    void addTriangle(helper::vector<helper::fixed_array <unsigned int,3> >* pTriangles, const helper::fixed_array <unsigned int,3> &p);
    void addTriangle(helper::vector<helper::fixed_array <unsigned int,3> >* pTriangles, unsigned int p0, unsigned int p1, unsigned int p2);

    void addQuad(helper::vector<helper::fixed_array <unsigned int,4> >* pQuads, const helper::fixed_array <unsigned int,4> &p);
    void addQuad(helper::vector<helper::fixed_array <unsigned int,4> >* pQuads, unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3);

    void addPolygon(helper::vector< helper::vector <unsigned int> >* pPolygons, const helper::vector<unsigned int> &p);

    void addTetrahedron(helper::vector< helper::fixed_array<unsigned int,4> >* pTetrahedra, const helper::fixed_array<unsigned int,4> &p);
    void addTetrahedron(helper::vector< helper::fixed_array<unsigned int,4> >* pTetrahedra, unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3);

    void addHexahedron(helper::vector< helper::fixed_array<unsigned int,8> >* pHexahedra, const helper::fixed_array<unsigned int,8> &p);
    void addHexahedron(helper::vector< helper::fixed_array<unsigned int,8> >* pHexahedra,
            unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3,
            unsigned int p4, unsigned int p5, unsigned int p6, unsigned int p7);

};


} // namespace loader

} // namespace core

} // namespace sofa

#endif
