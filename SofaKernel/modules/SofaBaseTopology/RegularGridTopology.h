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
#ifndef SOFA_COMPONENT_TOPOLOGY_REGULARGRIDTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_REGULARGRIDTOPOLOGY_H
#include "config.h"

#include <SofaBaseTopology/GridTopology.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace topology
{

/**
  Regular grid in space.
  In c++, resolution is set in the constructor or using method setSize of the parent class, while the spatial extent is set using method setPos.
  In xml, see example files.
  */
class SOFA_BASE_TOPOLOGY_API RegularGridTopology : public GridTopology
{
public:
    typedef sofa::defaulttype::Vec<3, int> Vec3i;
    typedef sofa::defaulttype::Vec<3, unsigned int> Vec3ui;
    typedef sofa::defaulttype::Vector3 Vector3;
    typedef sofa::defaulttype::BoundingBox BoundingBox;
    SOFA_CLASS(RegularGridTopology,GridTopology);
protected:
    RegularGridTopology();
    /// Define using number of vertices
    RegularGridTopology(int nx, int ny, int nz);
    /// Define using number of vertices and size
    RegularGridTopology( Vec3i numVertices, BoundingBox box );
public:
    /// set the spatial extent
    void setPos(SReal xmin, SReal xmax, SReal ymin, SReal ymax, SReal zmin, SReal zmax);
    /// set the spatial extent
    void setPos(BoundingBox box);

    virtual void init();

    virtual void reinit()
    {
        setPos(min.getValue()[0],max.getValue()[0],min.getValue()[1],max.getValue()[1],min.getValue()[2],max.getValue()[2]);

        Inherit1::reinit();
    }
    void parse(core::objectmodel::BaseObjectDescription* arg);




    const Vector3& getP0() const { return p0.getValue(); }
    const Vector3& getDx() const { return dx; }
    const Vector3& getDy() const { return dy; }
    const Vector3& getDz() const { return dz; }

    unsigned getIndex( int i, int j, int k ) const; ///< one-dimensional index of a grid point
    Vector3 getPoint(int i) const;
    Vector3 getPoint(int x, int y, int z) const ;
    bool hasPos()  const { return true; }
    SReal getPX(int i)  const { return getPoint(i)[0]; }
    SReal getPY(int i) const { return getPoint(i)[1]; }
    SReal getPZ(int i) const { return getPoint(i)[2]; }


    unsigned getCubeIndex( int i, int j, int k ) const; ///< one-dimensional index of a grid cube
    Vector3 getCubeCoordinate( int i ) const; ///< from the one-dimensional index of a grid cube, give its three-dimensional indices


    Vector3   getMin() const { return min.getValue();}
    Vector3   getMax() const { return max.getValue();}


    /// return the cube containing the given point (or -1 if not found).
    virtual int findCube(const Vector3& pos);
    int findHexa(const Vector3& pos) { return findCube(pos); }

    /// return the nearest cube (or -1 if not found).
    virtual int findNearestCube(const Vector3& pos);
    int findNearestHexa(const Vector3& pos) { return findNearestCube(pos); }

    /// return the cube containing the given point (or -1 if not found),
    /// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
    virtual int findCube(const Vector3& pos, SReal& fx, SReal &fy, SReal &fz);
    int findHexa(const Vector3& pos, SReal& fx, SReal &fy, SReal &fz) { return findCube(pos, fx, fy, fz); }

    /// return the cube containing the given point (or -1 if not found),
    /// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
    virtual int findNearestCube(const Vector3& pos, SReal& fx, SReal &fy, SReal &fz);
    int findNearestHexa(const Vector3& pos, SReal& fx, SReal &fy, SReal &fz) { return findNearestCube(pos, fx, fy, fz); }

    void setP0(const Vector3& val) { p0 = val; }
    void setDx(const Vector3& val) { dx = val; inv_dx2 = 1/(dx*dx); }
    void setDy(const Vector3& val) { dy = val; inv_dy2 = 1/(dy*dy); }
    void setDz(const Vector3& val) { dz = val; inv_dz2 = 1/(dz*dz); }

    virtual void createTexCoords();

protected:

    Data<bool> computeHexaList, computeQuadList, computeEdgeList, computePointList;
    Data< Vector3 > min, max;
    /// Position of point 0
    Data< Vector3 > p0;
    Data< SReal > _cellWidth; ///< if > 0 : dimension of each cell in the created grid
    /// Distance between points in the grid. Must be perpendicular to each other
    Vector3 dx,dy,dz;
    SReal inv_dx2, inv_dy2, inv_dz2;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
