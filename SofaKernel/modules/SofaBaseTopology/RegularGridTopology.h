/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
    /// Delegated constructor
    RegularGridTopology(const Vec3i &dimXYZ);

    /// Base constructor
    RegularGridTopology();

    /// Constructor for regular grid defined using number of vertices
    RegularGridTopology(int nx, int ny, int nz);

    /// Constructor for regular grid defined using number of vertices and size
    RegularGridTopology(const Vec3i &numVertices, BoundingBox box );

    virtual void changeGridResolutionPostProcess() override;
public:
    /// BaseObject method should be overwritten by children
    virtual void init() override;

    /// BaseObject method should be overwritten by children
    virtual void reinit() override;

    /// Overload method from \sa BaseObject::parse . /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse(core::objectmodel::BaseObjectDescription* arg) override;

    /** \brief Overload method of @sa GridTopology::getPointInGrid.
     * Get Point in grid @return Vector3 given its position in grid @param i, @param j, @param k
     * */
    Vector3 getPointInGrid(int i, int j, int k) const override;


    /// set the spatial extent
    void setPos(SReal xmin, SReal xmax, SReal ymin, SReal ymax, SReal zmin, SReal zmax);
    /// set the spatial extent
    void setPos(BoundingBox box);

    /// Set the offset of the grid ( first point)
    void setP0(const Vector3& val) { d_p0 = val; }
    /// Get the offset of the grid ( first point)
    const Vector3& getP0() const { return d_p0.getValue(); }

    /// Set the distance between points in the grid
    void setDx(const Vector3& val) { dx = val; inv_dx2 = 1/(dx*dx); }
    void setDy(const Vector3& val) { dy = val; inv_dy2 = 1/(dy*dy); }
    void setDz(const Vector3& val) { dz = val; inv_dz2 = 1/(dz*dz); }

    /// Get the distance between points in the grid
    const Vector3& getDx() const { return dx; }
    const Vector3& getDy() const { return dy; }
    const Vector3& getDz() const { return dz; }


    /// Get the one-dimensional index of a grid cube, give its three-dimensional indices
    unsigned getCubeIndex( int i, int j, int k ) const;
    /// Get the position of the given cube
    Vector3 getCubeCoordinate( int i ) const;

    /// get min value of the grid bounding box @return Vector3
    Vector3   getMin() const { return d_min.getValue();}
    /// get max value of the grid bounding box @return Vector3
    Vector3   getMax() const { return d_max.getValue();}

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

    /// Overload Method of @sa GridTopology::createTexCoords called at init if @sa d_createTexCoords is true
    virtual void createTexCoords() override;

public:
    /// Data storing min and max 3D position of the grid bounding box
    Data< Vector3 > d_min, d_max;

    /// Data storing Position of point 0
    Data< Vector3 > d_p0;

    /// Data if > 0 : dimension of each cell in the created grid
    Data< SReal > d_cellWidth;

protected:
    /// Distance between points in the grid. Must be perpendicular to each other
    Vector3 dx,dy,dz;

    /// Inverse value of dx, dy and dz
    SReal inv_dx2, inv_dy2, inv_dz2;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
