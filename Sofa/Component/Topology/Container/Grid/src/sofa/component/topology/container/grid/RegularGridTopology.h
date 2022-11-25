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
#include <sofa/component/topology/container/grid/config.h>

#include <sofa/component/topology/container/grid/GridTopology.h>
#include <sofa/type/Vec.h>

namespace sofa::component::topology::container::grid
{

/**
  Regular grid in space.
  In c++, resolution is set in the constructor or using method setSize of the parent class, while the spatial extent is set using method setPos.
  In xml, see example files.
  */
class SOFA_COMPONENT_TOPOLOGY_CONTAINER_GRID_API RegularGridTopology : public GridTopology
{
public:
    SOFA_CLASS(RegularGridTopology,GridTopology);

    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vec3i, sofa::type::Vec3i);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vec3ui, sofa::type::Vec3u);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vector3, sofa::type::Vec3);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(BoundingBox, sofa::type::BoundingBox);

protected:
    /// Delegated constructor
    RegularGridTopology(const type::Vec3i &dimXYZ);

    /// Base constructor
    RegularGridTopology();

    /// Constructor for regular grid defined using number of vertices
    RegularGridTopology(int nx, int ny, int nz);

    /// Constructor for regular grid defined using number of vertices and size
    RegularGridTopology(const type::Vec3i &numVertices, type::BoundingBox box );

    void changeGridResolutionPostProcess() override;
public:
    /// BaseObject method should be overwritten by children
    void init() override;

    /// BaseObject method should be overwritten by children
    void reinit() override;

    /// Overload method from \sa BaseObject::parse . /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse(core::objectmodel::BaseObjectDescription* arg) override;

    /** \brief Overload method of @sa GridTopology::getPointInGrid.
     * Get Point in grid @return Vec3 given its position in grid @param i, @param j, @param k
     * */
    type::Vec3 getPointInGrid(int i, int j, int k) const override;

    /**
     * Get the index of a node located close to a given position.
     *
     * @param position The world coordinates of the queried node.
     * @return The index of the node, or -1 if no such node exists at that position.
     */
    virtual Index findPoint(const type::Vec3& position) const;

    /**
     * Get the index of a node located at a given position.
     *
     * @param position The world coordinates of the queried node.
     * @param epsilon Allows a small margin around the queried position to find the node. This value
     * is relative to the size of a cell. As an example, setting epsilon = 0.01 will return the index of the closest
     * node within a sphere radius of 1% of the cell size around the queried position. Setting this value outside of
     * [0, 1] will have no effect as only the nodes of the containing cell are taken.
     *
     * @return The index of the node, or -1 if no such node exists at that position.
     */
    virtual Index findPoint(const type::Vec3& position, SReal epsilon) const;


    /// set the spatial extent
    void setPos(SReal xmin, SReal xmax, SReal ymin, SReal ymax, SReal zmin, SReal zmax);
    /// set the spatial extent
    void setPos(type::BoundingBox box);

    /// Set the offset of the grid ( first point)
    void setP0(const type::Vec3& val) { d_p0 = val; }
    /// Get the offset of the grid ( first point)
    const type::Vec3& getP0() const { return d_p0.getValue(); }

    /// Set the distance between points in the grid
    void setDx(const type::Vec3& val) { dx = val; inv_dx2 = 1/(dx*dx); }
    void setDy(const type::Vec3& val) { dy = val; inv_dy2 = 1/(dy*dy); }
    void setDz(const type::Vec3& val) { dz = val; inv_dz2 = 1/(dz*dz); }

    /// Get the distance between points in the grid
    const type::Vec3& getDx() const { return dx; }
    const type::Vec3& getDy() const { return dy; }
    const type::Vec3& getDz() const { return dz; }


    /// Get the one-dimensional index of a grid cube, give its three-dimensional indices
    Index getCubeIndex( int i, int j, int k ) const;
    /// Get the position of the given cube
    type::Vec3 getCubeCoordinate(RegularGridTopology::Index i ) const;

    /// get min value of the grid bounding box @return Vec3
    type::Vec3   getMin() const { return d_min.getValue();}
    /// get max value of the grid bounding box @return Vec3
    type::Vec3   getMax() const { return d_max.getValue();}

    /// return the cube containing the given point (or -1 if not found).
    virtual Index findCube(const type::Vec3& pos);
    Index findHexa(const type::Vec3& pos) { return findCube(pos); }

    /// return the nearest cube (or -1 if not found).
    virtual Index findNearestCube(const type::Vec3& pos);
    Index findNearestHexa(const type::Vec3& pos) { return findNearestCube(pos); }

    /// return the cube containing the given point (or -1 if not found),
    /// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
    virtual Index findCube(const type::Vec3& pos, SReal& fx, SReal &fy, SReal &fz);
    Index findHexa(const type::Vec3& pos, SReal& fx, SReal &fy, SReal &fz) { return findCube(pos, fx, fy, fz); }

    /// return the cube containing the given point (or -1 if not found),
    /// as well as deplacements from its first corner in terms of dx, dy, dz (i.e. barycentric coordinates).
    virtual Index findNearestCube(const type::Vec3& pos, SReal& fx, SReal &fy, SReal &fz);
    Index findNearestHexa(const type::Vec3& pos, SReal& fx, SReal &fy, SReal &fz) { return findNearestCube(pos, fx, fy, fz); }

    /// Overload Method of @sa GridTopology::createTexCoords called at init if @sa d_createTexCoords is true
    void createTexCoords() override;

public:
    /// Data storing min and max 3D position of the grid bounding box
    Data< type::Vec3 > d_min;
    Data< type::Vec3 > d_max; ///< Max end of the diagonal

    /// Data storing Position of point 0
    Data< type::Vec3 > d_p0;

    /// Data if > 0 : dimension of each cell in the created grid
    Data< SReal > d_cellWidth;

protected:
    /// Distance between points in the grid. Must be perpendicular to each other
    type::Vec3 dx,dy,dz;

    /// Inverse value of dx, dy and dz
    SReal inv_dx2, inv_dy2, inv_dz2;
};

} //namespace sofa::component::topology::container::grid
