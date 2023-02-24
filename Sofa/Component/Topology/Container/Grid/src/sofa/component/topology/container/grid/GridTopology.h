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

#include <sofa/component/topology/container/constant/MeshTopology.h>
#include <sofa/core/DataEngine.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::topology::container::grid
{

namespace
{
    using sofa::type::Vec2;
    using sofa::type::Vec3;
}

enum class Grid_dimension
{
    GRID_nullptr = 0,
    GRID_1D,
    GRID_2D,
    GRID_3D
};

/** \brief
 * Define a regular grid topology, with no spatial information.
  */
class SOFA_COMPONENT_TOPOLOGY_CONTAINER_GRID_API GridTopology : public container::constant::MeshTopology
{

public:

using MeshTopology::getQuad;
using MeshTopology::getHexahedron;

    SOFA_CLASS(GridTopology,MeshTopology);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vec3i, sofa::type::Vec3i);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vector2, sofa::type::Vec2);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Vector3, sofa::type::Vec3);
    typedef Vec2 TextCoords2D;
    friend class GridUpdate;

private:
    class GridUpdate : public sofa::core::DataEngine
    {
    public:
        typedef MeshTopology::Edge Edge;
        typedef MeshTopology::Quad Quad;
        typedef MeshTopology::Hexa Hexa;
        SOFA_CLASS(GridUpdate,sofa::core::DataEngine);
        GridUpdate(GridTopology* t);
        void doUpdate() override;
    protected:
        void updateEdges();
        void updateQuads();
        void updateTriangles();
        void updateHexas();
    private:
        GridTopology* m_topology;
    };

protected:
    /// Default constructor
    GridTopology();
    /// Constructor with grid size by int
    GridTopology(int nx, int ny, int nz);
    /// Constructor with grid size by Vec3
    GridTopology(const type::Vec3i& dimXYZ);

    /// Internal method to set the number of point using grid resolution. Will call \sa MeshTopology::setNbPoints
    virtual void setNbGridPoints();

    /// Method to create grid texture coordinates, should be overwritten by children
    virtual void createTexCoords(){}
    /// Method to compute Hexa list, called if \sa d_computeHexaList is true at init. Should be overwritten by children.
    virtual void computeHexaList();
    /// Method to compute Quad list, called if \sa d_computeQuadList is true at init. Should be overwritten by children.
    virtual void computeQuadList();
    /// Method to compute Edge list, called if \sa d_computeEdgeList is true at init. Should be overwritten by children.
    virtual void computeEdgeList();
    /// Method to compute Point list, called if \sa d_computePointList is true at init. Should be overwritten by children.
    virtual void computePointList();

    /// Method that will check current grid resolution, if invalide, will set default value: [2; 2; 2]
    void checkGridResolution();

    /// Internal Method called by \sa checkGridResolution if resolution need to be changed. Should be overwritten by children.
    virtual void changeGridResolutionPostProcess(){}

public:
    /// BaseObject method should be overwritten by children
    void init() override;

    /// BaseObject method should be overwritten by children
    void reinit() override;


    /** \brief Set grid resolution in the 3 directions
     * @param nx x resolution
     * @param ny y resolution
     * @param nz z resolution
     * */
    void setSize(int nx, int ny, int nz);

    /** \brief Set grid resolution in the 3 directions, similar to \sa setSize(int nx, int ny, int nz)
     * @param Vec3i nXnYnZ resolution in 3D
     * */
    void setSize( type::Vec3i nXnYnZ );

    /// Set grid X resolution, @param value
    void setNx(int value) { (*d_n.beginEdit())[0] = value; setNbGridPoints(); }
    /// Set grid Y resolution, @param value
    void setNy(int value) { (*d_n.beginEdit())[1] = value; setNbGridPoints(); }
    /// Set grid Z resolution, @param value
    void setNz(int value) { (*d_n.beginEdit())[2] = value; setNbGridPoints(); }

    /// Get X grid resolution, @return int
    int getNx() const { return d_n.getValue()[0]; }
    /// Get Y grid resolution, @return int
    int getNy() const { return d_n.getValue()[1]; }
    /// Get Z grid resolution, @return int
    int getNz() const { return d_n.getValue()[2]; }

    /// Get the one-dimensional index of a grid point given its @param i @param j @param k indices
    Index getIndex( int i, int j, int k ) const;

    /// Overwrite from @sa MeshTopology::hasPos always @return bool true
    bool hasPos()  const override { return true; }

    /// Get Point in grid @return Vec3 given its @param id i. Will call @sa getPointInGrid. This method should be overwritten by children.
    virtual Vec3 getPoint(Index i) const;

    /// Get Point in grid @return Vec3 given its position in grid @param i, @param j, @param k
    virtual Vec3 getPointInGrid(int i, int j, int k) const;

    /// get X from Point index @param i, will call @sa getPoint
    SReal getPX(Index i)  const override { return getPoint(i)[0]; }
    /// get Y from Point index @param i, will call @sa getPoint
    SReal getPY(Index i) const override { return getPoint(i)[1]; }
    /// get Z from Point index @param i, will call @sa getPoint
    SReal getPZ(Index i) const override { return getPoint(i)[2]; }

    /// Overload method from \sa BaseObject::parse . /// Parse the given description to assign values to this object's fields and potentially other parameters
    void parse(core::objectmodel::BaseObjectDescription* arg) override ;

    /// Overload Method from @sa MeshTopology::getNbHexahedra
    Size getNbHexahedra() override;

    /// Overload Method from @sa MeshTopology::getQuad
    Quad getQuad(int x, int y, int z);


    Hexa getHexahedron(int x, int y, int z);
    Hexa getHexaCopy(Index i);
    Quad getQuadCopy(Index i);

    /// Get Point index in Grid, will call method @sa getIndex
    Index point(int x, int y, int z) const { return getIndex(x,y,z); }
    /// Get Hexa index in Grid
    Index hexa(int x, int y, int z) const
    {
        const auto& n = d_n.getValue();
        return x+(n[0]-1)*(y+(n[1]-1)*z);
    }
    /// Get Cube index, similar to \sa hexa method
    Index cube(int x, int y, int z) const { return hexa(x,y,z); }

    /// Get the actual dimension of this grid using Enum @sa Grid_dimension
    Grid_dimension getDimensions() const;

public:
    /// Data storing the size of the grid in the 3 directions
    Data<type::Vec3i> d_n;

    /// Data bool to set option to compute topological elements
    Data<bool> d_computeHexaList;
    Data<bool> d_computeQuadList; ///< put true if the list of Quad is needed during init (default=true)
    Data<bool> d_computeTriangleList; ///< put true if the list of Triangles is needed during init (default=true)
    Data<bool> d_computeEdgeList; ///< put true if the list of Lines is needed during init (default=true)
    Data<bool> d_computePointList; ///< put true if the list of Points is needed during init (default=true)
    /// Data bool to set option to compute texcoords
    Data<bool> d_createTexCoords;
};

} //namespace sofa::component::topology::container::grid
