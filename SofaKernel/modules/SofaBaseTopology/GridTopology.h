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
#ifndef SOFA_COMPONENT_TOPOLOGY_GRIDTOPOLOGY_H
#define SOFA_COMPONENT_TOPOLOGY_GRIDTOPOLOGY_H
#include "config.h"

#include <SofaBaseTopology/MeshTopology.h>
#include <sofa/core/DataEngine.h>
namespace sofa
{

namespace component
{

namespace topology
{


enum class Grid_dimension
{
    GRID_NULL = 0,
    GRID_1D,
    GRID_2D,
    GRID_3D
};

/** \brief
 * Define a regular grid topology, with no spatial information.
  */
class SOFA_BASE_TOPOLOGY_API GridTopology : public MeshTopology
{

public:

using MeshTopology::getQuad;
using MeshTopology::getHexahedron;

    SOFA_CLASS(GridTopology,MeshTopology);
    typedef sofa::defaulttype::Vec3i Vec3i;
    typedef sofa::defaulttype::Vector2 Vector2;
    typedef sofa::defaulttype::Vector3 Vector3;
    typedef sofa::defaulttype::ResizableExtVector<Vector2> TextCoords2D;
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
        virtual void update() override;
    protected:
        void updateEdges();
        void updateQuads();
        void updateTriangles();
        void updateHexas();
    private:
        GridTopology* topology;
    };

protected:
    /// Default constructor
    GridTopology();
    /// Constructor with grid size by int
    GridTopology(int nx, int ny, int nz);
    /// Constructor with grid size by Vec3
    GridTopology(const Vec3i& dimXYZ);

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
    virtual void init() override;

    /// BaseObject method should be overwritten by children
    virtual void reinit() override;


    /** \brief Set grid resolution in the 3 directions
     * @param nx x resolution
     * @param ny y resolution
     * @param nz z resolution
     * */
    void setSize(int nx, int ny, int nz);

    /** \brief Set grid resolution in the 3 directions, similar to \sa setSize(int nx, int ny, int nz)
     * @param Vec3i nXnYnZ resolution in 3D
     * */
    void setSize( Vec3i nXnYnZ );

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
    unsigned getIndex( int i, int j, int k ) const;

    /// Overwrite from @sa MeshTopology::hasPos always @return bool true
    virtual bool hasPos()  const override { return true; }

    /// Get Point in grid @return Vector3 given its @param id i. Will call @sa getPointInGrid. This method should be overwritten by children.
    virtual Vector3 getPoint(int i) const;

    /// Get Point in grid @return Vector3 given its position in grid @param i, @param j, @param k
    virtual Vector3 getPointInGrid(int i, int j, int k) const;

    /// get X from Point index @param i, will call @sa getPoint
    virtual SReal getPX(int i)  const override { return getPoint(i)[0]; }
    /// get Y from Point index @param i, will call @sa getPoint
    virtual SReal getPY(int i) const override { return getPoint(i)[1]; }
    /// get Z from Point index @param i, will call @sa getPoint
    virtual SReal getPZ(int i) const override { return getPoint(i)[2]; }

    /// Overload method from \sa BaseObject::parse . /// Parse the given description to assign values to this object's fields and potentially other parameters
    virtual void parse(core::objectmodel::BaseObjectDescription* arg) override
    {
        this->MeshTopology::parse(arg);

        if (arg->getAttribute("nx")!=NULL && arg->getAttribute("ny")!=NULL && arg->getAttribute("nz")!=NULL )
        {
            int nx = arg->getAttributeAsInt("nx", d_n.getValue().x());
            int ny = arg->getAttributeAsInt("ny", d_n.getValue().y());
            int nz = arg->getAttributeAsInt("nz", d_n.getValue().z());
            d_n.setValue(Vec3i(nx,ny,nz));
        }

        this->setNbGridPoints();
    }


    /// Overload Method from @sa MeshTopology::getNbHexahedra
    virtual int getNbHexahedra() override { return (d_n.getValue()[0]-1)*(d_n.getValue()[1]-1)*(d_n.getValue()[2]-1); }
    /// Overload Method from @sa MeshTopology::getQuad
    Quad getQuad(int x, int y, int z);


    Hexa getHexahedron(int x, int y, int z);
    Hexa getHexaCopy(int i);
    Quad getQuadCopy(int i);

#ifndef SOFA_NEW_HEXA
    Cube getCubeCopy(int i) { return getHexaCopy(i); }
    Cube getCube(int x, int y, int z) { return getHexahedron(x,y,z); }
#endif

    /// Get Point index in Grid, will call method @sa getIndex
    int point(int x, int y, int z) const { return getIndex(x,y,z); }
    /// Get Hexa index in Grid
    int hexa(int x, int y, int z) const { return x+(d_n.getValue()[0]-1)*(y+(d_n.getValue()[1]-1)*z); }
    /// Get Cube index, similar to \sa hexa method
    int cube(int x, int y, int z) const { return hexa(x,y,z); }

	/// Get the actual dimension of this grid using Enum @sa Grid_dimension
	Grid_dimension getDimensions() const;
public:
    /// Data storing the size of the grid in the 3 directions
    Data<Vec3i> d_n;

    /// Data bool to set option to compute topological elements
    Data<bool> d_computeHexaList, d_computeQuadList, d_computeEdgeList, d_computePointList;
    /// Data bool to set option to compute texcoords
    Data<bool> d_createTexCoords;
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
