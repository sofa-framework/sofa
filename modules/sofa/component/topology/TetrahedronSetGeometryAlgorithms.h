/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETGEOMETRYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_TETRAHEDRONSETGEOMETRYALGORITHMS_H

#include <sofa/component/topology/TriangleSetGeometryAlgorithms.h>

namespace sofa
{

namespace component
{

namespace topology
{
template <class DataTypes>
class TetrahedronSetTopology;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::TetraID TetraID;
typedef BaseMeshTopology::Tetra Tetra;
typedef BaseMeshTopology::SeqTetras SeqTetras;
typedef BaseMeshTopology::VertexTetras VertexTetras;
typedef BaseMeshTopology::EdgeTetras EdgeTetras;
typedef BaseMeshTopology::TriangleTetras TriangleTetras;
typedef BaseMeshTopology::TetraEdges TetraEdges;
typedef BaseMeshTopology::TetraTriangles TetraTriangles;

typedef Tetra Tetrahedron;
typedef TetraEdges TetrahedronEdges;
typedef TetraTriangles TetrahedronTriangles;

/**
* A class that provides geometry information on an TetrahedronSet.
*/
template < class DataTypes >
class TetrahedronSetGeometryAlgorithms : public TriangleSetGeometryAlgorithms<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;

    TetrahedronSetGeometryAlgorithms()
        : TriangleSetGeometryAlgorithms<DataTypes>()
    {}

    TetrahedronSetGeometryAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top)
        : TriangleSetGeometryAlgorithms<DataTypes>(top)
    {}

    virtual ~TetrahedronSetGeometryAlgorithms() {}

    TetrahedronSetTopology< DataTypes >* getTetrahedronSetTopology() const;

    /// computes the volume of tetrahedron no i and returns it
    Real computeTetrahedronVolume(const unsigned int i) const;

    /// computes the tetrahedron volume of all tetrahedra are store in the array interface
    void computeTetrahedronVolume( BasicArrayInterface<Real> &ai) const;

    /// computes the tetrahedron volume  of tetrahedron no i and returns it
    Real computeRestTetrahedronVolume(const unsigned int i) const;

    /// finds the indices of all tetrahedra in the ball of center ind_ta and of radius dist(ind_ta, ind_tb)
    void getTetraInBall(unsigned int ind_ta, unsigned int ind_tb, sofa::helper::vector<unsigned int> &indices);

    /** \brief Write the current mesh into a msh file
    */
    void writeMSHfile(const char *filename);
};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
