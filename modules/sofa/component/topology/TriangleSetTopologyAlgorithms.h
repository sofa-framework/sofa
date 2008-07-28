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
#ifndef SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_TRIANGLESETTOPOLOGYALGORITHMS_H

#include <sofa/component/topology/EdgeSetTopologyAlgorithms.h>

namespace sofa
{
namespace component
{
namespace topology
{
template<class DataTypes>
class TriangleSetTopology;

using core::componentmodel::topology::BaseMeshTopology;
typedef BaseMeshTopology::TriangleID TriangleID;
typedef BaseMeshTopology::Triangle Triangle;
typedef BaseMeshTopology::SeqTriangles SeqTriangles;
typedef BaseMeshTopology::VertexTriangles VertexTriangles;
typedef BaseMeshTopology::EdgeTriangles EdgeTriangles;
typedef BaseMeshTopology::TriangleEdges TriangleEdges;

/**
* A class that performs topology algorithms on an TriangleSet.
*/
template < class DataTypes >
class TriangleSetTopologyAlgorithms : public EdgeSetTopologyAlgorithms<DataTypes>
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;

    TriangleSetTopologyAlgorithms()
        :EdgeSetTopologyAlgorithms<DataTypes>()
    {}

    TriangleSetTopologyAlgorithms(sofa::core::componentmodel::topology::BaseTopology *top)
        : EdgeSetTopologyAlgorithms<DataTypes>(top)
    {}

    virtual ~TriangleSetTopologyAlgorithms() {}

    TriangleSetTopology< DataTypes >* getTriangleSetTopology() const;

    /** \brief Remove a set  of triangles
    @param triangles an array of triangle indices to be removed (note that the array is not const since it needs to be sorted)
    *
    @param removeIsolatedEdges if true isolated edges are also removed
    @param removeIsolatedPoints if true isolated vertices are also removed
    *
    */
    virtual void removeTriangles(sofa::helper::vector< unsigned int >& triangles,
            const bool removeIsolatedEdges,
            const bool removeIsolatedPoints);

    /** \brief Generic method to remove a list of items.
     */
    virtual void removeItems(sofa::helper::vector< unsigned int >& items);

    /** \brief Generic method to write the current mesh into a msh file
     */
    virtual void writeMSH(const char *filename);

    /** \brief Generic method for points renumbering
      */
    virtual void renumberPoints( const sofa::helper::vector<unsigned int> &index,
            const sofa::helper::vector<unsigned int> &inv_index);

    /** \brief  Moves and fixes the two closest points of two triangles to their median point
    */
    bool Suture2Points(unsigned int ind_ta, unsigned int ind_tb, unsigned int &ind1, unsigned int &ind2);

    /** \brief  Incises along the list of points (ind_edge,coord) intersected by the vector from point a to point b and the triangular mesh
    */
    bool InciseAlongPointsList(bool is_first_cut,
            const sofa::defaulttype::Vec<3,double>& a,
            const sofa::defaulttype::Vec<3,double>& b,
            const unsigned int ind_ta, const unsigned int ind_tb,
            unsigned int& a_last, sofa::helper::vector< unsigned int > &a_p12_last,
            sofa::helper::vector< unsigned int > &a_i123_last,
            unsigned int& b_last, sofa::helper::vector< unsigned int > &b_p12_last,
            sofa::helper::vector< unsigned int > &b_i123_last,
            sofa::helper::vector< sofa::helper::vector<unsigned int> > &new_points,
            sofa::helper::vector< sofa::helper::vector<unsigned int> > &closest_vertices);

    /** \brief Removes triangles along the list of points (ind_edge,coord) intersected by the vector from point a to point b and the triangular mesh
    */
    void RemoveAlongTrianglesList(const sofa::defaulttype::Vec<3,double>& a,
            const sofa::defaulttype::Vec<3,double>& b,
            const unsigned int ind_ta, const unsigned int ind_tb);

    /** \brief Incises along the list of points (ind_edge,coord) intersected by the sequence of input segments (list of input points) and the triangular mesh
    */
    void InciseAlongLinesList(const sofa::helper::vector< sofa::defaulttype::Vec<3,double> >& input_points,
            const sofa::helper::vector< unsigned int > &input_triangles);

    /** \brief Duplicates the given edge. Only works of at least one of its points is adjacent to a border.
     * @returns the number of newly created points, or -1 if the incision failed.
     */
    virtual int InciseAlongEdge(unsigned int edge);

};

} // namespace topology

} // namespace component

} // namespace sofa

#endif
