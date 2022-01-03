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

#include <SofaMiscTopology/config.h>

#include <SofaBaseTopology/EdgeSetTopologyContainer.h>

namespace sofa::component::topology
{

/**
 * Builds an edge topology from 2 point sets.
 *
 * Both point sets are merged and edges are created based on the minimum distance between one point to the other point set.
 */
class SOFA_SOFAMISCTOPOLOGY_API EdgeTopologyFrom2PointSets : public EdgeSetTopologyContainer
{
public:
    SOFA_CLASS(EdgeTopologyFrom2PointSets, EdgeSetTopologyContainer);

    using InitTypes = defaulttype::Vec3Types;
    using InsertionMapping = std::map< unsigned int, unsigned int >;

protected:

    /// Input
    ///@{
    Data< InitTypes::VecCoord > d_positions1;
    Data< InitTypes::VecCoord > d_positions2;

    Data< sofa::type::vector<sofa::Index> > d_indices1;
    Data< sofa::type::vector<sofa::Index> > d_indices2;

    Data< bool > d_project1on2;
    Data< bool > d_project2on1;

    ///@}

    /// List of two indices: index of the object (first or second), index of a point within this object. Could be used
    /// in association with a SubsetMultiMapping
    Data< type::vector<unsigned> > d_indexPairs;

    EdgeTopologyFrom2PointSets();
    void init() override;

    static void addPointSet(
        InitTypes::VecCoord& outPositions,
        const InitTypes::VecCoord& inPositions,
        const sofa::type::vector<sofa::Index>& indices,
        type::vector<unsigned>& indexPairs,
        InsertionMapping& insertionMapping,
        unsigned int pointSetIndex);

    void projectPointSet(
        const sofa::type::vector<sofa::Index>& indices1, const sofa::type::vector<sofa::Index>& indices2,
        const InitTypes::VecCoord& p1, const InitTypes::VecCoord& p2,
        const InsertionMapping& insertionMapping1, const InsertionMapping& insertionMapping2);
};

} //namespace sofa::component::topology