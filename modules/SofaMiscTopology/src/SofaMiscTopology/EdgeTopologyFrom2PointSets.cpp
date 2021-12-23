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
#include <SofaMiscTopology/EdgeTopologyFrom2PointSets.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::topology
{

int EdgeTopologyFrom2PointSetsClass = core::RegisterObject("Edge set topology container created from two point sets")
        .add< EdgeTopologyFrom2PointSets >()
        ;

EdgeTopologyFrom2PointSets::EdgeTopologyFrom2PointSets()
    : Inherit1()
    , d_positions1(initData(&d_positions1, "positions1", "List of vertices corresponding to the first point set"))
    , d_positions2(initData(&d_positions2, "positions2", "List of vertices corresponding to the second point set"))
    , d_indices1(initData(&d_indices1, "indices1", "Only indices from this list are considered in the first point set"))
    , d_indices2(initData(&d_indices2, "indices2", "Only indices from this list are considered in the second point set"))
    , d_project1on2(initData(&d_project1on2, false, "project1on2", "To create edges, project the first point set on the second based on the minimum distance"))
    , d_project2on1(initData(&d_project2on1, true, "project2on1", "To create edges, project the second point set on the first based on the minimum distance"))
    , d_indexPairs( initData( &d_indexPairs, type::vector<unsigned>(), "indexPairs", "List of two indices: index of the object (first or second), index of a point within this object. Could be used in association with a SubsetMultiMapping"))
{
    d_positions1.setGroup("Input");
    d_positions2.setGroup("Input");

    d_indices1.setGroup("Input");
    d_indices2.setGroup("Input");

    d_indexPairs.setGroup("Output");
}

void EdgeTopologyFrom2PointSets::projectPointSet(
    const sofa::type::vector<sofa::Index>& indices1, const sofa::type::vector<sofa::Index>& indices2,
    const InitTypes::VecCoord& p1, const InitTypes::VecCoord& p2,
    const InsertionMapping& insertionMapping1, const InsertionMapping& insertionMapping2)
{
    for (const auto id1 : indices1)
    {
        if (id1 < p1.size())
        {
            const auto& v1 = p1[id1];
            unsigned int closestIn2 = -1;
            {
                InitTypes::Real shortestDistance = std::numeric_limits<InitTypes::Real>::max();
                for (const auto id2 : indices2)
                {
                    if (id2 < p2.size())
                    {
                        const auto& v2 = p2[id2];
                        const auto d = (v1-v2).norm2();
                        if ( d < shortestDistance)
                        {
                            closestIn2 = id2;
                            shortestDistance = d;
                        }
                    }
                }
            }

            const auto it1 = insertionMapping1.find(id1);
            const auto it2 = insertionMapping2.find(closestIn2);
            if (it1 != insertionMapping1.end() && it2 != insertionMapping2.end())
            {
                this->addEdge(it1->second, it2->second);
            }
        }
    }
}

void EdgeTopologyFrom2PointSets::addPointSet(
    InitTypes::VecCoord& outPositions,
    const InitTypes::VecCoord& inPositions,
    const sofa::type::vector<sofa::Index>& indices,
    type::vector<unsigned>& indexPairs,
    InsertionMapping& insertionMapping,
    unsigned int pointSetIndex)
{
    for (const auto id : indices)
    {
        if (id < inPositions.size())
        {
            outPositions.push_back(inPositions[id]);
            insertionMapping[id] = outPositions.size()-1;
            indexPairs.push_back(pointSetIndex);
            indexPairs.push_back(id);
        }
    }
}

void EdgeTopologyFrom2PointSets::init()
{
    auto p = sofa::helper::getWriteOnlyAccessor(d_initPoints);
    const auto p1 = sofa::helper::getReadAccessor(d_positions1);
    const auto p2 = sofa::helper::getReadAccessor(d_positions2);
    const auto indices1 = sofa::helper::getReadAccessor(d_indices1);
    const auto indices2 = sofa::helper::getReadAccessor(d_indices2);
    auto indexPairs = sofa::helper::getWriteOnlyAccessor(d_indexPairs);

    InsertionMapping insertionMapping1;
    InsertionMapping insertionMapping2;

    addPointSet(*p, p1, indices1, *indexPairs, insertionMapping1, 0);
    addPointSet(*p, p2, indices2, *indexPairs, insertionMapping2, 1);

    if (d_project1on2.getValue())
    {
        projectPointSet(indices1, indices2, p1, p2, insertionMapping1, insertionMapping2);
    }

    if (d_project2on1.getValue())
    {
        projectPointSet(indices2, indices1, p2, p1, insertionMapping2, insertionMapping1);
    }

    EdgeSetTopologyContainer::init();
}

}//namespace sofa::component::topology