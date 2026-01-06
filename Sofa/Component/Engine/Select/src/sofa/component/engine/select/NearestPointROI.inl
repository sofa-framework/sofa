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
#include <sofa/component/engine/select/NearestPointROI.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::component::engine::select
{

template <class DataTypes>
NearestPointROI<DataTypes>::NearestPointROI(core::behavior::MechanicalState<DataTypes>* mm1, core::behavior::MechanicalState<DataTypes>* mm2)
    : Inherit1()
    , Inherit2(mm1, mm2)
    , d_inputIndices1( initData(&d_inputIndices1,"inputIndices1","Indices of the points to consider on the first model") )
    , d_inputIndices2( initData(&d_inputIndices2,"inputIndices2","Indices of the points to consider on the first model") )
    , f_radius( initData(&f_radius,(Real)1,"radius", "Radius to search corresponding fixed point") )
    , d_useRestPosition(initData(&d_useRestPosition, true, "useRestPosition", "If true will use restPosition only at init"))
    , f_indices1( initData(&f_indices1,"indices1","Indices from the first model associated to a dof from the second model") )
    , f_indices2( initData(&f_indices2,"indices2","Indices from the second model associated to a dof from the first model") )
    , d_edges(initData(&d_edges, "edges", "List of edge indices"))
    , d_indexPairs(initData(&d_indexPairs, "indexPairs", "list of couples (parent index + index in the parent)"))
    , d_distances(initData(&d_distances, "distances", "List of distances between pairs of points"))
    , d_drawPairs(initData(&d_drawPairs, false, "drawPairs", "Option to draw the positions pairs computed"))
    
{
    addOutput(&f_indices1);
    addOutput(&f_indices2);
    addOutput(&d_edges);
    addOutput(&d_indexPairs);
    addOutput(&d_distances);
}

template <class DataTypes>
NearestPointROI<DataTypes>::~NearestPointROI()
{
}

template <class DataTypes>
void NearestPointROI<DataTypes>::init()
{
    Inherit2::init();

    if (!this->mstate1 || !this->mstate2)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    const std::string dataString = d_useRestPosition.getValue() ? "rest_position" : "position";

    for (const core::behavior::BaseMechanicalState* mstate : {this->mstate1.get(), this->mstate2.get()})
    {
        if (mstate)
        {
            if (auto* mstateData = mstate->findData(dataString))
            {
                addInput(mstateData);
            }
        }
    }
}

template <class DataTypes>
void NearestPointROI<DataTypes>::reinit()
{
    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    if(f_radius.getValue() <= 0)
    {
        msg_error() << "Radius must be a positive real.";
        return;
    }

    if(!this->mstate1 || !this->mstate2)
    {
        msg_error() << "2 valid mechanicalobjects are required.";
        return;
    }

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    doUpdate();
}

template <class DataTypes>
void NearestPointROI<DataTypes>::doUpdate()
{
    if (this->mstate1 && this->mstate2)
    {
        const auto vecCoordId = d_useRestPosition.getValue() ? core::vec_id::read_access::restPosition : core::vec_id::read_access::position;
        const VecCoord& x1 = this->mstate1->read(vecCoordId)->getValue();
        const VecCoord& x2 = this->mstate2->read(vecCoordId)->getValue();

        if (x1.empty() || x2.empty())
            return;

        computeNearestPointMaps(x1, x2);
    }
}


template <class DataTypes>
void NearestPointROI<DataTypes>::computeNearestPointMaps(const VecCoord& x1, const VecCoord& x2)
{
    Coord pt2;
    constexpr auto dist = [](const Coord& a, const Coord& b) { return (b - a).norm2(); };
    const auto cmp = [&pt2, &x1, &dist](const Index a, const Index b) {
        return dist(x1[a], pt2) < dist(x1[b], pt2);
    };

    auto filterIndices1 = sofa::helper::getWriteAccessor(d_inputIndices1);
    auto filterIndices2 = sofa::helper::getWriteAccessor(d_inputIndices2);
    if (filterIndices1.empty())
    {
        filterIndices1.resize(x1.size());
        std::iota(filterIndices1.begin(), filterIndices1.end(), 0);
    }
    if (filterIndices2.empty())
    {
        filterIndices2.resize(x2.size());
        std::iota(filterIndices2.begin(), filterIndices2.end(), 0);
    }

    auto indices1 = sofa::helper::getWriteOnlyAccessor(f_indices1);
    auto indices2 = sofa::helper::getWriteOnlyAccessor(f_indices2);
    indices1->clear();
    indices2->clear();

    auto edges = sofa::helper::getWriteOnlyAccessor(d_edges);
    edges->clear();

    auto indexPairs = sofa::helper::getWriteOnlyAccessor(d_indexPairs);
    indexPairs->clear();

    auto distances = sofa::helper::getWriteOnlyAccessor(d_distances);
    distances->clear();

    const Real maxR = f_radius.getValue();
    const auto maxRSquared = maxR * maxR;

    for (const auto i2 : filterIndices2)
    {
        pt2 = x2[i2];

        //find the nearest element from pt2 in x1
        auto i1 = *std::min_element(std::begin(filterIndices1), std::end(filterIndices1), cmp);
        const auto& pt1 = x1[i1];

        const auto d = dist(pt1, pt2);
        if (d < maxRSquared)
        {
            indices1->push_back(i1);
            indices2->push_back(i2);
            edges->emplace_back(i2 * 2, i2 * 2 + 1);

            indexPairs->push_back(0);
            indexPairs->push_back(indices1->back());

            indexPairs->push_back(1);
            indexPairs->push_back(indices2->back());

            distances->push_back(std::sqrt(d));
        }
    }

    // Check coherency of size between indices vectors 1 and 2
    if (indices1.size() != indices2.size())
    {
        msg_error() << "Size mismatch between indices1 and indices2";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
}


template <class DataTypes>
void NearestPointROI<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    auto indices1 = sofa::helper::getReadAccessor(f_indices1);
    auto indices2 = sofa::helper::getReadAccessor(f_indices2);

    if (d_drawPairs.getValue() == false)
        return;

    if (!this->isComponentStateValid() || indices1.empty())
        return;

    
    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    const auto vecCoordId = d_useRestPosition.getValue() ? core::vec_id::read_access::restPosition : core::vec_id::read_access::position;
    const VecCoord& x1 = this->mstate1->read(vecCoordId)->getValue();
    const VecCoord& x2 = this->mstate2->read(vecCoordId)->getValue();
    std::vector<sofa::type::Vec3> vertices;
    vertices.reserve(indices1.size()*2);
    std::vector<sofa::type::RGBAColor> colors;
    colors.reserve(indices1.size());
    const float nbrIds = static_cast<float>(indices1.size());
    for (unsigned int i = 0; i < indices1.size(); ++i)
    {
        const auto v1 = type::toVec3(DataTypes::getCPos(x1[indices1[i]]));
        const auto v2 = type::toVec3(DataTypes::getCPos(x2[indices2[i]]));

        vertices.emplace_back(v1);
        vertices.emplace_back(v2);
        const float col = static_cast<float>(i) / nbrIds;
        colors.emplace_back(col, 1.f, 0.5f, 1.f);
    }

    vparams->drawTool()->drawLines(vertices, 1, colors);
}

} //namespace sofa::component::engine::select
