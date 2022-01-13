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
#include <SofaGeneralEngine/NearestPointROI.h>

namespace sofa::component::engine
{

template <class DataTypes>
NearestPointROI<DataTypes>::NearestPointROI()
    : Inherit1()
    , Inherit2(nullptr, nullptr)
    , f_indices1( initData(&f_indices1,"indices1","Indices of the points on the first model") )
    , f_indices2( initData(&f_indices2,"indices2","Indices of the points on the second model") )
    , f_radius( initData(&f_radius,(Real)1,"radius", "Radius to search corresponding fixed point") )
    , d_useRestPosition(initData(&d_useRestPosition, true, "useRestPosition", "If true will use restPosition only at init"))
{

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

    if (d_useRestPosition.getValue())
    {
        addInput(this->mstate1->findData("rest_position"));
        addInput(this->mstate2->findData("rest_position"));
    }
    else
    {
        addInput(this->mstate1->findData("position"));
        addInput(this->mstate2->findData("position"));
    }

    addOutput(&f_indices1);
    addOutput(&f_indices2);
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
    const auto vecCoordId = d_useRestPosition.getValue() ? core::ConstVecCoordId::restPosition() : core::ConstVecCoordId::position();
    const VecCoord& x1 = this->mstate1->read(vecCoordId)->getValue();
    const VecCoord& x2 = this->mstate2->read(vecCoordId)->getValue();

    if (x1.empty() || x2.empty())
        return;

    computeNearestPointMaps(x1, x2);
}


template <class DataTypes>
void NearestPointROI<DataTypes>::computeNearestPointMaps(const VecCoord& x1, const VecCoord& x2)
{
    Coord pt2;
    constexpr auto dist = [](const Coord& a, const Coord& b) { return (b - a).norm(); };
    constexpr auto cmp = [&pt2, &dist](const Coord& a, const Coord& b) {
        return dist(a, pt2) < dist(b, pt2);
    };

    auto indices1 = sofa::helper::getWriteOnlyAccessor(f_indices1);
    auto indices2 = sofa::helper::getWriteOnlyAccessor(f_indices2);
    indices1->clear();
    indices2->clear();

    const Real maxR = f_radius.getValue();

    for (unsigned int i2 = 0; i2 < x2.size(); ++i2)
    {
        pt2 = x2[i2];
        auto el = std::min_element(std::begin(x1), std::end(x1), cmp);
        if (dist(*el, pt2) < maxR) 
        {
            indices1->push_back(std::distance(std::begin(x1), el));
            indices2->push_back(i2);
        }
    }

    // Check coherency of size between indices vectors 1 and 2
    if (indices1.size() != indices2.size())
    {
        msg_error() << "Size mismatch between indices1 and indices2";
    }
}

} //namespace sofa::component::engine
