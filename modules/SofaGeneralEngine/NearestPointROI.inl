/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_NearestPointROI_INL
#define SOFA_COMPONENT_ENGINE_NearestPointROI_INL

#include <SofaGeneralEngine/NearestPointROI.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/RGBAColor.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <iostream>
#include <SofaBaseTopology/TopologySubsetData.inl>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/AnimateBeginEvent.h>

namespace sofa
{

namespace component
{

namespace engine
{

using sofa::simulation::Node ;

template <class DataTypes>
NearestPointROI<DataTypes>::NearestPointROI()
    : f_indices1( initData(&f_indices1,"indices1","Indices of the points on the first model") )
    , f_indices2( initData(&f_indices2,"indices2","Indices of the points on the second model") )
    , f_radius( initData(&f_radius,(Real)1,"radius", "Radius to search corresponding fixed point") )
    , mstate1(initLink("object1", "First object to constrain"))
    , mstate2(initLink("object2", "Second object to constrain"))
{
}

template <class DataTypes>
NearestPointROI<DataTypes>::~NearestPointROI()
{
}

template <class DataTypes>
void NearestPointROI<DataTypes>::init()
{
    if(!mstate1 || !mstate2) {
        msg_error() << "Cannot Initialize without valid objects";
        //mstate1->set(static_cast<simulation::Node*>(this->getContext())->getMechanicalState());
    }

    addInput(this->mstate1->findData("rest_position"));
    addInput(this->mstate2->findData("rest_position"));
    addOutput(&f_indices1);
    addOutput(&f_indices2);

    reinit();
}

template <class DataTypes>
void NearestPointROI<DataTypes>::reinit()
{
    if(f_radius.getValue() <= 0){
        msg_error() << "Radius must be a positive real.";
        return;
    }
    if(!this->mstate1 || !this->mstate2){
        msg_error() << "2 valid mechanicalobjects are required.";
        return;
    }
    doUpdate();
}

template <class DataTypes>
void NearestPointROI<DataTypes>::doUpdate()
{
    Coord pt2;
    auto dist = [](const Coord& a, const Coord& b) { return (b - a).norm(); };
    auto cmp = [&pt2, &dist](const Coord& a, const Coord& b) {
        return dist(a, pt2) < dist(b, pt2);
    };

    auto indices1 = f_indices1.beginEdit();
    auto indices2 = f_indices2.beginEdit();
    indices1->clear();
    indices2->clear();

    const Real maxR = f_radius.getValue();
    const VecCoord& x1 = this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& x2 = this->mstate2->read(core::ConstVecCoordId::position())->getValue();

    for (unsigned int i2=0; i2<x2.size(); ++i2)
    {
        pt2 = x2[i2];
        auto el = std::min_element(std::begin(x1), std::end(x1), cmp);
        if(dist(*el, pt2) < maxR) {
            indices1->push_back(std::distance(std::begin(x1), el));
            indices2->push_back(i2);
        }
    }
    f_indices1.endEdit();
    f_indices2.endEdit();

    // Check coherency of size between indices vectors 1 and 2
    if(f_indices1.getValue().size() != f_indices2.getValue().size())
    {
        msg_error() << "Size mismatch between indices1 and indices2";
    }
}

template<class DataTypes>
void NearestPointROI<DataTypes>::handleEvent(core::objectmodel::Event *event)
{
    if (simulation::AnimateBeginEvent::checkEventType(event))
    {
        setDirtyValue();
        doUpdate();
    }
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
