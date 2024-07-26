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

#include <sofa/component/solidmechanics/spring/BaseRestShapeSpringsForceField.h>
#include <sofa/type/isRigidType.h>

namespace sofa::component::solidmechanics::spring {

template<class DataTypes>
BaseRestShapeSpringsForceField<DataTypes>::BaseRestShapeSpringsForceField()
        : d_indices(initData(&d_indices, "indices", "points controlled by the rest shape springs"))
        , d_stiffness(initData(&d_stiffness, {1.0},"stiffness", "stiffness value between the actual position and the rest shape position"))
        , d_angularStiffness(initData(&d_angularStiffness,{1.0}, "angularStiffness", "angularStiffness assigned when controlling the rotation of the points"))
        , d_drawSpring(initData(&d_drawSpring,false,"drawSpring","draw Spring"))
        , d_springColor(initData(&d_springColor, sofa::type::RGBAColor::green(), "springColor","spring color. (default=[0.0,1.0,0.0,1.0])"))
        , l_topology(initLink("topology", "Link to be set to the topology container in the component graph"))
{
    this->addUpdateCallback("updateInputs", {&d_indices}, [this](const core::DataTracker& t)
    {
        SOFA_UNUSED(t);
        if(this->checkOutOfBoundsIndices())
        {
            msg_error() << "Input indices out of bound";
            return sofa::core::objectmodel::ComponentState::Invalid;
        }

        return sofa::core::objectmodel::ComponentState::Valid;
    }, {});
}

template<class DataTypes>
void BaseRestShapeSpringsForceField<DataTypes>::init()
{
    if(d_stiffness.getValue().size() != d_indices.getValue().size())
    {
        msg_warning() << "Input stiffness isn't the same size as indices. Either shrinking it or copying the first value to match indices size.";
        helper::WriteAccessor< Data<VecReal> > stiffness = d_stiffness;
        const unsigned oldSize = stiffness.size();
        stiffness.resize(d_indices.getValue().size());
        for(unsigned i=oldSize; i< stiffness.size(); ++i)
        {
            stiffness[i] = stiffness[0];
        }
    }
    if constexpr (sofa::type::isRigidType<DataTypes>())
    {
        if (d_angularStiffness.getValue().size() != d_indices.getValue().size())
        {
            msg_warning() << "Input angular stiffness isn't the same size as indices. Either shrinking it or copying the first value to match indices size.";
            helper::WriteAccessor<Data<VecReal> > stiffness = d_angularStiffness;
            const unsigned oldSize = stiffness.size();
            stiffness.resize(d_indices.getValue().size());
            for(unsigned i=oldSize; i< stiffness.size(); ++i)
            {
                stiffness[i] = stiffness[0];
            }
        }
    }
}

}