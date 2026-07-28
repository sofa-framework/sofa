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
#include <sofa/component/solidmechanics/fem/elastic/VonMisesStress.h>
#include <sofa/simulation/AnimateEndEvent.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType>
VonMisesStress<DataTypes, ElementType>::VonMisesStress()
    : d_nodalStress(initData(&d_nodalStress, sofa::type::vector<sofa::Real_t<DataTypes> >{}, "nodalStress", "Nodal von Mises stress values"))
{
    // This component must receive events
    f_listening.setValue(true);
}

template <class DataTypes, class ElementType>
void VonMisesStress<DataTypes, ElementType>::init()
{
    core::behavior::SingleStateAccessor<DataTypes>::init();

    if (!this->isComponentStateInvalid())
    {
        this->validateTopology();
    }

    if (!this->isComponentStateInvalid())
    {
        auto nodalStress = sofa::helper::getWriteOnlyAccessor(d_nodalStress);
        nodalStress->resize(this->mstate->getSize());

        // std::iota(nodalStress->begin(), nodalStress->end(), 0);
    }
}

template <class DataTypes, class ElementType>
void VonMisesStress<DataTypes, ElementType>::handleEvent(core::objectmodel::Event* event)
{
    if (simulation::AnimateEndEvent::checkEventType(event))
    {
        auto nodalStress = sofa::helper::getWriteOnlyAccessor(d_nodalStress);
        nodalStress->resize(this->mstate->getSize());

        const auto& elements = trait::FiniteElement::getElementSequence(*this->l_topology);

        for (const auto& element : elements)
        {
            for (const auto& [quadraturePoint, weight] : trait::FiniteElement::quadraturePoints())
            {
            }
        }
    }
}

}  // namespace sofa::component::solidmechanics::fem::elastic
