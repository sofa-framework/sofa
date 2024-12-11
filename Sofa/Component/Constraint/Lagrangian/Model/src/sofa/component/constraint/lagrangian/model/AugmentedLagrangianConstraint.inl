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
#include <sofa/component/constraint/lagrangian/model/AugmentedLagrangianConstraint.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/Vec.h>
#include <sofa/type/RGBAColor.h>

namespace sofa::component::constraint::lagrangian::model
{

template<class DataTypes>
AugmentedLagrangianConstraint<DataTypes>::AugmentedLagrangianConstraint(MechanicalState* object1, MechanicalState* object2)
    : Inherit(object1, object2)
{
}

template<class DataTypes>
void AugmentedLagrangianConstraint<DataTypes>::getConstraintResolution(const core::ConstraintParams *, std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset)
{
    if(this->contactsStatus)
    {
        delete[] this->contactsStatus;
        this->contactsStatus = nullptr;
    }

    if (this->contacts.size() > 0)
    {
        this->contactsStatus = new bool[this->contacts.size()];
        memset(this->contactsStatus, 0, sizeof(bool)*this->contacts.size());
    }

    for(unsigned int i=0; i<this->contacts.size(); i++)
    {
        Contact& c = this->contacts[i];
        if(c.parameters.hasTangentialComponent())
        {
            AugmentedLagrangianResolutionWithFriction* ucrwf = new AugmentedLagrangianResolutionWithFriction(c.parameters.mu,c.parameters.epsilon, nullptr, &(this->contactsStatus[i]));
            ucrwf->setTolerance(this->customTolerance);
            resTab[offset] = ucrwf;

            // TODO : cette méthode de stockage des forces peu mal fonctionner avec 2 threads quand on utilise l'haptique
            offset += 3;
        }
        else
            resTab[offset++] = new AugmentedLagrangianResolution(c.parameters.epsilon);
    }
}

} //namespace sofa::component::constraint::lagrangian::model
