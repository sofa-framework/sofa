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
#include <sofa/component/statecontainer/MappedObject.h>

#include <sofa/core/behavior/BaseMechanicalState.h>

namespace sofa::component::statecontainer
{

template <class DataTypes>
MappedObject<DataTypes>::MappedObject()
    : d_X(initData(&d_X, "position", "position vector") )
    , d_V(initData(&d_V, "velocity", "velocity vector") )
{
    f_X.setParent(&d_X);
    f_V.setParent(&d_V);

}

template <class DataTypes>
MappedObject<DataTypes>::~MappedObject()
{
}

template <class DataTypes>
void MappedObject<DataTypes>::init()
{
    if (getSize() == 0)
    {
        const sofa::core::behavior::BaseMechanicalState* mstate = this->getContext()->getMechanicalState();
        auto nbp = mstate->getSize();
        if (nbp > 0)
        {
            VecCoord& x = *getX();
            x.resize(nbp);
            for (Index i=0; i<nbp; i++)
            {
                DataTypes::set(x[i], mstate->getPX(i), mstate->getPY(i), mstate->getPZ(i));
            }
        }
    }
}

} // namespace sofa::component::statecontainer
