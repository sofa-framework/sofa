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

#include <sofa/core/behavior/StateAccessor.h>
#include <sofa/core/behavior/MechanicalState.h>

namespace sofa::core::behavior
{

/**
 * Base class for components having access to one mechanical state with a specific template parameter, in order to read
 * and/or write state variables.
 */
template<class DataTypes>
class SingleStateAccessor : public virtual StateAccessor
{
public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE(SingleStateAccessor, DataTypes), StateAccessor);

    void init() override;

    MechanicalState<DataTypes>* getMState();
    const MechanicalState<DataTypes>* getMState() const;

protected:

    explicit SingleStateAccessor(MechanicalState<DataTypes>* mm = nullptr);

    ~SingleStateAccessor() override = default;

    SingleLink<SingleStateAccessor<DataTypes>, MechanicalState<DataTypes>, BaseLink::FLAG_STRONGLINK> mstate;
};

#if !defined(SOFA_CORE_BEHAVIOR_SINGLESTATEACCESSOR_CPP)
extern template class SOFA_CORE_API SingleStateAccessor<sofa::defaulttype::Vec1Types>;
extern template class SOFA_CORE_API SingleStateAccessor<sofa::defaulttype::Vec2Types>;
extern template class SOFA_CORE_API SingleStateAccessor<sofa::defaulttype::Vec3Types>;
extern template class SOFA_CORE_API SingleStateAccessor<sofa::defaulttype::Vec6Types>;
extern template class SOFA_CORE_API SingleStateAccessor<sofa::defaulttype::Rigid2Types>;
extern template class SOFA_CORE_API SingleStateAccessor<sofa::defaulttype::Rigid3Types>;
#endif

}
