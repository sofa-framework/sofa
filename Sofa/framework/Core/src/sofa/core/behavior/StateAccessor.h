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

#include <sofa/core/config.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/BaseMechanicalState.h>


namespace sofa::core::behavior
{

/**
 * Base class for components having access to one or more mechanical states, in order to read and/or write state variables.
 * Example: force field, mass, constraints etc
 *
 * Those components store a list of BaseMechanicalState. It does not prevent them to store the same BaseMechanicalState
 * as a derived type.
 */
class SOFA_CORE_API StateAccessor : public virtual objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(StateAccessor, objectmodel::BaseObject);

    /// Return a list of mechanical states to which this component is associated
    virtual const MultiLink<StateAccessor, BaseMechanicalState, BaseLink::FLAG_DUPLICATE>::Container& getMechanicalStates() const
    {
        return l_mechanicalStates.getValue();
    }

    void computeBBox(const core::ExecParams* params, bool onlyVisible=false) override;

protected:

    StateAccessor()
        : Inherit1()
        , l_mechanicalStates(initLink("mechanicalStates", "List of mechanical states to which this component is associated"))
    {}

    ~StateAccessor() override = default;

    /// List of mechanical states to which this component is associated
    /// The list can contain more than one mechanical states. In an interaction force field, for example.
    MultiLink < StateAccessor, BaseMechanicalState, BaseLink::FLAG_DUPLICATE > l_mechanicalStates;

};


inline void StateAccessor::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    SOFA_UNUSED(params);

    if( !onlyVisible ) return;

    static constexpr SReal max_real = std::numeric_limits<SReal>::max();
    static constexpr SReal min_real = std::numeric_limits<SReal>::lowest();
    SReal maxBBox[3] { min_real, min_real, min_real };
    SReal minBBox[3] { max_real, max_real, max_real };

    bool anyMState = false;

    for (const auto mstate : l_mechanicalStates)
    {
        if (mstate)
        {
            const auto& bbox = mstate->f_bbox.getValue();
            for (unsigned int i = 0; i < 3; ++i)
            {
                maxBBox[i] = std::max(maxBBox[i], bbox.maxBBox()[i]);
                minBBox[i] = std::min(minBBox[i], bbox.minBBox()[i]);
            }
            anyMState = true;
        }
    }

    if (anyMState)
    {
        this->f_bbox.setValue(sofa::type::TBoundingBox(minBBox,maxBBox));
    }
}

} // namespace sofa::core::behavior
