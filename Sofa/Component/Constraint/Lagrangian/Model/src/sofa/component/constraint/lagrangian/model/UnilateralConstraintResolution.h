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
#include <sofa/component/constraint/lagrangian/model/config.h>

#include <sofa/core/behavior/ConstraintResolution.h>
#include <sofa/defaulttype/VecTypes.h>
#include <iostream>
#include <map>
#include <deque>


namespace sofa::component::constraint::lagrangian::model
{

class UnilateralConstraintResolution : public core::behavior::ConstraintResolution
{
   public:
    UnilateralConstraintResolution() : core::behavior::ConstraintResolution(1) {}

    void resolution(int line, SReal** w, SReal* d, SReal* force, SReal* dfree) override
    {
        SOFA_UNUSED(dfree);
        force[line] -= d[line] / w[line][line];
        if (force[line] < 0) force[line] = 0.0;
    }
};

// A little experiment on how to best save the forces for the hot start.
//  TODO : save as a map (index of the contact <-> force)
class PreviousForcesContainer
{
   public:
    PreviousForcesContainer() : resetFlag(true) {}
    SReal popForce()
    {
        resetFlag = true;
        if (forces.empty()) return 0;
        const SReal f = forces.front();
        forces.pop_front();
        return f;
    }

    void pushForce(SReal f)
    {
        if (resetFlag)
        {
            forces.clear();
            resetFlag = false;
        }

        forces.push_back(f);
    }

   protected:
    std::deque<SReal> forces;
    bool resetFlag;  // We delete all forces that were not read
};

class SOFA_COMPONENT_CONSTRAINT_LAGRANGIAN_MODEL_API UnilateralConstraintResolutionWithFriction
    : public core::behavior::ConstraintResolution
{
   public:
    UnilateralConstraintResolutionWithFriction(SReal mu, PreviousForcesContainer* prev = nullptr,
                                               bool* active = nullptr)
        : core::behavior::ConstraintResolution(3), _mu(mu), _prev(prev), _active(active)
    {
    }

    void init(int line, SReal** w, SReal* force) override;
    void resolution(int line, SReal** w, SReal* d, SReal* force, SReal* dFree) override;
    void store(int line, SReal* force, bool /*convergence*/) override;

   protected:
    SReal _mu;
    SReal _W[6];
    PreviousForcesContainer* _prev;
    bool* _active;  // Will set this after the resolution
};

}
