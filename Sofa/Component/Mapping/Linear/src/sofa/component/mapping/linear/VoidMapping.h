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

#include <sofa/component/mapping/linear/config.h>
#include <sofa/component/mapping/linear/LinearMapping.h>

#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/type/vector.h>

namespace sofa::component::mapping::linear
{

class VoidMapping : public LinearBaseMapping
{
public:
    SOFA_CLASS(VoidMapping, LinearBaseMapping);

    typedef LinearBaseMapping Inherit;
    typedef sofa::core::behavior::BaseMechanicalState In;
    typedef sofa::core::behavior::BaseMechanicalState Out;

    void init() override
    {
        fromModel = dynamic_cast<In*>(this->getContext()->getMechanicalState());
        toModel = dynamic_cast<Out*>(this->getContext()->getMechanicalState());
    }

    /// Accessor to the input model of this mapping
    virtual  type::vector<core::BaseState*> getFrom() override
    {
        type::vector<core::BaseState*> vec(1,fromModel);
        return vec;
    }

    /// Accessor to the output model of this mapping
    virtual type::vector<core::BaseState*> getTo() override
    {
        type::vector<core::BaseState*> vec(1,toModel);
        return vec;
    }

    /// Disable the mapping to get the original coordinates of the mapped model.
    void disable() override
    {
    }

    /// Get the source (upper) model.
    virtual type::vector<sofa::core::behavior::BaseMechanicalState*> getMechFrom() override
    {
        type::vector<sofa::core::behavior::BaseMechanicalState*> vec(1, fromModel);
        return vec;
    }

    /// Get the destination (lower, mapped) model.
    virtual type::vector<sofa::core::behavior::BaseMechanicalState*> getMechTo() override
    {
        type::vector<sofa::core::behavior::BaseMechanicalState*> vec(1, toModel);
        return vec;
    }

    void apply (const core::MechanicalParams* /* mparams */, core::MultiVecCoordId /* outPos */, core::ConstMultiVecCoordId /* inPos */) override
    {
    }

    void applyJ(const core::MechanicalParams* /* mparams */, core::MultiVecDerivId /* outVel */, core::ConstMultiVecDerivId /* inVel */) override
    {
    }

    void applyJT(const core::MechanicalParams* /* mparams */, core::MultiVecDerivId /* inForce */, core::ConstMultiVecDerivId /* outForce */) override
    {
    }

    void applyDJT(const core::MechanicalParams* /*mparams*/, core::MultiVecDerivId /*inForce*/, core::ConstMultiVecDerivId /*outForce*/) override {}

    void applyJT(const core::ConstraintParams * /*cparams*/, core::MultiMatrixDerivId /* inConst */, core::ConstMultiMatrixDerivId /* outConst */) override
    {
    }

    void computeAccFromMapping(const core::MechanicalParams* /*mparams */, core::MultiVecDerivId /* outAcc */, core::ConstMultiVecDerivId /* inVel */, core::ConstMultiVecDerivId /* inAcc */) override
    {
    }

protected:
    In* fromModel;
    Out* toModel;

    VoidMapping():Inherit(),fromModel(nullptr),toModel(nullptr)
    {
        this->f_mapForces.setValue(false);
        this->f_mapConstraints.setValue(false);
        this->f_mapMasses.setValue(false);
    }

    ~VoidMapping() override
    {
    }

};

} // namespace sofa::component::mapping::linear
