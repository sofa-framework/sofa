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

#include <sofa/component/haptics/config.h>
#include <sofa/component/haptics/ForceFeedback.h>
#include <sofa/simulation/fwd.h>

namespace sofa::component::haptics
{

template<class TDataTypes>
class SOFA_COMPONENT_HAPTICS_API MechanicalStateForceFeedback : public ForceFeedback
{

public:
    SOFA_CLASS(SOFA_TEMPLATE(MechanicalStateForceFeedback,TDataTypes),ForceFeedback);

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;

public:
    virtual void computeForce(const  VecCoord& state,  VecDeriv& forces) = 0;

    void init() override {context = sofa::simulation::node::getNodeFrom(getContext());}
    void computeForce(SReal x, SReal y, SReal z, SReal u, SReal v, SReal w, SReal q, SReal& fx, SReal& fy, SReal& fz) override = 0;
    void computeWrench(const sofa::type::Transform<SReal> &, const sofa::type::SpatialVector<SReal> &, sofa::type::SpatialVector<SReal> & ) override = 0;
    void setReferencePosition(sofa::type::Transform<SReal>& /*referencePosition*/) override {}

protected:
    MechanicalStateForceFeedback(void) {}
};

} // namespace sofa::component::haptics
