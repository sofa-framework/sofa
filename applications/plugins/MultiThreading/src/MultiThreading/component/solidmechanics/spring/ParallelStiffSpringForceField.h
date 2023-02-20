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

#include <MultiThreading/config.h>
#include <MultiThreading/TaskSchedulerUser.h>
#include <sofa/component/solidmechanics/spring/StiffSpringForceField.h>

namespace sofa::simulation
{
class TaskScheduler;
}

namespace multithreading::component::solidmechanics::spring
{

template <class DataTypes>
using StiffSpringForceField = sofa::component::solidmechanics::spring::StiffSpringForceField<DataTypes>;

template <class DataTypes>
class ParallelStiffSpringForceField : public virtual StiffSpringForceField<DataTypes>, public TaskSchedulerUser
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ParallelStiffSpringForceField, DataTypes),
               SOFA_TEMPLATE(StiffSpringForceField, DataTypes));

    using VecCoord = typename DataTypes::VecCoord;
    using VecDeriv = typename DataTypes::VecDeriv;
    using DataVecCoord = sofa::core::objectmodel::Data<VecCoord>;
    using DataVecDeriv = sofa::core::objectmodel::Data<VecDeriv>;
    using Real = typename Inherit1::Real;

    using Spring = typename Inherit1::Spring;
    using SpringForce = typename Inherit1::SpringForce;
    using StiffSpringForce = typename Inherit1::StiffSpringForce;

    void init() override;

    void addForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2 ) override;
    void addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2) override;
};

}
