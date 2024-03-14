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

#include <sofa/component/mapping/nonlinear/config.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/OptionsGroup.h>

namespace sofa::component::mapping::nonlinear
{

template<bool HasStabilizedGeometricStiffness>
class NonLinearMappingData : public virtual sofa::core::objectmodel::Base
{
public:
    Data<helper::OptionsGroup> d_geometricStiffness; ///< Method used to compute the geometric stiffness

    NonLinearMappingData();

protected:

    void checkLinearSolverSymmetry(const core::MechanicalParams* mparams) const;
};

template <bool HasStabilizedGeometricStiffness>
NonLinearMappingData<HasStabilizedGeometricStiffness>::NonLinearMappingData()
: d_geometricStiffness(initData(&d_geometricStiffness,
    helper::OptionsGroup{{"None", "Exact", "Stabilized"}}.setSelectedItem(2),
    "geometricStiffness",
    "Method used to compute the geometric stiffness:\n"
        "-None: geometric stiffness is not computed\n"
        "-Exact: the exact geometric stiffness is computed\n"
        "-Stabilized: the exact geometric stiffness is approximated in order to improve stability")
)
{}

template <>
inline NonLinearMappingData<false>::NonLinearMappingData()
: d_geometricStiffness(initData(&d_geometricStiffness,
    helper::OptionsGroup{{"None", "Exact"}}.setSelectedItem(1),
    "geometricStiffness",
    "Method used to compute the geometric stiffness:\n"
        "-None: geometric stiffness is not computed\n"
        "-Exact: the exact geometric stiffness is computed")
)
{}

template <bool HasStabilizedGeometricStiffness>
void NonLinearMappingData<HasStabilizedGeometricStiffness>::checkLinearSolverSymmetry(
    const core::MechanicalParams* mparams) const
{
    if (mparams && mparams->supportOnlySymmetricMatrix())
    {
        std::stringstream ss;
        ss << "The geometric stiffness of this mapping is a non-symmetric matrix. "
            "It means a linear solver supporting non-symmetric matrices must be used, but it is not"
            " the case here. ";

        const std::string stabilizedName = "Stabilized";
        if (d_geometricStiffness.getValue().isInOptionsList(stabilizedName) == -1)
        {
            if constexpr (HasStabilizedGeometricStiffness)
            {
                dmsg_fatal() << "The option '" << stabilizedName << "' is not available in the Data '"
                   << d_geometricStiffness.getName() << "'";
            }
            ss << "To fix your scene, use a linear solver supporting non-symmetric matrices";
        }
        else
        {
            ss << "To fix your scene, you have two options: 1) Use a linear solver "
                  "supporting non-symmetric matrices, 2) stabilize the geometric stiffness with "
                  "the Data '" + d_geometricStiffness.getName() + "' set to '" + stabilizedName + "'";
        }
        msg_error() << ss.str();
    }
}
}
