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
#include <sofa/core/ConstraintParams.h>
#include <sofa/helper/system/thread/thread_specific_ptr.h>
#include <sofa/helper/BackTrace.h>
#include <cassert>
#include <iostream>


namespace sofa::core
{

ConstraintParams::ConstraintParams(const sofa::core::ExecParams& p)
    : sofa::core::ExecParams(p)
    , m_x(vec_id::read_access::position)
    , m_v(vec_id::read_access::velocity)
    , m_j(vec_id::write_access::constraintJacobian)
    , m_dx(vec_id::write_access::dx)
    , m_lambda(vec_id::write_access::externalForce)
    , m_constOrder (ConstraintOrder::POS_AND_VEL)
    , m_smoothFactor (1)
{
}

ConstraintParams& ConstraintParams::setExecParams(const core::ExecParams* params)
{
    sofa::core::ExecParams::operator=(*params);
    return *this;
}

/// Get the default ConstraintParams, to be used to provide a default values for method parameters
const ConstraintParams* ConstraintParams::defaultInstance()
{
    thread_local ConstraintParams threadParams;
    return &threadParams;
}

} // namespace sofa::core


