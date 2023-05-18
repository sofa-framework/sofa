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
#include <sofa/core/MechanicalParams.h>
#include <sofa/helper/system/thread/thread_specific_ptr.h>
#include <sofa/helper/BackTrace.h>
#include <cassert>
#include <iostream>

namespace sofa
{

namespace core
{

MechanicalParams::MechanicalParams(const sofa::core::ExecParams& p)
    : sofa::core::ExecParams(p)
    , m_dt(0.0)
    , m_implicit(false)
    , m_energy(false)
    , m_x (ConstVecCoordId::position())
    , m_v (ConstVecDerivId::velocity())
    , m_f (ConstVecDerivId::force())
    , m_dx(ConstVecDerivId::dx())
    , m_df(ConstVecDerivId::dforce())
    , m_mFactor(0)
    , m_bFactor(0)
    , m_kFactor(0)
    , m_supportOnlySymmetricMatrix(true)
    , m_implicitVelocity(1)
    , m_implicitPosition(1)
{
}

MechanicalParams::MechanicalParams(const MechanicalParams& p)
    : sofa::core::ExecParams(p)
    , m_dt(p.m_dt)
    , m_implicit(p.m_implicit)
    , m_energy(p.m_energy)
    , m_x (p.m_x)
    , m_v (p.m_v)
    , m_f (p.m_f)
    , m_dx(p.m_dx)
    , m_df(p.m_df)
    , m_mFactor(p.m_mFactor)
    , m_bFactor(p.m_bFactor)
    , m_kFactor(p.m_kFactor)
    , m_supportOnlySymmetricMatrix(p.m_supportOnlySymmetricMatrix)
    , m_implicitVelocity(p.m_implicitVelocity)
    , m_implicitPosition(p.m_implicitPosition)
{
}

MechanicalParams* MechanicalParams::setExecParams(const core::ExecParams* params)
{
    sofa::core::ExecParams::operator=(*params);
    return this;
}

MechanicalParams* MechanicalParams::operator= ( const MechanicalParams& mparams )
{
    sofa::core::ExecParams::operator=(mparams);
    m_dt = mparams.m_dt;
    m_implicit = mparams.m_implicit;
    m_energy = mparams.m_energy;
    m_x = mparams.m_x;
    m_v = mparams.m_v;
    m_f = mparams.m_f;
    m_dx = mparams.m_dx;
    m_df = mparams.m_df;
    m_mFactor = mparams.m_mFactor;
    m_bFactor = mparams.m_bFactor;
    m_kFactor = mparams.m_kFactor;
    m_supportOnlySymmetricMatrix = mparams.m_supportOnlySymmetricMatrix;
    m_implicitVelocity = mparams.m_implicitVelocity;
    m_implicitPosition = mparams.m_implicitPosition;
    return this;
}

const MechanicalParams* MechanicalParams::defaultInstance()
{
    SOFA_THREAD_SPECIFIC_PTR(MechanicalParams, threadParams);
    MechanicalParams* ptr = threadParams;
    if (!ptr)
    {
        ptr = new MechanicalParams;
        threadParams = ptr;
    }
    return ptr;
}

} // namespace core

} // namespace sofa
