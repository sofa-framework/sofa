/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

/// Get the default MechanicalParams, to be used to provide a default values for method parameters
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
