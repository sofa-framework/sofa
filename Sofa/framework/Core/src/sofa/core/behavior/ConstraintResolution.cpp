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
#include <sofa/core/behavior/ConstraintResolution.h>
#include <sofa/helper/logging/Messaging.h>

namespace sofa::core::behavior
{

ConstraintResolution::ConstraintResolution(unsigned int nbLines, SReal tolerance)
    :m_nbLines(nbLines)
    ,m_tolerance(tolerance)
{
}

ConstraintResolution::~ConstraintResolution()
{

}

void ConstraintResolution::init(int /*line*/, SReal** /*w*/, SReal* /*force*/)
{

}

void ConstraintResolution::initForce(int /*line*/, SReal* /*force*/)
{

}

/// Resolution of the constraint for one Gauss-Seidel iteration
void ConstraintResolution::resolution(int line, SReal** w, SReal* d, SReal* force, SReal * dFree)
{
    SOFA_UNUSED(line);
    SOFA_UNUSED(w);
    SOFA_UNUSED(d);
    SOFA_UNUSED(force);
    SOFA_UNUSED(dFree);
    dmsg_error("ConstraintResolution")
            << "resolution(int , SReal** , SReal* , SReal* , SReal * ) not implemented." ;
}
void ConstraintResolution::store(int /*line*/, SReal* /*force*/, bool /*convergence*/)
{

}

} /// namespace sofa::core::behavior

