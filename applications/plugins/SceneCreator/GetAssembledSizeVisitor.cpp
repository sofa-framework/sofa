/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
//
// C++ Implementation: GetAssembledSizeVisitor
//
// Description:
//
//
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include "GetAssembledSizeVisitor.h"
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace simulation
{


GetAssembledSizeVisitor::GetAssembledSizeVisitor( const sofa::core::ExecParams* params )
    : Visitor(params)
    , xsize(0)
    , vsize(0)
    , independentOnly(false)
{}

GetAssembledSizeVisitor::~GetAssembledSizeVisitor()
{}

void GetAssembledSizeVisitor::setIndependentOnly(bool b){ independentOnly=b; }

Visitor::Result GetAssembledSizeVisitor::processNodeTopDown( simulation::Node* gnode )
{
    if (gnode->mechanicalState != NULL && ( gnode->mechanicalMapping ==NULL || independentOnly==false) )
    {
        xsize += gnode->mechanicalState->getSize() * gnode->mechanicalState->getCoordDimension();
        vsize += gnode->mechanicalState->getMatrixSize();
    }
    return Visitor::RESULT_CONTINUE;
}

} // namespace simulation

} // namespace sofa

