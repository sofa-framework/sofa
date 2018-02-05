/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
//
// C++ Implementation: GetVectorVisitor
//
// Description:
//
//
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include "GetVectorVisitor.h"
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace simulation
{


GetVectorVisitor::GetVectorVisitor( const sofa::core::ExecParams* params, Vector* vec, core::ConstVecId src )
    : Visitor(params), vec(vec), src(src), offset(0)
    , independentOnly(false)
{}

GetVectorVisitor::~GetVectorVisitor()
{}

void GetVectorVisitor::setIndependentOnly(bool b){ independentOnly=b; }


Visitor::Result GetVectorVisitor::processNodeTopDown( simulation::Node* gnode )
{
//    cerr << "GetVectorVisitor::processNodeTopDown, node "<< gnode->getName() << endl;
    if (gnode->mechanicalState != NULL && ( gnode->mechanicalMapping ==NULL || independentOnly==false) )
    {
//        cerr << "GetVectorVisitor::processNodeTopDown, node has mechanical state "<< endl;
        gnode->mechanicalState->copyToBaseVector(vec,src,offset);
    }
    return Visitor::RESULT_CONTINUE;
}

} // namespace simulation

} // namespace sofa

