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
#ifndef SOFA_SIMULATION_CORE_COMPUTEBOUNDINGBOXVISITOR_H
#define SOFA_SIMULATION_CORE_COMPUTEBOUNDINGBOXVISITOR_H

#include <sofa/simulation/Visitor.h>
#include <sofa/core/ExecParams.h>
namespace sofa
{
namespace simulation
{

class SOFA_SIMULATION_CORE_API UpdateBoundingBoxVisitor : public Visitor
{
public:

    UpdateBoundingBoxVisitor(const sofa::core::ExecParams* params);

    virtual Result processNodeTopDown(simulation::Node* node);

    void processNodeBottomUp(simulation::Node* node);

};


}
}


#endif
