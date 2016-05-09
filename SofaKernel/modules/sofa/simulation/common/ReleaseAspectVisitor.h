/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef SOFA_SIMULATION_RELEASEASPECTVISITOR_H
#define SOFA_SIMULATION_RELEASEASPECTVISITOR_H

#include <sofa/simulation/common/Visitor.h>
#include <sofa/core/ExecParams.h>

namespace sofa
{

namespace simulation
{

class SOFA_SIMULATION_COMMON_API ReleaseAspectVisitor : public Visitor
{
public:
    ReleaseAspectVisitor(const core::ExecParams* params, int aspect);
    ~ReleaseAspectVisitor();

    Result processNodeTopDown(Node* node);

protected:
    void processObject(sofa::core::objectmodel::BaseObject* obj);

    int aspect;
};

} // namespace sofa

} // namespace simulation

#endif /* SOFA_SIMULATION_RELEASEASPECTVISITOR_H */
