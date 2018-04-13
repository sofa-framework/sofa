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
#ifndef SOFA_SIMULATION_CORE_VISITOREXECUTE_H
#define SOFA_SIMULATION_CORE_VISITOREXECUTE_H


#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/simulation/MechanicalVisitor.h>

namespace sofa
{
namespace simulation
{
namespace common
{

struct VisitorExecuteFunc
{
protected:
    core::objectmodel::BaseContext& ctx;

public:

    bool precomputedTraversalOrder;

    VisitorExecuteFunc(core::objectmodel::BaseContext& ctx, bool precomputedTraversalOrder=false)
        : ctx(ctx)
        , precomputedTraversalOrder(precomputedTraversalOrder)
    {}

    template< class Visitor >
    void operator()(Visitor* pv)
    {
        prepareVisitor(pv);
        pv->execute(&ctx,precomputedTraversalOrder);
    }
    template< class Visitor >
    void operator()(Visitor v)
    {
        prepareVisitor(&v);
        v.execute(&ctx,precomputedTraversalOrder);
    }
protected:
    void prepareVisitor( sofa::simulation::Visitor* v)
    {
        v->setTags( ctx.getTags() );
    }
    void prepareVisitor( sofa::simulation::BaseMechanicalVisitor* mv)
    {
        prepareVisitor( (sofa::simulation::Visitor*)mv );
    }
};
}
}
}

#endif // SOFA_SIMULATION_CORE_VISITOREXECUTE_H
