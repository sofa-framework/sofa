/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_SIMULATION_PRINTACTION_H
#define SOFA_SIMULATION_PRINTACTION_H

#include <sofa/simulation/common/Visitor.h>
#include <string>


namespace sofa
{

namespace simulation
{


class SOFA_SIMULATION_COMMON_API PrintVisitor : public Visitor
{
protected:
    int verbose;
    int level;
public:
    PrintVisitor(const sofa::core::ExecParams* params) : Visitor(params), verbose(0), level(0) {}

    void setVerbose(int v) { verbose = v; }
    int getVerbose() const { return verbose; }

    template<class T>
    void processObject(T obj);

    template<class Seq>
    void processObjects(Seq& list, const char* name);

    virtual Result processNodeTopDown(simulation::Node* node);
    virtual void processNodeBottomUp(simulation::Node* node);
    virtual const char* getClassName() const { return "PrintVisitor"; }
};

} // namespace simulation

} // namespace sofa

#endif
