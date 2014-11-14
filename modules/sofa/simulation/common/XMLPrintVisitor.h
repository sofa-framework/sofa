/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_SIMULATION_XMLPRINTACTION_H
#define SOFA_SIMULATION_XMLPRINTACTION_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/simulation/common/Visitor.h>
#include <string>


namespace sofa
{

namespace simulation
{


class SOFA_SIMULATION_COMMON_API XMLPrintVisitor : public Visitor
{
protected:
    std::ostream& m_out;
    int level;
public:
    XMLPrintVisitor(const sofa::core::ExecParams* params /* PARAMS FIRST */, std::ostream& out) : Visitor(params), m_out(out),level(0) {}

    template<class T>
    void processObject(T obj);

    template<class Seq>
    void processObjects(Seq& list);

    void processBaseObject(core::objectmodel::BaseObject* obj);

    virtual Result processNodeTopDown(simulation::Node* node);
    virtual void processNodeBottomUp(simulation::Node* node);
    virtual const char* getClassName() const { return "XMLPrintVisitor"; }
    int getLevel() const {return level;}
    void setLevel(int l) {level=l;};
	virtual bool treeTraversal(TreeTraversalRepetition& repeat);
};

} // namespace simulation

} // namespace sofa

#endif
