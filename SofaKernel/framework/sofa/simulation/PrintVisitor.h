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
#ifndef SOFA_SIMULATION_PRINTACTION_H
#define SOFA_SIMULATION_PRINTACTION_H

#include <sofa/simulation/Visitor.h>
#include <string>


namespace sofa
{

namespace simulation
{


class SOFA_SIMULATION_CORE_API PrintVisitor : public Visitor
{
protected:
    int verbose;
    int level;
    bool visitingOrder; ///< by default print the graph organisation but can print the graph visiting by setting visitingOrder at true
public:
    PrintVisitor(const sofa::core::ExecParams* params, bool visitingOrder=false) : Visitor(params), verbose(0), level(0), visitingOrder(visitingOrder) {}

    void setVerbose(int v) { verbose = v; }
    int getVerbose() const { return verbose; }

    bool treeTraversal( TreeTraversalRepetition& repeat )
    {
        if( visitingOrder )
            return Visitor::treeTraversal( repeat ); // run the visitor with a regular traversal
        else
        {
             // run the visitor with a tree traversal
            repeat=REPEAT_ONCE;
            return true;
        }
    }

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
