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
#include <sofa/simulation/Visitor.h>
#include <sofa/simulation/VisualVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/Simulation.h>


namespace sofa
{

namespace simulation
{

Visitor::Visitor(const core::ExecParams* p)
    : canAccessSleepingNode(true)
    , params(p)
{
    //params = core::MechanicalParams::defaultInstance();
#ifdef SOFA_DUMP_VISITOR_INFO
    enteringBase=NULL; infoPrinted=false;
#endif
}

Visitor::~Visitor()
{
}

void Visitor::execute(sofa::core::objectmodel::BaseContext* c, bool precomputedOrder)
{
    c->executeVisitor(this, precomputedOrder);
}

#ifdef SOFA_DUMP_VISITOR_INFO
Visitor::ctime_t Visitor::initDumpTime;
std::vector< Visitor::ctime_t  > Visitor::initNodeTime=std::vector< Visitor::ctime_t >();
bool Visitor::printActivated=false;
bool Visitor::outputStateVector=false;
unsigned int Visitor::firstIndexStateVector=0;
int Visitor::rangeStateVector=1;

std::ostream *Visitor::outputVisitor=NULL;

void Visitor::setNode(core::objectmodel::Base* c)
{
    if (!enteringBase) enteringBase=c;
}
void Visitor::printInfo(const core::objectmodel::BaseContext* context, bool dirDown)
{
    if (!Visitor::printActivated) return;
    //Traversing the Graph: print the name of the context
    if (context != enteringBase)
    {
        //std::string info;
        if (dirDown)
        {
            printNode("Node",context->getName());
        }
        else
        {
            printCloseNode("Node");
        }
        return;
    }
    else if (!this->infoPrinted)
    {
        //Beginning processing: Visitor entered its first node
        this->infoPrinted=true;

        std::string infos(this->getInfos());
        std::string NodeName;
        if (enteringBase) NodeName=enteringBase->getName();

        TRACE_ARGUMENT arg;
        arg.push_back(std::make_pair("infos",infos));
        printNode(this->getClassName(), std::string(), arg);


        arg.clear();
        printNode("Node",NodeName,arg);
    }
    else
    {
        //Ending the traversal: The visitor has finished its work
        if (this->infoPrinted)
        {
            if (enteringBase)
            {
                printCloseNode("Node");
            }
            printCloseNode(this->getClassName());
        }
        //Reinit the Visitor debug variables
        enteringBase=NULL;
        infoPrinted=false;
    }

}

void Visitor::printComment(const std::string &s)
{
    if (Visitor::printActivated)
    {
        std::string info;
        info+= "<!--";
        info+=  s + " -->\n";
        dumpInfo(info);
    }
}

void Visitor::dumpInfo( const std::string &info)
{
    if (printActivated) {(*outputVisitor) << info; /*outputVisitor->flush();*/}
}

void Visitor::startDumpVisitor(std::ostream *s, SReal time)
{
    initDumpTime = sofa::helper::system::thread::CTime::getRefTime();
    printActivated=true; outputVisitor=s;
    //std::string initDump;
    std::ostringstream ff; ff << "<TraceVisitor time=\"" << time << "\">\n";
    dumpInfo(ff.str());
};
void Visitor::stopDumpVisitor()
{
    std::ostringstream s;
    s << "<TotalTime value=\"" << getTimeSpent(initDumpTime,  sofa::helper::system::thread::CTime::getRefTime() ) << "\" />\n";
    s << "</TraceVisitor>\n";
    dumpInfo(s.str());
    printActivated=false;
};

SReal Visitor::getTimeSpent(ctime_t initTime, ctime_t endTime)
{
    return (SReal)(endTime-initTime);
}


void Visitor::printVector(core::behavior::BaseMechanicalState *mm, core::ConstVecId id)
{
    if (id.type != core::V_COORD && id.type != core::V_DERIV) return;
    std::ostringstream infoStream;
    TRACE_ARGUMENT arg;
    mm->printDOF(id, infoStream,firstIndexStateVector, rangeStateVector);
    std::string vectorValue=infoStream.str();
    if (vectorValue.empty()) return;

    infoStream.str("");
    if      (id.type == core::V_COORD) infoStream << mm->getCoordDimension() << " ";
    else if (id.type == core::V_DERIV) infoStream << mm->getDerivDimension() << " ";
    vectorValue = infoStream.str()+vectorValue;
    arg.push_back(std::make_pair("value", vectorValue));

    printNode("Vector", id.getName(), arg);
    printCloseNode("Vector");
}

void Visitor::printNode(const std::string &type, const std::string &name, const TRACE_ARGUMENT &arguments)
{
    if (Visitor::printActivated)
    {
        std::ostringstream s;
        s << "<" << type;
        if (!name.empty()) s << " name=\"" << name << "\"";
        for (unsigned int i=0; i<arguments.size(); ++i)
        {
            if (!arguments[i].second.empty())
                s << " " <<arguments[i].first << "=\"" << arguments[i].second << "\"";
        }
        s << ">\n";

        initNodeTime.push_back(CTime::getRefTime());
        dumpInfo(s.str());
    }
}
void Visitor::printCloseNode(const std::string &type)
{
    if (Visitor::printActivated)
    {
        std::ostringstream s;
        ctime_t tSpent = initNodeTime.back(); initNodeTime.pop_back();
        s << "<Time value=\"" << getTimeSpent(tSpent,CTime::getRefTime()) << "\" />\n";
        s << "</" << type << ">\n";
        dumpInfo(s.str());
    }
}

void Visitor::printComment(const char* s)
{
    if (!Visitor::printActivated) return;
    printComment(std::string(s));
}
void Visitor::printNode(const char* type, const std::string &name, const TRACE_ARGUMENT &arguments)
{
    if (!Visitor::printActivated) return;
    printNode(std::string(type),name,arguments);
}
void Visitor::printNode(const char* type, const std::string &name)
{
    if (!Visitor::printActivated) return;
    printNode(std::string(type),name);
}
void Visitor::printNode(const char* type)
{
    if (!Visitor::printActivated) return;
    printNode(std::string(type));
}
void Visitor::printCloseNode(const char* type)
{
    if (!Visitor::printActivated) return;
    printCloseNode(std::string(type));
}

#endif
/// Optional helper method to call before handling an object if not using the for_each method.
/// It currently takes care of time logging, but could be extended (step-by-step execution for instance)
simulation::Visitor::ctime_t Visitor::begin(simulation::Node* /*node*/, core::objectmodel::BaseObject*
#ifdef SOFA_DUMP_VISITOR_INFO
        obj
#endif
        , const std::string &
#ifdef SOFA_DUMP_VISITOR_INFO
        info
#endif
                                           )
{
#ifdef SOFA_DUMP_VISITOR_INFO
    if (printActivated)
    {
        TRACE_ARGUMENT arg;
        arg.push_back(std::make_pair(info,std::string(obj->getClassName())));

        std::ostringstream s; s << obj;
        arg.push_back(std::make_pair("ptr",s.str()));
        printNode("Component", obj->getName(), arg);
    }
#endif
    return ctime_t();
}

/// Optional helper method to call after handling an object if not using the for_each method.
/// It currently takes care of time logging, but could be extended (step-by-step execution for instance)
void Visitor::end(simulation::Node* /*node*/, core::objectmodel::BaseObject* /*obj*/, ctime_t)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    if (printActivated)
    {
        printCloseNode("Component");
    }
#endif
}

/// Optional helper method to call before handling an object if not using the for_each method.
/// It currently takes care of time logging, but could be extended (step-by-step execution for instance)
simulation::Visitor::ctime_t Visitor::begin(simulation::Visitor::VisitorContext* vc, core::objectmodel::BaseObject* obj, const std::string &info)
{
    return begin(vc->node, obj, info);
}

/// Optional helper method to call after handling an object if not using the for_each method.
/// It currently takes care of time logging, but could be extended (step-by-step execution for instance)
void Visitor::end(simulation::Visitor::VisitorContext* vc, core::objectmodel::BaseObject* obj, ctime_t t0)
{
    end(vc->node, obj, t0);
}

#ifdef SOFA_VERBOSE_TRAVERSAL
void Visitor::debug_write_state_before( core::objectmodel::BaseObject* obj )
{
    if( dynamic_cast<VisualVisitor*>(this) ) return;
    std::stringstream tmp;
    tmp<<"Visitor "<<getClassName()<<" enter component "<<obj->getName();
    using core::behavior::BaseMechanicalState;
    if( BaseMechanicalState* dof = obj->getContext()->getMechanicalState() )
    {
        tmp<<", state:\nx= "; dof->writeX(tmp);
        tmp<<"\nv= ";        dof->writeV(tmp);
        tmp<<"\ndx= ";       dof->writeDx(tmp);
        tmp<<"\nf= ";        dof->writeF(tmp);
    }
    dmsg_info("Visitor(debug)") << tmp.str() ;
}

void Visitor::debug_write_state_after( core::objectmodel::BaseObject* obj )
{
    if( dynamic_cast<VisualVisitor*>(this) ) return;
    std::stringstream tmp ;
    tmp<<"Visitor "<<getClassName()<<" leave component "<<obj->getName();
    using core::behavior::BaseMechanicalState;
    if( BaseMechanicalState* dof = obj->getContext()->getMechanicalState() )
    {
        tmp<<", state:\nx= "; dof->writeX(tmp);
        tmp<<"\nv= ";        dof->writeV(tmp);
        tmp<<"\ndx= ";       dof->writeDx(tmp);
        tmp<<"\nf= ";        dof->writeF(tmp);
    }
    dmsg_info() << tmp.str() ;
}
#endif

} // namespace simulation

} // namespace sofa

