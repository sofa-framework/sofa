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
#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/common/VisualVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>

#ifdef SOFA_DUMP_VISITOR_INFO
#include <sofa/simulation/common/Simulation.h>
#endif
namespace sofa
{

namespace simulation
{



void Visitor::execute(sofa::core::objectmodel::BaseContext* c)
{
    c->executeVisitor(this);
}
#ifdef SOFA_DUMP_VISITOR_INFO
unsigned int Visitor::depthLevel=0;
simulation::Node::ctime_t Visitor::initDumpTime;
bool Visitor::printActivated=false;
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
        std::string info;
        if (dirDown)
        {
            info += "<Node name=\"" + context->getName() + "\">\n";
            Visitor::depthLevel++;
            initNodeTime.push_back(CTime::getRefTime());
        }
        else
        {
            Visitor::depthLevel--;
            ctime_t tSpent=initNodeTime.back();
            initNodeTime.pop_back();
            std::ostringstream s;
            s << "<Time value=\"" << getTimeSpent(tSpent,CTime::getRefTime()) << "\" />\n";

            info+= s.str()+"</Node>\n";
        }
        dumpInfo(info);
        return;
    }
    else if (!this->infoPrinted)
    {
        //Beginning processing: Visitor entered its first node
        this->infoPrinted=true;
        std::string info;

        std::string infos(this->getInfos());
        std::string NodeName;
        if (enteringBase) NodeName=enteringBase->getName();

        initVisitTime = CTime::getRefTime();

        info +="<" + std::string(this->getClassName());
        if (!infos.empty())
        {
            info += " infos=\"" + infos + "\"";
        }
        info+= ">\n";

        Visitor::depthLevel++;
        info+= "<Node name=\"" + NodeName + "\">\n";
        Visitor::depthLevel++;
        initNodeTime.push_back(CTime::getRefTime());
        dumpInfo(info);
    }
    else
    {
        //Ending the traversal: The visitor has finished its work
        if (this->infoPrinted)
        {
            std::ostringstream s;
            if (enteringBase)
            {
                Visitor::depthLevel--;
                ctime_t tSpent=initNodeTime.back();
                initNodeTime.pop_back();
                s << "<Time value=\"" << getTimeSpent(tSpent,CTime::getRefTime()) << "\" />\n";
                s << "</Node>\n";
            }
            Visitor::depthLevel--;



            s << "<Time value=\"" << getTimeSpent(initVisitTime,CTime::getRefTime()) << "\" />\n";

            s << "</" + std::string(this->getClassName()) + ">\n";
            dumpInfo(s.str());
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
#endif
/// Optional helper method to call before handling an object if not using the for_each method.
/// It currently takes care of time logging, but could be extended (step-by-step execution for instance)
simulation::Node::ctime_t Visitor::begin(simulation::Node* node, core::objectmodel::BaseObject*
#ifdef SOFA_DUMP_VISITOR_INFO
        obj
#endif
                                        )
{
#ifdef SOFA_DUMP_VISITOR_INFO
    if (printActivated)
    {

        std::ostringstream info;


        info<< "<Component type=\"" << obj->getClassName() << "\" name=\"" << obj->getName() << "\" ptr=\"" << obj << "\" >\n";
        dumpInfo(info.str());
        Visitor::depthLevel++;
        initComponentTime = CTime::getRefTime();
    }
#endif
    return node->startTime();
}

/// Optional helper method to call after handling an object if not using the for_each method.
/// It currently takes care of time logging, but could be extended (step-by-step execution for instance)
void Visitor::end(simulation::Node* node, core::objectmodel::BaseObject* obj, ctime_t t0)
{
    node->endTime(t0, getCategoryName(), obj);
#ifdef SOFA_DUMP_VISITOR_INFO
    if (printActivated)
    {
        Visitor::depthLevel--;
        std::string info;

        std::ostringstream s;
        s << "<Time value=\"" << getTimeSpent(initComponentTime,CTime::getRefTime()) << "\" />\n";
        info += s.str();
        info += "</Component>\n";
        dumpInfo(info);
    }
#endif
}

#ifdef SOFA_VERBOSE_TRAVERSAL
void Visitor::debug_write_state_before( core::objectmodel::BaseObject* obj )
{
    using std::cerr;
    using std::endl;
    if( dynamic_cast<VisualVisitor*>(this) ) return;
    cerr<<"Visitor "<<getClassName()<<" enter component "<<obj->getName();
    using core::componentmodel::behavior::BaseMechanicalState;
    if( BaseMechanicalState* dof = dynamic_cast<BaseMechanicalState*> ( obj->getContext()->getMechanicalState() ) )
    {
        cerr<<", state:\nx= "; dof->writeX(cerr);
        cerr<<"\nv= ";        dof->writeV(cerr);
        cerr<<"\ndx= ";       dof->writeDx(cerr);
        cerr<<"\nf= ";        dof->writeF(cerr);
    }
    cerr<<endl;
}

void Visitor::debug_write_state_after( core::objectmodel::BaseObject* obj )
{
    using std::cerr;
    using std::endl;
    if( dynamic_cast<VisualVisitor*>(this) ) return;
    cerr<<"Visitor "<<getClassName()<<" leave component "<<obj->getName();
    using core::componentmodel::behavior::BaseMechanicalState;
    if( BaseMechanicalState* dof = dynamic_cast<BaseMechanicalState*> ( obj->getContext()->getMechanicalState() ) )
    {
        cerr<<", state:\nx= "; dof->writeX(cerr);
        cerr<<"\nv= ";        dof->writeV(cerr);
        cerr<<"\ndx= ";       dof->writeDx(cerr);
        cerr<<"\nf= ";        dof->writeF(cerr);
    }
    cerr<<endl;
}
#endif

} // namespace simulation

} // namespace sofa

