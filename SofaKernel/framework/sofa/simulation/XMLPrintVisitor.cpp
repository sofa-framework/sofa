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
#include <sofa/simulation/XMLPrintVisitor.h>
#include <sofa/helper/Factory.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/core/behavior/BaseInteractionProjectiveConstraintSet.h>
#include <sofa/core/behavior/BaseInteractionConstraint.h>
#include <sofa/core/behavior/BaseLMConstraint.h>

namespace sofa
{

namespace simulation
{


static std::string xmlencode(const std::string& str)
{
    std::string res;
    for (unsigned int i=0; i<str.length(); ++i)
    {
        switch(str[i])
        {
        case '<': res += "&lt;"; break;
        case '>': res += "&gt;"; break;
        case '&': res += "&amp;"; break;
        case '"': res += "&quot;"; break;
        case '\'': res += "&apos;"; break;
        default:  res += str[i];
        }
    }
    return res;
}

template<class T>
void XMLPrintVisitor::processObject(T obj)
{
    for (int i=0; i<level; i++)
        m_out << "\t";

    std::string classname = obj->getClassName();
    std::string templatename = obj->getTemplateName();

    m_out << "<" << xmlencode(classname);
    if (!templatename.empty())
        m_out << " template=\"" << xmlencode(templatename) << "\"";

    obj->writeDatas( m_out, " " );

    m_out << "/>" << std::endl;
}

void XMLPrintVisitor::processBaseObject(core::objectmodel::BaseObject* obj)
{
    processObject(obj);
}

template<class Seq>
void XMLPrintVisitor::processObjects(Seq& list)
{
    if (list.empty()) return;
    // the following line breaks the compilator on Visual2003
    //for_each<XMLPrintVisitor, Seq, typename Seq::value_type>(this, list, &XMLPrintVisitor::processObject<typename Seq::value_type>);
    for (typename Seq::iterator it = list.begin(); it != list.end(); ++it)
    {
        typename Seq::value_type obj = *it;
        this->processObject<typename Seq::value_type>(obj);
    }
}

Visitor::Result XMLPrintVisitor::processNodeTopDown(simulation::Node* node)
{
    for (int i=0; i<level; i++)
        m_out << "\t";
    m_out << "<Node \t";

    ++level;
    node->writeDatas(m_out," ");

    m_out << " >\n";

    if (node->mechanicalMapping)
        (*(node->mechanicalMapping.begin()))->disable();

    //processObjects(node->object);
    // BUGFIX(Jeremie A.): filter objects to output interactions classes after the children nodes to resolve dependencies at creation time
    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        sofa::core::objectmodel::BaseObject* obj = it->get();
        if (    obj->toBaseInteractionForceField() == NULL
            &&  obj->toBaseInteractionConstraint() == NULL
            &&  obj->toBaseInteractionProjectiveConstraintSet() == NULL
            &&  obj->toBaseLMConstraint() == NULL
           )
            this->processObject(obj);
    }

    return RESULT_CONTINUE;
}

void XMLPrintVisitor::processNodeBottomUp(simulation::Node* node)
{
    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        sofa::core::objectmodel::BaseObject* obj = it->get();
        if (    obj->toBaseInteractionForceField() != NULL
            ||  obj->toBaseInteractionConstraint() != NULL
            ||  obj->toBaseInteractionProjectiveConstraintSet() != NULL
            ||  obj->toBaseLMConstraint() != NULL
           )
            this->processObject(obj);
    }

    --level;

    for (int i=0; i<level; i++)
        m_out << "\t";
    m_out << "</Node>"<<std::endl;

}

bool XMLPrintVisitor::treeTraversal(TreeTraversalRepetition& repeat)
{
	repeat = NO_REPETITION;
	return true;
}

} // namespace simulation

} // namespace sofa

