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
#include <sofa/simulation/common/XMLPrintVisitor.h>
#include <sofa/helper/Factory.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/core/behavior/BaseInteractionProjectiveConstraintSet.h>
#include <sofa/core/behavior/BaseInteractionConstraint.h>
#ifdef SOFA_HAVE_EIGEN2
#include <sofa/core/behavior/BaseLMConstraint.h>
#endif

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

    if (!compact) m_out << ">\n";

    obj->xmlWriteDatas( m_out, level+1, compact );

    if (compact)
    {
        m_out << "/>" << std::endl;
    }
    else
    {
        for (int i=0; i<level; i++)
            m_out << "\t";
        m_out << "</" << xmlencode(classname)  <<">" << std::endl;
    }
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
    node->xmlWriteNodeDatas(m_out,level);

    m_out << " >\n";

    if (node->mechanicalMapping != NULL)
        (*(node->mechanicalMapping.begin()))->disable();

    //processObjects(node->object);
    // BUGFIX(Jeremie A.): filter objects to output interactions classes after the children nodes to resolve dependencies at creation time
    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        sofa::core::objectmodel::BaseObject* obj = *it;
        if (   dynamic_cast<sofa::core::behavior::BaseInteractionForceField*> (obj) == NULL
                && dynamic_cast<sofa::core::behavior::BaseInteractionConstraint*> (obj) == NULL
                && dynamic_cast<sofa::core::behavior::BaseInteractionProjectiveConstraintSet*> (obj) == NULL
#ifdef SOFA_HAVE_EIGEN2
                && dynamic_cast<sofa::core::behavior::BaseLMConstraint*> (obj) == NULL
#endif
           )
            this->processObject(obj);
    }

    return RESULT_CONTINUE;
}

void XMLPrintVisitor::processNodeBottomUp(simulation::Node* node)
{
    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        sofa::core::objectmodel::BaseObject* obj = *it;
        if (   dynamic_cast<sofa::core::behavior::BaseInteractionForceField*> (obj) != NULL
                || dynamic_cast<sofa::core::behavior::BaseInteractionConstraint*> (obj) != NULL
                || dynamic_cast<sofa::core::behavior::BaseInteractionProjectiveConstraintSet*> (obj) != NULL
#ifdef SOFA_HAVE_EIGEN2
                || dynamic_cast<sofa::core::behavior::BaseLMConstraint*> (obj) != NULL
#endif
           )
            this->processObject(obj);
    }

    --level;

    for (int i=0; i<level; i++)
        m_out << "\t";
    m_out << "</Node>"<<std::endl;

}

} // namespace simulation

} // namespace sofa

