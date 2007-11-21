/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/simulation/tree/XMLPrintVisitor.h>
#include <sofa/helper/Factory.h>
#include <sofa/simulation/tree/GNode.h>

namespace sofa
{

namespace simulation
{

namespace tree
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

    m_out << "<Object type=\"" << xmlencode(classname) << "\"";
    if (!templatename.empty())
        m_out << " template=\"" << xmlencode(templatename) << "\"";

    m_out << ">\n";

    obj->xmlWriteDatas( m_out, level+1 );


    for (int i=0; i<level; i++)
        m_out << "\t";
    m_out << "</Object>" << std::endl;
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

Visitor::Result XMLPrintVisitor::processNodeTopDown(GNode* node)
{
    for (int i=0; i<level; i++)
        m_out << "\t";
    m_out << "<Node ";

    ++level;
    node->xmlWriteNodeDatas(m_out,level);

    for (int i=0; i<level; i++)
        m_out << "\t";
    m_out << " >\n";

    if (node->mechanicalMapping != NULL)
        (*(node->mechanicalMapping.begin()))->disable();

    processObjects(node->object);

    return RESULT_CONTINUE;
}

void XMLPrintVisitor::processNodeBottomUp(GNode* /*node*/)
{
    --level;

    for (int i=0; i<level; i++)
        m_out << "\t";
    m_out << "</Node>"<<std::endl;

}

} // namespace tree

} // namespace simulation

} // namespace sofa

