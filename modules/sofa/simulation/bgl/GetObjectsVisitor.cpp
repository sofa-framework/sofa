/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/simulation/bgl/GetObjectsVisitor.h>

namespace sofa
{

namespace simulation
{

namespace bgl
{
Visitor::Result GetObjectsVisitor::processNodeTopDown( simulation::Node* node )
{
    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        core::objectmodel::BaseObject* obj = it->get();
        void* result = class_info.dynamicCast(obj);
        if (result != NULL &&  (tags.empty() || (obj)->getTags().includes(tags)))
            container(result);
    }
    return Visitor::RESULT_CONTINUE;
}

Visitor::Result GetObjectVisitor::processNodeTopDown( simulation::Node* node )
{
    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        core::objectmodel::BaseObject* obj = it->get();
        void* r = class_info.dynamicCast(obj);
        if (r != NULL &&  (tags.empty() || (obj)->getTags().includes(tags)))
        {
            result=r; return Visitor::RESULT_PRUNE;
        }
    }
    return Visitor::RESULT_CONTINUE;
}

}
}
}
