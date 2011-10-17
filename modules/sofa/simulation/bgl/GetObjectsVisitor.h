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
#ifndef SOFA_SIMULATION_BGL_GETOBJECTSVISITOR_H
#define SOFA_SIMULATION_BGL_GETOBJECTSVISITOR_H

#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/bgl/BglSimulation.h>
namespace sofa
{

namespace simulation
{

namespace bgl
{


class SOFA_SIMULATION_BGL_API GetObjectsVisitor : public Visitor
{
public:
    typedef sofa::core::objectmodel::ClassInfo ClassInfo;
    typedef sofa::core::objectmodel::BaseContext::GetObjectsCallBack GetObjectsCallBack;

    GetObjectsVisitor(const sofa::core::ExecParams* params /* PARAMS FIRST */, const ClassInfo& class_inf, GetObjectsCallBack& cont, simulation::Node* ignoreNode = NULL)
        : Visitor(params), class_info(class_inf), container(cont), ignoreNode(ignoreNode)
    {}

    void setTags(const sofa::core::objectmodel::TagSet& t) {tags=t;}

    Result processNodeTopDown( simulation::Node* node );

    virtual const char* getClassName() const { return "GetObjectsVisitor"; }
    virtual std::string getInfos() const { return std::string("["+sofa::helper::gettypename(class_info)+"]");}


protected:

    const ClassInfo& class_info;
    GetObjectsCallBack& container;
    sofa::core::objectmodel::TagSet tags;
    simulation::Node* ignoreNode;
};


class SOFA_SIMULATION_BGL_API GetObjectVisitor : public Visitor
{
public:
    typedef sofa::core::objectmodel::ClassInfo ClassInfo;
    typedef sofa::core::objectmodel::BaseContext::GetObjectsCallBack GetObjectsCallBack;
    typedef sofa::core::objectmodel::BaseContext::SearchDirection SearchDirection;
    GetObjectVisitor(const sofa::core::ExecParams* params /* PARAMS FIRST */, const ClassInfo& class_inf, simulation::Node* ignoreNode = NULL)
        : Visitor(params), class_info(class_inf), result(NULL), ignoreNode(ignoreNode)
    {}

    void setTags(const sofa::core::objectmodel::TagSet& t) {tags=t;}

    Result processNodeTopDown( simulation::Node* node );
    void *getObject() {return result;}
    virtual const char* getClassName() const { return "GetObjectVisitor"; }
    virtual std::string getInfos() const { return std::string("["+sofa::helper::gettypename(class_info)+"]"); }
protected:

    const ClassInfo& class_info;
    sofa::core::objectmodel::TagSet tags;
    void *result;
    simulation::Node* ignoreNode;
};

} // namespace bgl

} // namespace simulation

} // namespace sofa

#endif
