/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_SIMULATION_BGL_GETOBJECTSVISITOR_H
#define SOFA_SIMULATION_BGL_GETOBJECTSVISITOR_H

#include <sofa/simulation/common/Visitor.h>
#include "BglSimulation.h"
namespace sofa
{

namespace simulation
{

namespace bgl
{


class GetObjectsVisitor : public Visitor
{
public:
    typedef sofa::core::objectmodel::ClassInfo ClassInfo;
    typedef sofa::core::objectmodel::BaseContext::GetObjectsCallBack GetObjectsCallBack;

    GetObjectsVisitor(const ClassInfo& class_inf, GetObjectsCallBack& cont)
        : class_info(class_inf), container(cont)
    {}

    void setTags(const sofa::core::objectmodel::TagSet& t) {tags=t;}

    Result processNodeTopDown( simulation::Node* node );

    virtual const char* getClassName() const { return "GetObjectsVisitor"; }
    virtual const char* getInfos() const { std::string name="["+sofa::helper::gettypename(class_info)+"]"; return name.c_str();}


protected:

    const ClassInfo& class_info;
    GetObjectsCallBack& container;
    sofa::core::objectmodel::TagSet tags;
};


class GetObjectVisitor : public Visitor
{
public:
    typedef sofa::core::objectmodel::ClassInfo ClassInfo;
    typedef sofa::core::objectmodel::BaseContext::GetObjectsCallBack GetObjectsCallBack;
    typedef sofa::core::objectmodel::BaseContext::SearchDirection SearchDirection;
    GetObjectVisitor(const ClassInfo& class_inf)
        : class_info(class_inf), result(NULL)
    {}

    void setTags(const sofa::core::objectmodel::TagSet& t) {tags=t;}

    Result processNodeTopDown( simulation::Node* node );
    void *getObject() {return result;}
    virtual const char* getClassName() const { return "GetObjectVisitor"; }
    virtual const char* getInfos() const { std::string name="["+sofa::helper::gettypename(class_info)+"]"; return name.c_str();}
protected:

    const ClassInfo& class_info;
    sofa::core::objectmodel::TagSet tags;
    void *result;
};

} // namespace bgl

} // namespace simulation

} // namespace sofa

#endif
