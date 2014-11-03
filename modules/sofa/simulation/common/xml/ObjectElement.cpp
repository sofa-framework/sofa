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
#include <sofa/simulation/common/xml/ObjectElement.h>
#include <sofa/simulation/common/xml/Element.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace simulation
{

namespace xml
{

using namespace sofa::defaulttype;
using helper::Creator;

//template class Factory< std::string, objectmodel::BaseObject, Node<objectmodel::BaseObject*>* >;

ObjectElement::ObjectElement(const std::string& name, const std::string& type, BaseElement* parent)
    : Element<core::objectmodel::BaseObject>(name, type, parent)
{
}

ObjectElement::~ObjectElement()
{
}


bool ObjectElement::init()
{
    int i=0;
    for (child_iterator<> it = begin(); it != end(); ++it)
    {
        i++;
        it->init();
    }

    return initNode();
}



bool ObjectElement::initNode()
{
    core::objectmodel::BaseContext* ctx = dynamic_cast<core::objectmodel::BaseContext*>(getParent()->getObject());

    for (AttributeMap::iterator it = attributes.begin(), itend = attributes.end(); it != itend; ++it)
    {
        if (replaceAttribute.find(it->first) != replaceAttribute.end())
        {
            setAttribute(it->first,replaceAttribute[it->first].c_str());
        }
    }

    core::objectmodel::BaseObject::SPtr obj = core::ObjectFactory::CreateObject(ctx, this);

    if (obj == NULL)
        obj = Factory::CreateObject(this->getType(), this);
    if (obj == NULL)
    {
        getParent()->logWarning(std::string("Object type \"" + getType() + "\" creation Failed" ));
        return false;
    }
    setObject(obj);
    // display any unused attributes
    //std::string unused;
    for (AttributeMap::iterator it = attributes.begin(), itend = attributes.end(); it != itend; ++it)
    {
        if (!it->second.isAccessed())
        {
            std::string name = it->first;
            // ignore some prefix that are used to quickly disable parameters in XML files
            if (name.substr(0,1) == "_" || name.substr(0,2) == "NO") continue;
            //unused += ' ';
            //unused += name;

            obj->serr <<"Unused Attribute: \""<<it->first <<"\" with value: \"" <<it->second.c_str() <<"\"" << obj->sendl;
        }
    }
//     if (!unused.empty())
//     {
//         std::cerr << "WARNING: Unused attribute(s) in "<<getFullName()<<" :"<<unused<<std::endl;
//     }

    return true;
}

SOFA_DECL_CLASS(Object)

Creator<BaseElement::NodeFactory, ObjectElement> ObjectNodeClass("Object");

const char* ObjectElement::getClass() const
{
    return ObjectNodeClass.c_str();
}

} // namespace xml

} // namespace simulation

} // namespace sofa

