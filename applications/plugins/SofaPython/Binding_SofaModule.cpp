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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "Binding_SofaModule.h"


/*
// factory!
BaseObject::SPtr createObject(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
{
    BaseObject::SPtr obj = ObjectFactory::getInstance()->createObject(context,arg);//.get();
    if (obj==0)
        printf("<PYTHON> ERROR createObject '%s' of type '%s' in node '%s'\n",
                        arg->getName().c_str(),
                        arg->getAttribute("type",""),
                        context->getName().c_str());
    return obj;
}
// fonction templatisée, on passe par une autre qui ne l'est pas sinon c'est juste illisible...
BaseObject::SPtr getObject(objectmodel::BaseContext* context,const std::string& path)
{
    BaseObject::SPtr sptr;
    context->get<BaseObject>(sptr,path);
    return sptr;
}
BaseNode::SPtr getChildNode(objectmodel::BaseNode* node,const std::string& path)
{
    const objectmodel::BaseNode::Children& children = node->getChildren();
    BaseNode::SPtr sptr;
    for (unsigned int i=0;i<children.size();++i)
        if (children[i]->getName() == path)
        {
            sptr = children[i];
            break;
        }
    return sptr;
}
// send a message to the GUI
void sendGUIMessage(const std::string& msgType, const std::string& msgValue)
{
    SofaGUI *gui = GUIManager::getGUI();
    if (!gui)
    {
        printf("<PYTHON> ERROR sendGUIMessage(%s,%s): no GUI !!\n",msgType.c_str(),msgValue.c_str());
        return;
    }
    gui->sendMessage(msgType,msgValue);
}

*/







// Méthodes du module
PyMethodDef SofaModuleMethods[] =
{
//    { "HelloCWorld", module_HelloCWorld, METH_VARARGS, "Helloworld func (with a string argument).\n" },
//    { "createObject", module_createObject, METH_VARARGS, "Create a DummyClass object.\n" },
    {0,0,0,0}
};
