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
#include <sofa/component/collision/DefaultContactManager.h>
#include <sofa/core/ObjectFactory.h>
using std::cerr;
using std::endl;


namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(DefaultContactManager)

int DefaultContactManagerClass = core::RegisterObject("Default class to create reactions to the collisions")
        .add< DefaultContactManager >()
        .addAlias("CollisionResponse")
        ;

DefaultContactManager::DefaultContactManager()
    : response(initData(&response, std::string("default"), "response", "contact response class"))
{
}

DefaultContactManager::~DefaultContactManager()
{
    // Contacts are now attached to the graph.
    // So they will be deleted by the DeleteVisitor
    // FIX crash on unload bug. -- J. Allard
    //clear();
}

void DefaultContactManager::clear()
{
    for (sofa::helper::vector<core::componentmodel::collision::Contact*>::iterator it=contacts.begin(); it!=contacts.end(); ++it)
    {
        (*it)->removeResponse();
        (*it)->cleanup();
        delete *it;
    }
    contacts.clear();
    contactMap.clear();
}

void DefaultContactManager::createContacts(DetectionOutputMap& outputsMap)
{
    //cerr<<"DefaultContactManager::createContacts"<<endl;

    //outputsMap.clear();
    //for (sofa::helper::vector<core::componentmodel::collision::DetectionOutput*>::const_iterator it = outputs.begin(); it!=outputs.end(); ++it)
    //{
    //	core::componentmodel::collision::DetectionOutput* o = *it;
    //	outputsMap[std::make_pair(o->elem.first.getCollisionModel(),o->elem.second.getCollisionModel())].push_back(o);
    //}

    // Remove any inactive contacts or add any new contact
    DetectionOutputMap::iterator outputsIt = outputsMap.begin();
    std::map< std::pair<core::CollisionModel*,core::CollisionModel*>, core::componentmodel::collision::Contact* >::iterator contactIt = contactMap.begin();
    int nbContact = 0;
    //static DetectionOutputVector emptyContacts;
    while (outputsIt!=outputsMap.end() || contactIt!=contactMap.end())
    {
        if (outputsIt!=outputsMap.end() && (contactIt == contactMap.end() || outputsIt->first < contactIt->first))
        {
            // new contact
            //std::cout << "Creation new "<<contacttype<<" contact"<<std::endl;
            core::CollisionModel* model1 = outputsIt->first.first;
            core::CollisionModel* model2 = outputsIt->first.second;
            std::string response1 = model1->getContactResponse();
            std::string response2 = model2->getContactResponse();
            std::string responseUsed = response.getValue();
            if (!response1.empty()) responseUsed = response1;
            else if (!response2.empty()) responseUsed = response2;
            core::componentmodel::collision::Contact* contact = core::componentmodel::collision::Contact::Create(responseUsed, model1, model2, intersectionMethod);
            if (contact == NULL) std::cerr << "Contact "<<responseUsed<<" between " << model1->getClassName()<<" and "<<model2->getClassName() << " creation failed"<<std::endl;
            else
            {
                contactMap[std::make_pair(model1, model2)] = contact;
                contact->setName(model1->getName()+std::string("-")+model2->getName());
                contact->f_printLog.setValue(this->f_printLog.getValue());
                contact->init();
                contact->setDetectionOutputs(outputsIt->second);
                ++nbContact;
            }
            ++outputsIt;
        }
        else if (contactIt!=contactMap.end() && (outputsIt == outputsMap.end() || contactIt->first < outputsIt->first))
        {
            // inactive contact
            //std::cout << "Deleting inactive "<<contacttype<<" contact"<<std::endl;
            if (contactIt->second->keepAlive())
            {
                contactIt->second->setDetectionOutputs(NULL);
                ++nbContact;
                ++contactIt;
            }
            else
            {
                std::map< std::pair<core::CollisionModel*,core::CollisionModel*>, core::componentmodel::collision::Contact* >::iterator contactIt2 = contactIt;
                ++contactIt2;
                contactIt->second->removeResponse();
                contactIt->second->cleanup();
                delete contactIt->second;
                contactMap.erase(contactIt);
                contactIt = contactIt2;
            }
        }
        else
        {
            // corresponding contact and outputs
            contactIt->second->setDetectionOutputs(outputsIt->second);
            ++nbContact;
            ++outputsIt;
            ++contactIt;
        }
    }
    // now update contactVec
    contacts.clear();
    contacts.reserve(nbContact);
    contactIt = contactMap.begin();
    while (contactIt!=contactMap.end())
    {
        contacts.push_back(contactIt->second);
        ++contactIt;
    }
}

void DefaultContactManager::draw()
{
    for (sofa::helper::vector<core::componentmodel::collision::Contact*>::iterator it = contacts.begin(); it!=contacts.end(); it++)
    {
        if ((*it)!=NULL)
            (*it)->draw();
    }
}

} // namespace collision

} // namespace component

} // namespace sofa

