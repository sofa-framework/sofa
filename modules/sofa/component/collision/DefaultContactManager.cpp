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
#include <sofa/component/collision/DefaultContactManager.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>




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
    : response(initData(&response, "response", "contact response class"))
{
}

DefaultContactManager::~DefaultContactManager()
{
    // Contacts are now attached to the graph.
    // So they will be deleted by the DeleteVisitor
    // FIX crash on unload bug. -- J. Allard
    //clear();
}

sofa::helper::OptionsGroup DefaultContactManager::initializeResponseOptions(core::collision::Pipeline *pipeline)
{
    helper::set<std::string> listResponse;
    if (pipeline) listResponse=pipeline->getResponseList();
    else
    {
        core::collision::Contact::Factory::iterator it;
        for (it=core::collision::Contact::Factory::getInstance()->begin(); it!=core::collision::Contact::Factory::getInstance()->end(); ++it)
        {
            listResponse.insert(it->first);
        }
    }
    sofa::helper::OptionsGroup responseOptions(listResponse);
    if (listResponse.find("default") != listResponse.end())
        responseOptions.setSelectedItem("default");
    return responseOptions;
}

void DefaultContactManager::init()
{
    if (response.getValue().size() == 0)
    {
        core::collision::Pipeline *pipeline=static_cast<simulation::Node*>(getContext())->collisionPipeline;
        response.setValue(initializeResponseOptions(pipeline));
    }
}
void DefaultContactManager::cleanup()
{
    for (sofa::helper::vector<core::collision::Contact::SPtr>::iterator it=contacts.begin(); it!=contacts.end(); ++it)
    {
        (*it)->removeResponse();
        (*it)->cleanup();
        //delete *it;
        it->reset();
    }
    contacts.clear();
    contactMap.clear();
}

void DefaultContactManager::createContacts(DetectionOutputMap& outputsMap)
{
    //serr<<"DefaultContactManager::createContacts"<<sendl;

    //outputsMap.clear();
    //for (sofa::helper::vector<core::collision::DetectionOutput*>::const_iterator it = outputs.begin(); it!=outputs.end(); ++it)
    //{
    //	core::collision::DetectionOutput* o = *it;
    //	outputsMap[std::make_pair(o->elem.first.getCollisionModel(),o->elem.second.getCollisionModel())].push_back(o);
    //}

    // Remove any inactive contacts or add any new contact
    DetectionOutputMap::iterator outputsIt = outputsMap.begin();
    std::map< std::pair<core::CollisionModel*,core::CollisionModel*>, core::collision::Contact::SPtr >::iterator contactIt = contactMap.begin();
    int nbContact = 0;
    //static DetectionOutputVector emptyContacts;
    while (outputsIt!=outputsMap.end() || contactIt!=contactMap.end())
    {
        if (outputsIt!=outputsMap.end() && (contactIt == contactMap.end() || outputsIt->first < contactIt->first))
        {
            // new contact
            //sout << "Creation new "<<contacttype<<" contact"<<sendl;
            core::CollisionModel* model1 = outputsIt->first.first;
            core::CollisionModel* model2 = outputsIt->first.second;
            std::string responseUsed = getContactResponse(model1, model2);
            if(! responseUsed.compare("null") ) // We can create rules in order to not respond to specific collisions
            {
                ++outputsIt;
                continue;
            }
            core::collision::Contact::SPtr contact = core::collision::Contact::Create(responseUsed, model1, model2, intersectionMethod);
            if (contact == NULL) serr << "Contact "<<responseUsed<<" between " << model1->getClassName()<<" and "<<model2->getClassName() << " creation failed"<<sendl;
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
            //sout << "Deleting inactive "<<contacttype<<" contact"<<sendl;
            if (contactIt->second->keepAlive())
            {
                contactIt->second->setDetectionOutputs(NULL);
                ++nbContact;
                ++contactIt;
            }
            else
            {
                std::map< std::pair<core::CollisionModel*,core::CollisionModel*>, core::collision::Contact::SPtr >::iterator contactIt2 = contactIt;
                ++contactIt2;
                contactIt->second->removeResponse();
                contactIt->second->cleanup();
                contactIt->second.reset();
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
    // compute number of contacts attached to each collision model
    std::map<core::CollisionModel*,int> nbContactsMap;
    for (unsigned int i=0; i<contacts.size(); ++i)
    {
        std::pair< core::CollisionModel*, core::CollisionModel* > cms = contacts[i]->getCollisionModels();
        nbContactsMap[cms.first]++;
        if (cms.second != cms.first)
            nbContactsMap[cms.second]++;
    }
    sofa::helper::vector<core::CollisionModel*> collisionModels;
    simulation::Node* context = dynamic_cast<simulation::Node*>(getContext());
    context->getTreeObjects<core::CollisionModel>(&collisionModels);
    for (unsigned int i=0; i<collisionModels.size(); ++i)
    {
        collisionModels[i]->setNumberOfContacts(nbContactsMap[collisionModels[i]]);
    }
}

std::string DefaultContactManager::getContactResponse(core::CollisionModel* model1, core::CollisionModel* model2)
{
    std::string responseUsed = response.getValue().getSelectedItem();
    std::string response1 = model1->getContactResponse();
    std::string response2 = model2->getContactResponse();

    if (response1.empty()  &&  response2.empty()) return responseUsed;
    if (response1.empty()  && !response2.empty()) return response2;
    if (!response1.empty() &&  response2.empty()) return response1;

    if (response1 != response2) return responseUsed;
    else return response1;
}

void DefaultContactManager::draw(const core::visual::VisualParams* vparams)
{
    for (sofa::helper::vector<core::collision::Contact::SPtr>::iterator it = contacts.begin(); it!=contacts.end(); it++)
    {
        if ((*it)!=NULL)
            (*it)->draw(vparams);
    }
}


void DefaultContactManager::removeContacts(const ContactVector &c)
{
    ContactVector::const_iterator remove_it = c.begin();
    ContactVector::const_iterator remove_itEnd = c.end();

    ContactVector::iterator it;
    ContactVector::iterator itEnd;

    ContactMap::iterator map_it;
    ContactMap::iterator map_itEnd;

    while (remove_it != remove_itEnd)
    {
        // Whole scene contacts
        it = contacts.begin();
        itEnd = contacts.end();

        while (it != itEnd)
        {
            if (*it == *remove_it)
            {
                contacts.erase(it);
                break;
            }

            ++it;
        }

        // Stored contacts (keeping alive)
        map_it = contactMap.begin();
        map_itEnd = contactMap.end();

        while (map_it != map_itEnd)
        {
            if (map_it->second == *remove_it)
            {
                ContactMap::iterator erase_it = map_it;
                ++map_it;
                contactMap.erase(erase_it);
            }
            else
            {
                ++map_it;
            }
        }

        ++remove_it;
    }

}

} // namespace collision

} // namespace component

} // namespace sofa
