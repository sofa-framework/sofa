/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <SofaBaseCollision/DefaultContactManager.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/Tag.h>


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
    , responseParams(initData(&responseParams, "responseParams", "contact response parameters (syntax: name1=value1&name2=value2&...)"))
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
    std::set<std::string> listResponse;
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

void DefaultContactManager::reset()
{
    cleanup();
}

void DefaultContactManager::createContacts(const DetectionOutputMap& outputsMap)
{
    using core::CollisionModel;
    using core::collision::Contact;

    int nbContact = 0;

    // First iterate on the collision detection outputs and look for existing or new contacts
    for (DetectionOutputMap::const_iterator outputsIt = outputsMap.begin(),
        outputsItEnd = outputsMap.end(); outputsIt != outputsItEnd ; ++outputsIt)
    {
        std::pair<ContactMap::iterator,bool> contactInsert =
            contactMap.insert(ContactMap::value_type(outputsIt->first,Contact::SPtr()));
        ContactMap::iterator contactIt = contactInsert.first;
        if (contactInsert.second)
        {
            // new contact
            //sout << "Creation new "<<contacttype<<" contact"<<sendl;
            CollisionModel* model1 = outputsIt->first.first;
            CollisionModel* model2 = outputsIt->first.second;
            std::string responseUsed = getContactResponse(model1, model2);

            // We can create rules in order to not respond to specific collisions
            if (!responseUsed.compare("null"))
            {
                contactMap.erase(contactIt);
            }
            else
            {
                Contact::SPtr contact = Contact::Create(responseUsed, model1, model2, intersectionMethod,
                    notMuted());

                if (contact == NULL)
                {
                    std::string model1class = model1->getClassName();
                    std::string model2class = model2->getClassName();
                    int count = ++errorMsgCount[std::make_pair(responseUsed,
                        std::make_pair(model1class, model2class))];
                    if (count <= 10)
                    {
                        serr << "Contact " << responseUsed << " between " << model1->getClassName()
                            << " and " << model2->getClassName() << " creation failed" << sendl;
                        if (count == 1)
                        {
                            serr << "Supported models for contact " << responseUsed << ":" << sendl;
                            for (Contact::Factory::const_iterator it =
                                Contact::Factory::getInstance()->begin(),
                                itend = Contact::Factory::getInstance()->end(); it != itend; ++it)
                            {
                                if (it->first != responseUsed) continue;
                                serr << "   " << helper::gettypename(it->second->type()) << sendl;
                            }
                            serr << sendl;
                        }
                        if (count == 10) serr << "further messages suppressed" << sendl;
                    }
                    contactMap.erase(contactIt);
                }
                else
                {
                    contactIt->second = contact;
                    contact->setName(model1->getName() + std::string("-") + model2->getName());
                    setContactTags(model1, model2, contact);
                    contact->f_printLog.setValue(notMuted());
                    contact->init();
                    contact->setDetectionOutputs(outputsIt->second);
                    ++nbContact;
                }
            }
        }
        else
        {
            // pre-existing and still active contact
            contactIt->second->setDetectionOutputs(outputsIt->second);
            ++nbContact;
        }
    }

    // Then look at previous contacts
    // and remove inactive contacts
    for (ContactMap::iterator contactIt = contactMap.begin(), contactItEnd = contactMap.end();
        contactIt != contactItEnd;)
    {
        bool remove = false;
        DetectionOutputMap::const_iterator outputsIt = outputsMap.find(contactIt->first);
        if (outputsIt == outputsMap.end())
        {
            // inactive contact
            if (contactIt->second->keepAlive())
            {
                contactIt->second->setDetectionOutputs(NULL);
                ++nbContact;
            }
            else
            {
                remove = true;
            }
        }
        if (remove)
        {
            if (contactIt->second)
            {
                contactIt->second->removeResponse();
                contactIt->second->cleanup();
                contactIt->second.reset();
            }
            ContactMap::iterator eraseIt = contactIt;
            ++contactIt;
            contactMap.erase(eraseIt);
        }
        else
        {
            ++contactIt;
        }
    }

    // now update contactVec
    contacts.clear();
    contacts.reserve(nbContact);
    for (ContactMap::const_iterator contactIt = contactMap.begin(), contactItEnd = contactMap.end();
        contactIt != contactItEnd; ++contactIt)
    {
        contacts.push_back(contactIt->second);
    }

    // compute number of contacts attached to each collision model
    std::map< CollisionModel*, int > nbContactsMap;
    for (unsigned int i = 0; i < contacts.size(); ++i)
    {
        std::pair< CollisionModel*, CollisionModel* > cms = contacts[i]->getCollisionModels();
        nbContactsMap[cms.first]++;
        if (cms.second != cms.first)
            nbContactsMap[cms.second]++;
    }

    // TODO: this is VERY inefficient, should be replaced with a visitor
    helper::vector< CollisionModel* > collisionModels;
    simulation::Node* context = dynamic_cast< simulation::Node* >(getContext());
    context->getTreeObjects< CollisionModel >(&collisionModels);

    for (unsigned int i = 0; i < collisionModels.size(); ++i)
    {
        collisionModels[i]->setNumberOfContacts(nbContactsMap[collisionModels[i]]);
    }
}

std::string DefaultContactManager::getContactResponse(core::CollisionModel* model1, core::CollisionModel* model2)
{
    std::string responseUsed = response.getValue().getSelectedItem();
    std::string params = responseParams.getValue();
    if (!params.empty())
    {
        responseUsed += '?';
        responseUsed += params;
    }
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
    for (sofa::helper::vector<core::collision::Contact::SPtr>::iterator it = contacts.begin(); it!=contacts.end(); ++it)
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
                (*it)->removeResponse();
                (*it)->cleanup();
                it->reset();
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

                erase_it->second->removeResponse();
                erase_it->second->cleanup();
                erase_it->second.reset();
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

void DefaultContactManager::setContactTags(core::CollisionModel* model1, core::CollisionModel* model2, core::collision::Contact::SPtr contact)
{
    sofa::core::objectmodel::TagSet tagsm1 = model1->getTags();
    sofa::core::objectmodel::TagSet tagsm2 = model2->getTags();
    sofa::core::objectmodel::TagSet::iterator it;
    for(it=tagsm1.begin(); it != tagsm1.end(); ++it)
        contact->addTag(*it);
    for(it=tagsm2.begin(); it!=tagsm2.end(); ++it)
        contact->addTag(*it);
}

} // namespace collision

} // namespace component

} // namespace sofa
