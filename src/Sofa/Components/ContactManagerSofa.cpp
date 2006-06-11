#include "ContactManagerSofa.h"

#include "Common/ObjectFactory.h"

namespace Sofa
{

namespace Components
{

using namespace Collision;

void create(ContactManagerSofa*& obj, ObjectDescription* arg)
{
    obj = new ContactManagerSofa(arg->getAttribute("response","default"));
}

SOFA_DECL_CLASS(ContactManagerSofa)

Creator<ObjectFactory, ContactManagerSofa> ContactManagerSofaClass("CollisionResponse");

ContactManagerSofa::ContactManagerSofa(const std::string& contacttype)
    : contacttype(contacttype)
{
}

ContactManagerSofa::~ContactManagerSofa()
{
    // HACK: do not delete contacts as they might point to forcefields that are already deleted
    // FIX crash on unload bug. -- J. Allard
    //clear();
}

void ContactManagerSofa::clear()
{
    for (std::vector<Contact*>::iterator it=contactVec.begin(); it!=contactVec.end(); ++it)
        delete *it;
    contactVec.clear();
    contactMap.clear();
}

void ContactManagerSofa::createContacts(const std::vector<DetectionOutput*>& outputs)
{
    outputsMap.clear();
    for (std::vector<DetectionOutput*>::const_iterator it = outputs.begin(); it!=outputs.end(); ++it)
    {
        DetectionOutput* o = *it;
        outputsMap[std::make_pair(o->elem.first->getCollisionModel(),o->elem.second->getCollisionModel())].push_back(o);
    }
    // then remove any inactive contacts or add any new contact
    std::map< std::pair<Abstract::CollisionModel*,Abstract::CollisionModel*>, std::vector<DetectionOutput*> >::iterator outputsIt = outputsMap.begin();
    std::map< std::pair<Abstract::CollisionModel*,Abstract::CollisionModel*>, Contact* >::iterator contactIt = contactMap.begin();
    int nbContact = 0;
    while (outputsIt!=outputsMap.end() || contactIt!=contactMap.end())
    {
        if (outputsIt!=outputsMap.end() && (contactIt == contactMap.end() || outputsIt->first < contactIt->first))
        {
            // new contact
            std::cout << "Creation new "<<contacttype<<" contact"<<std::endl;
            Contact* contact = Contact::Create(contacttype, outputsIt->first.first, outputsIt->first.second, intersectionMethod);
            if (contact == NULL) std::cerr << "Contact creation failed"<<std::endl;
            else
            {
                contactMap[std::make_pair(outputsIt->first.first, outputsIt->first.second)] = contact;
                contact->setDetectionOutputs(outputsIt->second);
                ++nbContact;
            }
            ++outputsIt;
        }
        else if (contactIt!=contactMap.end() && (outputsIt == outputsMap.end() || contactIt->first < outputsIt->first))
        {
            // inactive contact
            //std::cout << "Deleting inactive "<<contacttype<<" contact"<<std::endl;
            std::map< std::pair<Abstract::CollisionModel*,Abstract::CollisionModel*>, Contact* >::iterator contactIt2 = contactIt;
            ++contactIt2;
            delete contactIt->second;
            contactMap.erase(contactIt);
            contactIt = contactIt2;
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
    contactVec.clear();
    contactVec.reserve(nbContact);
    contactIt = contactMap.begin();
    while (contactIt!=contactMap.end())
    {
        contactVec.push_back(contactIt->second);
        ++contactIt;
    }
}

void ContactManagerSofa::draw()
{
    for (std::vector<Contact*>::iterator it = contactVec.begin(); it!=contactVec.end(); it++)
    {
        if (dynamic_cast<Abstract::VisualModel*>(*it)!=NULL)
            dynamic_cast<Abstract::VisualModel*>(*it)->draw();
    }
}

} // namespace Components

} // namespace Sofa
