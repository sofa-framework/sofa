#include <sofa/component/collision/DefaultContactManager.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(DefaultContactManager)

int DefaultContactManagerClass = core::RegisterObject("TODO")
        .add< DefaultContactManager >()
        .addAlias("CollisionResponse")
        ;

DefaultContactManager::DefaultContactManager()
    : response(dataField(&response, std::string("default"), "response", "contact response class"))
{
}

DefaultContactManager::~DefaultContactManager()
{
    // HACK: do not delete contacts as they might point to forcefields that are already deleted
    // FIX crash on unload bug. -- J. Allard
    //clear();
}

void DefaultContactManager::clear()
{
    for (std::vector<core::componentmodel::collision::Contact*>::iterator it=contactVec.begin(); it!=contactVec.end(); ++it)
        delete *it;
    contactVec.clear();
    contactMap.clear();
}

void DefaultContactManager::createContacts(const std::vector<core::componentmodel::collision::DetectionOutput*>& outputs)
{
    outputsMap.clear();
    for (std::vector<core::componentmodel::collision::DetectionOutput*>::const_iterator it = outputs.begin(); it!=outputs.end(); ++it)
    {
        core::componentmodel::collision::DetectionOutput* o = *it;
        outputsMap[std::make_pair(o->elem.first.getCollisionModel(),o->elem.second.getCollisionModel())].push_back(o);
    }
    // then remove any inactive contacts or add any new contact
    std::map< std::pair<core::CollisionModel*,core::CollisionModel*>, std::vector<core::componentmodel::collision::DetectionOutput*> >::iterator outputsIt = outputsMap.begin();
    std::map< std::pair<core::CollisionModel*,core::CollisionModel*>, core::componentmodel::collision::Contact* >::iterator contactIt = contactMap.begin();
    int nbContact = 0;
    while (outputsIt!=outputsMap.end() || contactIt!=contactMap.end())
    {
        if (outputsIt!=outputsMap.end() && (contactIt == contactMap.end() || outputsIt->first < contactIt->first))
        {
            // new contact
            //std::cout << "Creation new "<<contacttype<<" contact"<<std::endl;
            core::componentmodel::collision::Contact* contact = core::componentmodel::collision::Contact::Create(response.getValue(), outputsIt->first.first, outputsIt->first.second, intersectionMethod);
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
            std::map< std::pair<core::CollisionModel*,core::CollisionModel*>, core::componentmodel::collision::Contact* >::iterator contactIt2 = contactIt;
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

void DefaultContactManager::draw()
{
    for (std::vector<core::componentmodel::collision::Contact*>::iterator it = contactVec.begin(); it!=contactVec.end(); it++)
    {
        if (dynamic_cast<core::VisualModel*>(*it)!=NULL)
            dynamic_cast<core::VisualModel*>(*it)->draw();
    }
}

} // namespace collision

} // namespace component

} // namespace sofa

