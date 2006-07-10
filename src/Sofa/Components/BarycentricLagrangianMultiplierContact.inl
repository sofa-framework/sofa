#ifndef SOFA_COMPONENTS_BARYCENTRICLAGRANGIANMULTIPLIERCONTACT_INL
#define SOFA_COMPONENTS_BARYCENTRICLAGRANGIANMULTIPLIERCONTACT_INL

#include "BarycentricLagrangianMultiplierContact.h"

namespace Sofa
{

namespace Components
{

using namespace Common;
using namespace Collision;
using Graph::GNode;

template < class TCollisionModel1, class TCollisionModel2 >
BarycentricLagrangianMultiplierContact<TCollisionModel1,TCollisionModel2>::BarycentricLagrangianMultiplierContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
    : model1(model1), model2(model2), intersectionMethod(intersectionMethod), mapper1(model1), mapper2(model2), ff(NULL), parent(NULL)
{
}

template < class TCollisionModel1, class TCollisionModel2 >
BarycentricLagrangianMultiplierContact<TCollisionModel1,TCollisionModel2>::~BarycentricLagrangianMultiplierContact()
{
    if (ff!=NULL)
    {
        if (parent!=NULL) parent->removeObject(ff);
        delete ff;
    }
}

template < class TCollisionModel1, class TCollisionModel2 >
void BarycentricLagrangianMultiplierContact<TCollisionModel1,TCollisionModel2>::setDetectionOutputs(const std::vector<DetectionOutput*>& outputs)
{
    // We need to remove duplicate contacts
    const double minDist2 = 0.01f;
    std::vector<DetectionOutput*> contacts;
    contacts.reserve(outputs.size());
    for (std::vector<DetectionOutput*>::const_iterator it = outputs.begin(); it!=outputs.end(); it++)
    {
        DetectionOutput* o = *it;
        bool found = false;
        for (unsigned int i=0; i<contacts.size() && !found; i++)
        {
            DetectionOutput* p = contacts[i];
            if ((o->point[0]-p->point[0]).norm2()+(o->point[1]-p->point[1]).norm2() < minDist2)
                found = true;
        }
        if (!found)
            contacts.push_back(o);
    }
    if (contacts.size()<outputs.size())
    {
        //std::cout << "Removed " << (outputs.size()-contacts.size()) <<" / " << outputs.size() << " collision points." << std::endl;
    }
    if (ff==NULL)
    {
        MechanicalModel1* mmodel1 = mapper1.createMapping();
        MechanicalModel2* mmodel2 = mapper2.createMapping();
        ff = new LagrangianMultiplierContactConstraint<Vec3Types>(mmodel1,mmodel2);
    }

    int size = contacts.size();
    ff->clear(size);
    mapper1.resize(size);
    mapper2.resize(size);
    int i = 0;
    for (std::vector<DetectionOutput*>::const_iterator it = contacts.begin(); it!=contacts.end(); it++, i++)
    {
        DetectionOutput* o = *it;
        CollisionElement1* elem1 = static_cast<CollisionElement1*>(o->elem.first);
        CollisionElement2* elem2 = static_cast<CollisionElement2*>(o->elem.second);
        int index1 = elem1->getIndex();
        int index2 = elem2->getIndex();
        // Create mapping for first point
        index1 = mapper1.addPoint(o->point[0], index1);
        // Create mapping for second point
        index2 = mapper2.addPoint(o->point[1], index2);
        if (model1->isStatic() || model2->isStatic()) // create stiffer springs for static models as only half of the force is really applied
            ff->addContact(index1, index2, o->normal, intersectionMethod->getContactDistance(), 300, 0.00f, 0.00f); /// \todo compute stiffness and damping
        else
            ff->addContact(index1, index2, o->normal, intersectionMethod->getContactDistance(), 150, 0.00f, 0.00f); /// \todo compute stiffness and damping
    }
    // Update mappings
    mapper1.update();
    mapper2.update();
}

template < class TCollisionModel1, class TCollisionModel2 >
void BarycentricLagrangianMultiplierContact<TCollisionModel1,TCollisionModel2>::createResponse(Abstract::BaseContext* group)
{
    if (ff!=NULL)
    {
        if (parent!=NULL) parent->removeObject(ff);
        parent = group;
        if (parent!=NULL)
        {
            //std::cout << "Attaching contact response to "<<parent->getName()<<std::endl;
            parent->addObject(ff);
        }
    }
}

template < class TCollisionModel1, class TCollisionModel2 >
void BarycentricLagrangianMultiplierContact<TCollisionModel1,TCollisionModel2>::removeResponse()
{
    if (ff!=NULL)
    {
        if (parent!=NULL)
        {
            //std::cout << "Removing contact response from "<<parent->getName()<<std::endl;
            parent->removeObject(ff);
        }
        parent = NULL;
    }
}

template < class TCollisionModel1, class TCollisionModel2 >
void BarycentricLagrangianMultiplierContact<TCollisionModel1,TCollisionModel2>::draw()
{
//	if (dynamic_cast<Abstract::VisualModel*>(ff)!=NULL)
//		dynamic_cast<Abstract::VisualModel*>(ff)->draw();
}

} // namespace Components

} // namespace Sofa

#endif
