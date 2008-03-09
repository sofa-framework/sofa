#ifndef SOFA_COMPONENT_COLLISION_FRICTIONCONTACT_INL
#define SOFA_COMPONENT_COLLISION_FRICTIONCONTACT_INL

#include <sofa/component/collision/FrictionContact.h>
#include <sofa/component/collision/DefaultContactManager.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace core::componentmodel::collision;
using simulation::tree::GNode;
using std::cout;
using std::cerr;
using std::endl;

template < class TCollisionModel1, class TCollisionModel2 >
FrictionContact<TCollisionModel1,TCollisionModel2>::FrictionContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
    : model1(model1), model2(model2), intersectionMethod(intersectionMethod), c(NULL), parent(NULL)
{
    mu = 0.6;
}

template < class TCollisionModel1, class TCollisionModel2 >
FrictionContact<TCollisionModel1,TCollisionModel2>::~FrictionContact()
{
    if (c!=NULL)
    {
        if (parent!=NULL) parent->removeObject(c);
        delete c;
    }
}
template < class TCollisionModel1, class TCollisionModel2 >
void FrictionContact<TCollisionModel1,TCollisionModel2>::cleanup()
{
    if (c!=NULL)
    {
        c->cleanup();
        if (parent!=NULL)
            parent->removeObject(c);
        delete c;
        parent = NULL;
        c = NULL;
        mapper1.cleanup();
        mapper2.cleanup();
    }
}


#ifdef DETECTIONOUTPUT_FREEMOTION
template < class TCollisionModel1, class TCollisionModel2 >
void FrictionContact<TCollisionModel1,TCollisionModel2>::setDetectionOutputs(OutputVector* o)
{
    TOutputVector& outputs = *static_cast<TOutputVector*>(o);
    // We need to remove duplicate contacts
    const double minDist2 = 0.0000001f;
    std::vector<DetectionOutput*> contacts;
    contacts.reserve(outputs.size());

    int SIZE = outputs.size();

    for (int cpt=0; cpt<SIZE; cpt++)
    {
        DetectionOutput* o = &outputs[cpt];

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

    if (c==NULL)
    {
        // Get the mechanical model from mapper1 to fill the constraint vector
        MechanicalState1* mmodel1 = mapper1.createMapping(model1);
        // Get the mechanical model from mapper2 to fill the constraints vector
        MechanicalState2* mmodel2 = mapper2.createMapping(model2);
        c = new constraint::UnilateralInteractionConstraint<Vec3Types>(mmodel1, mmodel2);
    }

    int size = contacts.size();
    c->clear(size);
    mapper1.resize(size);
    mapper2.resize(size);
    int i = 0;
    for (std::vector<DetectionOutput*>::const_iterator it = contacts.begin(); it!=contacts.end(); it++, i++)
    {
        DetectionOutput* o = *it;
        CollisionElement1 elem1(o->elem.first);
        CollisionElement2 elem2(o->elem.second);
        int index1 = elem1.getIndex();
        int index2 = elem2.getIndex();

        //double constraintValue = ((o->point[1] - o->point[0]) * o->normal) - intersectionMethod->getContactDistance();

        // Create mapping for first point
        index1 = mapper1.addPoint(o->point[0], index1);
        // Create mapping for second point
        index2 = mapper2.addPoint(o->point[1], index2);
        // Checks if friction is considered
        if (mu < 0.0 || mu > 1.0)
            cerr << endl << "Error: mu has to take values between 0.0 and 1.0" << endl;

        // Polynome de Cantor de N² sur N bijectif f(x,y)=((x+y)^2+3x+y)/2
        long index = cantorPolynomia(cantorPolynomia(index1, index2),id);
        c->addContact(mu, o->normal, o->point[1], o->point[0], intersectionMethod->getContactDistance(), index1, index2, o->freePoint[1], o->freePoint[0], index);
    }
    // Update mappings
    mapper1.update();
    mapper2.update();
}
#else
template < class TCollisionModel1, class TCollisionModel2 >
void FrictionContact<TCollisionModel1,TCollisionModel2>::setDetectionOutputs(OutputVector*)
{
    cerr << endl << "ERROR: FrictionContact requires DETECTIONOUTPUT_FREEMOTION to be defined in DetectionOutput.h" << endl;
}
#endif

template < class TCollisionModel1, class TCollisionModel2 >
void FrictionContact<TCollisionModel1,TCollisionModel2>::createResponse(core::objectmodel::BaseContext* group)
{
    if (c!=NULL)
    {
        if (parent!=NULL) parent->removeObject(c);
        parent = group;
        if (parent!=NULL)
        {
            parent->addObject(c);
        }
    }
}

template < class TCollisionModel1, class TCollisionModel2 >
void FrictionContact<TCollisionModel1,TCollisionModel2>::removeResponse()
{
    if (c!=NULL)
    {
        if (parent!=NULL)
        {
            parent->removeObject(c);
        }
        parent = NULL;
    }
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
