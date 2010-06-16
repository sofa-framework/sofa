/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_COLLISION_FRICTIONCONTACT_INL
#define SOFA_COMPONENT_COLLISION_FRICTIONCONTACT_INL

#include <sofa/component/collision/FrictionContact.h>
#include <sofa/component/collision/DefaultContactManager.h>
#include <sofa/component/collision/BarycentricContactMapper.h>
#include <sofa/component/collision/IdentityContactMapper.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace core::collision;
using simulation::Node;




template < class TCollisionModel1, class TCollisionModel2 >
FrictionContact<TCollisionModel1,TCollisionModel2>::FrictionContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
    : model1(model1), model2(model2), intersectionMethod(intersectionMethod), c(NULL), parent(NULL)
    , mu (initData(&mu, 0.8, "mu", "friction coefficient (0 for frictionless contacts)"))
{
    selfCollision = ((core::CollisionModel*)model1 == (core::CollisionModel*)model2);
    mapper1.setCollisionModel(model1);
    if (!selfCollision) mapper2.setCollisionModel(model2);
}

template < class TCollisionModel1, class TCollisionModel2 >
FrictionContact<TCollisionModel1,TCollisionModel2>::~FrictionContact()
{
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
        if (!selfCollision) mapper2.cleanup();
    }
}


template < class TCollisionModel1, class TCollisionModel2 >
void FrictionContact<TCollisionModel1,TCollisionModel2>::setDetectionOutputs(OutputVector* o)
{
    TOutputVector& outputs = *static_cast<TOutputVector*>(o);
    // We need to remove duplicate contacts
    const double minDist2 = 0.00000001f;
    std::vector<DetectionOutput*> contacts;
    contacts.reserve(outputs.size());

    const double mu = this->mu.getValue();
    // Checks if friction is considered
    if (mu < 0.0 || mu > 1.0)
        serr << sendl << "Error: mu has to take values between 0.0 and 1.0" << sendl;

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
        //sout << "Removed " << (outputs.size()-contacts.size()) <<" / " << outputs.size() << " collision points." << sendl;
    }

    if (c==NULL)
    {
        // Get the mechanical model from mapper1 to fill the constraint vector
        MechanicalState1* mmodel1 = mapper1.createMapping();
        // Get the mechanical model from mapper2 to fill the constraints vector
        MechanicalState2* mmodel2 = selfCollision ? mmodel1 : mapper2.createMapping();
        c = new constraintset::UnilateralInteractionConstraint<Vec3Types>(mmodel1, mmodel2);
        c->setName( getName() );
    }

    int size = contacts.size();
    c->clear(size);
    if (selfCollision) mapper1.resize(2*size);
    else
    {
        mapper1.resize(size);
        mapper2.resize(size);
    }
    int i = 0;
    const double d0 = intersectionMethod->getContactDistance() + model1->getProximity() + model2->getProximity(); // - 0.001;
    std::vector< std::pair< std::pair<int, int>, double > > mappedContacts;
    mappedContacts.resize(contacts.size());
    for (std::vector<DetectionOutput*>::const_iterator it = contacts.begin(); it!=contacts.end(); it++, i++)
    {
        DetectionOutput* o = *it;
        CollisionElement1 elem1(o->elem.first);
        CollisionElement2 elem2(o->elem.second);
        int index1 = elem1.getIndex();
        int index2 = elem2.getIndex();
        typename DataTypes1::Real r1 = 0.0;
        typename DataTypes2::Real r2 = 0.0;
        //double constraintValue = ((o->point[1] - o->point[0]) * o->normal) - intersectionMethod->getContactDistance();

        // Create mapping for first point
        index1 = mapper1.addPoint(o->point[0], index1, r1);
        // Create mapping for second point
        index2 = selfCollision ? mapper1.addPoint(o->point[1], index2, r2) : mapper2.addPoint(o->point[1], index2, r2);
        double distance = d0 + r1 + r2;

        mappedContacts[i].first.first = index1;
        mappedContacts[i].first.second = index2;
        mappedContacts[i].second = distance;
    }

    // Update mappings
    mapper1.update();
    mapper1.updateXfree();
    if (!selfCollision) mapper2.update();
    if (!selfCollision) mapper2.updateXfree();
    i=0;
    for (std::vector<DetectionOutput*>::const_iterator it = contacts.begin(); it!=contacts.end(); it++, i++)
    {
        DetectionOutput* o = *it;
        int index1 = mappedContacts[i].first.first;
        int index2 = mappedContacts[i].first.second;
        double distance = mappedContacts[i].second;
#if 0
        std::cout << "P0     = " << (*c->getObject1()->getX())[index1];
        if (((*c->getObject1()->getX())[index1] - o->point[0]).norm() > 1.0e-10) std::cout << " ( " << (*c->getObject1()->getX())[index1] - o->point[0] << " )";
        std::cout << std::endl;
        std::cout << "P0free = " << (*c->getObject1()->getXfree())[index1];
#ifdef DETECTIONOUTPUT_FREEMOTION
        if (((*c->getObject1()->getXfree())[index1] - o->freePoint[0]).norm() > 1.0e-10) std::cout << " ( " << (*c->getObject1()->getXfree())[index1] - o->freePoint[0] << " )";
        std::cout << std::endl;
#else
        if (((*c->getObject1()->getXfree())[index1] - o->point[0]).norm() > 1.0e-10) std::cout << " ( " << (*c->getObject1()->getXfree())[index1] - o->point[0] << " )";
        std::cout << std::endl;
#endif

        std::cout << "P1     = " << (*c->getObject2()->getX())[index2];
        if (((*c->getObject2()->getX())[index2] - o->point[1]).norm() > 1.0e-10) std::cout << " ( " << (*c->getObject2()->getX())[index2] - o->point[2] << " )";
        std::cout << std::endl;
        std::cout << "P1free = " << (*c->getObject2()->getXfree())[index2];
#ifdef DETECTIONOUTPUT_FREEMOTION
        if (((*c->getObject2()->getXfree())[index2] - o->freePoint[1]).norm() > 1.0e-10) std::cout << " ( " << (*c->getObject2()->getXfree())[index2] - o->freePoint[1] << " )";
        std::cout << std::endl;
#else
        if (((*c->getObject2()->getXfree())[index2] - o->point[1]).norm() > 1.0e-10) std::cout << " ( " << (*c->getObject2()->getXfree())[index2] - o->point[1] << " )";
        std::cout << std::endl;
#endif
#endif

        // Polynome de Cantor de N� sur N bijectif f(x,y)=((x+y)^2+3x+y)/2
        long index = cantorPolynomia(o->id /*cantorPolynomia(index1, index2)*/,id);
//#ifdef DETECTIONOUTPUT_FREEMOTION
//		c->addContact(mu, o->normal, o->point[1], o->point[0], distance, index1, index2, o->freePoint[1], o->freePoint[0], index);
//#else
        // as we called updateXfree on the mappings, the contact class can now use the Xfree value of the contact points instead of relying on the collision detection method to fill freePoint[0] and freePoint[1]
        //c->addContact(mu, o->normal, o->point[1], o->point[0], distance, index1, index2, index);
        c->addContact(mu, o->normal, distance, index1, index2, index, o->id);
//#endif
    }
}

template < class TCollisionModel1, class TCollisionModel2 >
void FrictionContact<TCollisionModel1,TCollisionModel2>::createResponse(core::objectmodel::BaseContext* group)
{
    if (c!=NULL)
    {
        if (parent!=NULL)
        {
            parent->removeObject(this);
            parent->removeObject(c);
        }
        parent = group;
        if (parent!=NULL)
        {
            //sout << "Attaching contact response to "<<parent->getName()<<sendl;
            parent->addObject(this);
            parent->addObject(c);
        }
    }
}

template < class TCollisionModel1, class TCollisionModel2 >
void FrictionContact<TCollisionModel1,TCollisionModel2>::removeResponse()
{
    if (c!=NULL)
    {
        mapper1.resize(0);
        mapper2.resize(0);
        if (parent!=NULL)
        {
            //sout << "Removing contact response from "<<parent->getName()<<sendl;
            parent->removeObject(this);
            parent->removeObject(c);
        }
        parent = NULL;
    }
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
