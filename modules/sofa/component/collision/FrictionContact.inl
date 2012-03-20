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
#ifndef SOFA_COMPONENT_COLLISION_FRICTIONCONTACT_INL
#define SOFA_COMPONENT_COLLISION_FRICTIONCONTACT_INL

#include <sofa/component/collision/FrictionContact.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/DefaultContactManager.h>
#include <sofa/component/collision/BarycentricContactMapper.h>
#include <sofa/component/collision/IdentityContactMapper.h>
#include <sofa/component/collision/RigidContactMapper.h>
#include <sofa/simulation/common/Node.h>
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
    : model1(model1)
    , model2(model2)
    , intersectionMethod(intersectionMethod)
    , m_constraint(NULL)
    , parent(NULL)
    , mu (initData(&mu, 0.1, "mu", "friction coefficient (0 for frictionless contacts)"))
    , tol (initData(&tol, 0.0, "tol", "tolerance for the constraints resolution (0 for default tolerance)"))
{
    selfCollision = ((core::CollisionModel*)model1 == (core::CollisionModel*)model2);
    mapper1.setCollisionModel(model1);
    if (!selfCollision) mapper2.setCollisionModel(model2);
    contacts.clear();
    mappedContacts.clear();

}

template < class TCollisionModel1, class TCollisionModel2 >
FrictionContact<TCollisionModel1,TCollisionModel2>::~FrictionContact()
{
}

template < class TCollisionModel1, class TCollisionModel2 >
void FrictionContact<TCollisionModel1,TCollisionModel2>::cleanup()
{
    if (m_constraint)
    {
        m_constraint->cleanup();

        if (parent != NULL)
            parent->removeObject(m_constraint);

        parent = NULL;
        //delete m_constraint;
        m_constraint.reset();

        mapper1.cleanup();

        if (!selfCollision)
            mapper2.cleanup();
    }

    contacts.clear();
    mappedContacts.clear();
}


template < class TCollisionModel1, class TCollisionModel2 >
void FrictionContact<TCollisionModel1,TCollisionModel2>::setDetectionOutputs(OutputVector* o)
{
    TOutputVector& outputs = *static_cast<TOutputVector*>(o);
    // We need to remove duplicate contacts
    const double minDist2 = 0.00000001f;

    contacts.clear();

    if (model1->getContactStiffness(0) == 0 || model2->getContactStiffness(0) == 0)
    {
        serr << "Disabled FrictionContact with " << (outputs.size()) << " collision points." << sendl;
        return;
    }

    contacts.reserve(outputs.size());

    int SIZE = outputs.size();

    // the following procedure cancels the duplicated detection outputs
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
        // DUPLICATED CONTACTS FOUND
        sout << "Removed " << (outputs.size()-contacts.size()) <<" / " << outputs.size() << " collision points." << sendl;
    }
}


template < class TCollisionModel1, class TCollisionModel2 >
void FrictionContact<TCollisionModel1,TCollisionModel2>::activateMappers()
{
    if (!m_constraint)
    {
        // Get the mechanical model from mapper1 to fill the constraint vector
        MechanicalState1* mmodel1 = mapper1.createMapping();
        // Get the mechanical model from mapper2 to fill the constraints vector
        MechanicalState2* mmodel2 = selfCollision ? mmodel1 : mapper2.createMapping();
        m_constraint = sofa::core::objectmodel::New<constraintset::UnilateralInteractionConstraint<Vec3Types> >(mmodel1, mmodel2);
        m_constraint->setName( getName() );
        m_constraint->setCustomTolerance( tol.getValue() );
    }

    int size = contacts.size();
    m_constraint->clear(size);
    if (selfCollision)
        mapper1.resize(2*size);
    else
    {
        mapper1.resize(size);
        mapper2.resize(size);
    }
    int i = 0;
    const double d0 = intersectionMethod->getContactDistance() + model1->getProximity() + model2->getProximity(); // - 0.001;

    //std::cout<<" d0 = "<<d0<<std::endl;

    mappedContacts.resize(contacts.size());
    for (std::vector<DetectionOutput*>::const_iterator it = contacts.begin(); it!=contacts.end(); it++, i++)
    {
        DetectionOutput* o = *it;
        //std::cout<<" collisionElements :"<<o->elem.first<<" - "<<o->elem.second<<std::endl;
        CollisionElement1 elem1(o->elem.first);
        CollisionElement2 elem2(o->elem.second);
        int index1 = elem1.getIndex();
        int index2 = elem2.getIndex();
        //std::cout<<" indices :"<<index1<<" - "<<index2<<std::endl;

        typename DataTypes1::Real r1 = 0.;
        typename DataTypes2::Real r2 = 0.;
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


    //std::cerr<<" end activateMappers call"<<std::endl;

}

template < class TCollisionModel1, class TCollisionModel2 >
void FrictionContact<TCollisionModel1,TCollisionModel2>::createResponse(core::objectmodel::BaseContext* group)
{

    activateMappers();
    const double mu_ = this->mu.getValue();
    // Checks if friction is considered
    if (mu_ < 0.0 || mu_ > 1.0)
        serr << sendl << "Error: mu has to take values between 0.0 and 1.0" << sendl;

    int i=0;
    if (m_constraint)
    {
        for (std::vector<DetectionOutput*>::const_iterator it = contacts.begin(); it!=contacts.end(); it++, i++)
        {
            DetectionOutput* o = *it;
            int index1 = mappedContacts[i].first.first;
            int index2 = mappedContacts[i].first.second;
            double distance = mappedContacts[i].second;

            // Polynome de Cantor de NxN sur N bijectif f(x,y)=((x+y)^2+3x+y)/2
            long index = cantorPolynomia(o->id /*cantorPolynomia(index1, index2)*/,id);

            // Add contact in unilateral constraint
            m_constraint->addContact(mu_, o->normal, distance, index1, index2, index, o->id);
        }

        if (parent!=NULL)
        {
            parent->removeObject(this);
            parent->removeObject(m_constraint);
        }

        parent = group;
        if (parent!=NULL)
        {
            //sout << "Attaching contact response to "<<parent->getName()<<sendl;
            parent->addObject(this);
            parent->addObject(m_constraint);
        }
    }
}

template < class TCollisionModel1, class TCollisionModel2 >
void FrictionContact<TCollisionModel1,TCollisionModel2>::removeResponse()
{
    if (m_constraint)
    {
        mapper1.resize(0);
        mapper2.resize(0);
        if (parent!=NULL)
        {
            //sout << "Removing contact response from "<<parent->getName()<<sendl;
            parent->removeObject(this);
            parent->removeObject(m_constraint);
        }
        parent = NULL;
    }
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
