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
#ifndef SOFA_COMPONENT_COLLISION_STICKCONTACTCONSTRAINT_INL
#define SOFA_COMPONENT_COLLISION_STICKCONTACTCONSTRAINT_INL

#include <sofa/component/collision/StickContactConstraint.h>
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
StickContactConstraint<TCollisionModel1,TCollisionModel2>::StickContactConstraint(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
    : model1(model1)
    , model2(model2)
    , intersectionMethod(intersectionMethod)
    , m_constraint(NULL)
    , parent(NULL)
    , f_keepAlive(initData(&f_keepAlive, true, "keepAlive", "set to true to keep this contact alive even after collisions are no longer detected"))
{
    selfCollision = ((core::CollisionModel*)model1 == (core::CollisionModel*)model2);
    mapper1.setCollisionModel(model1);
    if (!selfCollision) mapper2.setCollisionModel(model2);
    //contacts.clear();
    //mappedContacts.clear();

}

template < class TCollisionModel1, class TCollisionModel2 >
StickContactConstraint<TCollisionModel1,TCollisionModel2>::~StickContactConstraint()
{
}

template < class TCollisionModel1, class TCollisionModel2 >
void StickContactConstraint<TCollisionModel1,TCollisionModel2>::cleanup()
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

    //contacts.clear();
    //mappedContacts.clear();
}


template < class TCollisionModel1, class TCollisionModel2 >
void StickContactConstraint<TCollisionModel1,TCollisionModel2>::setDetectionOutputs(OutputVector* o)
{
    if (!o) return;
    TOutputVector& outputs = *static_cast<TOutputVector*>(o);
    // We need to remove duplicate contacts
    const double minDist2 = 0.00000001f;

    if (!m_constraint)
    {
        serr << "Creating StickContactConstraint bilateral constraints"<<sendl;
        MechanicalState1* mstate1 = mapper1.createMapping();
        MechanicalState2* mstate2 = mapper2.createMapping();
        m_constraint = sofa::core::objectmodel::New<constraintset::BilateralInteractionConstraint<Vec3Types> >(mstate1, mstate2);
        m_constraint->setName( getName() );
    }

    int SIZE = outputs.size();
    m_constraint->clear(SIZE);
    mapper1.resize(SIZE);
    mapper2.resize(SIZE);
    /*
        contacts.clear();

    	contacts.reserve(outputs.size());
    */
    int OUTSIZE = 0;
    const double d0 = intersectionMethod->getContactDistance() + model1->getProximity() + model2->getProximity(); // - 0.001;

    // the following procedure cancels the duplicated detection outputs
    for (int cpt=0; cpt<SIZE; cpt++)
    {
        DetectionOutput* o = &outputs[cpt];

        bool found = false;
        for (int i=0; i<cpt && !found; i++)
        {
            DetectionOutput* p = &outputs[i];
            if ((o->point[0]-p->point[0]).norm2()+(o->point[1]-p->point[1]).norm2() < minDist2)
                found = true;
        }

        if (found) continue;
        CollisionElement1 elem1(o->elem.first);
        CollisionElement2 elem2(o->elem.second);
        int index1 = elem1.getIndex();
        int index2 = elem2.getIndex();

        typename DataTypes1::Real r1 = 0.0;
        typename DataTypes2::Real r2 = 0.0;
        // Create mapping for first point
        index1 = mapper1.addPoint(o->point[0], index1, r1);
        // Create mapping for second point
        index2 = mapper2.addPoint(o->point[1], index2, r2);

        double distance = d0 + r1 + r2;

        m_constraint->addContact(o->normal, o->point[0], o->point[1], distance, index1, index2, o->point[0], o->point[1], OUTSIZE, 0);
        ++OUTSIZE;
    }

    if (OUTSIZE<SIZE)
    {
        // DUPLICATED CONTACTS FOUND
        sout << "Removed " << (SIZE-OUTSIZE) <<" / " << SIZE << " collision points." << sendl;
    }
    // Update mappings
    mapper1.update();
    mapper2.update();
    sout << OUTSIZE << "StickContactConstraint created"<<sendl;
}

#if 0
template < class TCollisionModel1, class TCollisionModel2 >
void StickContactConstraint<TCollisionModel1,TCollisionModel2>::activateMappers()
{
    if (!m_constraint)
    {
        // Get the mechanical model from mapper1 to fill the constraint vector
        MechanicalState1* mmodel1 = mapper1.createMapping();
        // Get the mechanical model from mapper2 to fill the constraints vector
        MechanicalState2* mmodel2 = selfCollision ? mmodel1 : mapper2.createMapping();
        m_constraint = sofa::core::objectmodel::New<constraintset::BilateralInteractionConstraint<Vec3Types> >(mmodel1, mmodel2);
        m_constraint->setName( getName() );
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
#endif

template < class TCollisionModel1, class TCollisionModel2 >
void StickContactConstraint<TCollisionModel1,TCollisionModel2>::createResponse(core::objectmodel::BaseContext* group)
{

    if (m_constraint!=NULL)
    {
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
#if 0
    activateMappers();

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

            // Add contact in bilateral constraint
            m_constraint->addContact(o->normal, distance, index1, index2, index, o->id);
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
#endif
}

template < class TCollisionModel1, class TCollisionModel2 >
void StickContactConstraint<TCollisionModel1,TCollisionModel2>::removeResponse()
{
    if (m_constraint)
    {
        //mapper1.resize(0);
        //mapper2.resize(0);
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
