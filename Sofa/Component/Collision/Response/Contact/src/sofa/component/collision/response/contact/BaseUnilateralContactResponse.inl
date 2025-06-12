/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/collision/response/contact/BaseUnilateralContactResponse.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/response/contact/CollisionResponse.h>
#include <sofa/component/collision/response/mapper/BarycentricContactMapper.h>
#include <sofa/component/collision/response/mapper/IdentityContactMapper.h>
#include <sofa/component/collision/response/mapper/RigidContactMapper.inl>
#include <sofa/simulation/Node.h>
#include <iostream>

namespace sofa::component::collision::response::contact
{

template <class TCollisionModel1, class TCollisionModel2, class ConstraintParameters, class ResponseDataTypes >
BaseUnilateralContactResponse<TCollisionModel1, TCollisionModel2, ConstraintParameters, ResponseDataTypes>::BaseUnilateralContactResponse()
    : BaseUnilateralContactResponse(nullptr, nullptr, nullptr)
{
}


template <class TCollisionModel1, class TCollisionModel2, class ConstraintParameters, class ResponseDataTypes >
BaseUnilateralContactResponse<TCollisionModel1, TCollisionModel2, ConstraintParameters, ResponseDataTypes>::BaseUnilateralContactResponse(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
    : model1(model1)
    , model2(model2)
    , intersectionMethod(intersectionMethod)
    , m_constraint(nullptr)
    , parent(nullptr)
    , d_tol (initData(&d_tol, 0.0, "tol", "tolerance for the constraints resolution (0 for default tolerance)"))
{
    selfCollision = ((core::CollisionModel*)model1 == (core::CollisionModel*)model2);
    mapper1.setCollisionModel(model1);
    if (!selfCollision) mapper2.setCollisionModel(model2);
    contacts.clear();
    mappedContacts.clear();
}

template <class TCollisionModel1, class TCollisionModel2, class ConstraintParameters, class ResponseDataTypes >
BaseUnilateralContactResponse<TCollisionModel1, TCollisionModel2, ConstraintParameters, ResponseDataTypes>::~BaseUnilateralContactResponse()
{
}

template <class TCollisionModel1, class TCollisionModel2, class ConstraintParameters, class ResponseDataTypes >
void BaseUnilateralContactResponse<TCollisionModel1, TCollisionModel2, ConstraintParameters, ResponseDataTypes>::cleanup()
{
    if (m_constraint)
    {
        m_constraint->cleanup();

        if (parent != nullptr)
            parent->removeObject(m_constraint);

        parent = nullptr;
        m_constraint.reset();

        mapper1.cleanup();

        if (!selfCollision)
            mapper2.cleanup();
    }

    contacts.clear();
    mappedContacts.clear();
}


template <class TCollisionModel1, class TCollisionModel2, class ConstraintParameters, class ResponseDataTypes >
void BaseUnilateralContactResponse<TCollisionModel1, TCollisionModel2, ConstraintParameters, ResponseDataTypes>::doSetDetectionOutputs(OutputVector* o)
{
    TOutputVector& outputs = *static_cast<TOutputVector*>(o);
    // We need to remove duplicate contacts
    constexpr double minDist2 = 0.00000001f;

    contacts.clear();

    if (model1->getContactStiffness(0) == 0 || model2->getContactStiffness(0) == 0)
    {
        msg_error() << "Disabled BaseUnilateralContactResponse with " << (outputs.size()) << " collision points.";
        return;
    }

    contacts.reserve(outputs.size());

    const int SIZE = outputs.size();

    // the following procedure cancels the duplicated detection outputs
    for (int cpt=0; cpt<SIZE; cpt++)
    {
        sofa::core::collision::DetectionOutput* detectionOutput = &outputs[cpt];

        bool found = false;
        for (unsigned int i=0; i<contacts.size() && !found; i++)
        {
            const sofa::core::collision::DetectionOutput* p = contacts[i];
            if ((detectionOutput->point[0]-p->point[0]).norm2()+(detectionOutput->point[1]-p->point[1]).norm2() < minDist2)
                found = true;
        }

        if (!found)
            contacts.push_back(detectionOutput);
    }

    // DUPLICATED CONTACTS FOUND
    msg_info_when(contacts.size()<outputs.size()) << "Removed " << (outputs.size()-contacts.size()) <<" / " << outputs.size() << " collision points." << msgendl;
}


template <class TCollisionModel1, class TCollisionModel2, class ConstraintParameters, class ResponseDataTypes >
void BaseUnilateralContactResponse<TCollisionModel1, TCollisionModel2, ConstraintParameters, ResponseDataTypes>::activateMappers()
{
    if (!m_constraint)
    {
        // Get the mechanical model from mapper1 to fill the constraint vector
        MechanicalState1* mmodel1 = mapper1.createMapping(getName().c_str());
        // Get the mechanical model from mapper2 to fill the constraints vector
        MechanicalState2* mmodel2;
        if (selfCollision)
        {
            mmodel2 = mmodel1;
        }
        else
        {
            mmodel2 = mapper2.createMapping(getName().c_str());
        }
        setupConstraint(mmodel1,mmodel2);
        m_constraint->setName( getName() );
        setInteractionTags(mmodel1, mmodel2);
        m_constraint->setCustomTolerance(d_tol.getValue() );
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

    mappedContacts.resize(contacts.size());
    for (std::vector<sofa::core::collision::DetectionOutput*>::const_iterator it = contacts.begin(); it!=contacts.end(); it++, i++)
    {
        sofa::core::collision::DetectionOutput* o = *it;
        CollisionElement1 elem1(o->elem.first);
        CollisionElement2 elem2(o->elem.second);
        int index1 = elem1.getIndex();
        int index2 = elem2.getIndex();

        typename DataTypes1::Real r1 = 0.;
        typename DataTypes2::Real r2 = 0.;

        // Create mapping for first point
        index1 = mapper1.addPointB(o->point[0], index1, r1);
        // Create mapping for second point
        if (selfCollision)
        {
            index2 = mapper1.addPointB(o->point[1], index2, r2);
        }
        else
        {
            index2 = mapper2.addPointB(o->point[1], index2, r2);
        }
        const double distance = d0 + r1 + r2;

        mappedContacts[i].first.first = index1;
        mappedContacts[i].first.second = index2;
        mappedContacts[i].second = distance;
    }

    // Update mappings
    mapper1.update();
    mapper1.updateXfree();
    if (!selfCollision) mapper2.update();
    if (!selfCollision) mapper2.updateXfree();
}

template <class TCollisionModel1, class TCollisionModel2, class ConstraintParameters, class ResponseDataTypes >
void BaseUnilateralContactResponse<TCollisionModel1, TCollisionModel2, ConstraintParameters, ResponseDataTypes>::doCreateResponse(core::objectmodel::BaseContext* group)
{

    activateMappers();

    if (m_constraint)
    {
        int i=0;
        for (std::vector<sofa::core::collision::DetectionOutput*>::const_iterator it = contacts.begin(); it!=contacts.end(); it++, i++)
        {
            const sofa::core::collision::DetectionOutput* o = *it;
            const int index1 = mappedContacts[i].first.first;
            const int index2 = mappedContacts[i].first.second;
            const double distance = mappedContacts[i].second;

            // Polynome de Cantor de NxN sur N bijectif f(x,y)=((x+y)^2+3x+y)/2
            const long index = cantorPolynomia(o->id /*cantorPolynomia(index1, index2)*/,id);

            const ConstraintParameters params = getParameterFromDatas();

            // Add contact in unilateral constraint
            m_constraint->addContact(params, o->normal, distance, index1, index2, index, o->id);
        }

        if (parent!=nullptr)
        {
            parent->removeObject(this);
            parent->removeObject(m_constraint);
        }

        parent = group;
        if (parent!=nullptr)
        {
            parent->addObject(this);
            parent->addObject(m_constraint);
        }
    }
}

template <class TCollisionModel1, class TCollisionModel2, class ConstraintParameters, class ResponseDataTypes >
void BaseUnilateralContactResponse<TCollisionModel1, TCollisionModel2, ConstraintParameters, ResponseDataTypes>::doRemoveResponse()
{
    if (m_constraint)
    {
        mapper1.resize(0);
        mapper2.resize(0);
        if (parent!=nullptr)
        {
            parent->removeObject(this);
            parent->removeObject(m_constraint);
        }
        parent = nullptr;
    }
}

template <class TCollisionModel1, class TCollisionModel2, class ConstraintParameters, class ResponseDataTypes >
void BaseUnilateralContactResponse<TCollisionModel1, TCollisionModel2, ConstraintParameters, ResponseDataTypes>::setInteractionTags(MechanicalState1* mstate1, MechanicalState2* mstate2)
{
    sofa::core::objectmodel::TagSet tagsm1 = mstate1->getTags();
    sofa::core::objectmodel::TagSet tagsm2 = mstate2->getTags();
    sofa::core::objectmodel::TagSet::iterator it;
    for(it=tagsm1.begin(); it != tagsm1.end(); it++)
        m_constraint->addTag(*it);
    for(it=tagsm2.begin(); it!=tagsm2.end(); it++)
        m_constraint->addTag(*it);
}

} //namespace sofa::component::collision::response::contact
