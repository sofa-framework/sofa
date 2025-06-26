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
#include <sofa/component/collision/response/contact/StickContactConstraint.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/response/mapper/BarycentricContactMapper.h>
#include <sofa/component/collision/response/mapper/IdentityContactMapper.h>
#include <sofa/component/collision/response/mapper/RigidContactMapper.inl>
#include <sofa/simulation/Node.h>

namespace sofa::component::collision::response::contact
{

template < class TCollisionModel1, class TCollisionModel2 >
StickContactConstraint<TCollisionModel1,TCollisionModel2>::StickContactConstraint()
    : StickContactConstraint(nullptr, nullptr, nullptr)
{
    //#TODO -> Check impact of calling setCollisionModel
}

template < class TCollisionModel1, class TCollisionModel2 >
StickContactConstraint<TCollisionModel1,TCollisionModel2>::StickContactConstraint(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
    : model1(model1)
    , model2(model2)
    , intersectionMethod(intersectionMethod)
    , m_constraint(nullptr)
    , parent(nullptr)
    , d_keepAlive(initData(&d_keepAlive, true, "keepAlive", "set to true to keep this contact alive even after collisions are no longer detected"))
{
    mapper1.setCollisionModel(model1);
    mapper2.setCollisionModel(model2);
    this->f_printLog.setValue(true);
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
        if (parent != nullptr)
            parent->removeObject(m_constraint);

        parent = nullptr;
        //delete m_constraint;
        intrusive_ptr_add_ref(m_constraint.get()); // HACK: keep created constraints to avoid crash
        m_constraint.reset();

        mapper1.cleanup();

        mapper2.cleanup();
    }
    contacts.clear();
    mappedContacts.clear();
}


template < class TCollisionModel1, class TCollisionModel2 >
void StickContactConstraint<TCollisionModel1,TCollisionModel2>::doSetDetectionOutputs(OutputVector* o)
{
    this->f_printLog.setValue(true);
    msg_info() << "setDetectionOutputs(" << (o == nullptr ? -1 : (int)static_cast<TOutputVector*>(o)->size()) << ")";
    contacts.clear();
    if (!o) return;
    TOutputVector& outputs = *static_cast<TOutputVector*>(o);
    // We need to remove duplicate contacts
    constexpr double minDist2 = 0.00000001f;


    int SIZE = outputs.size();
    msg_info() << SIZE << " contacts" ;

    contacts.reserve(SIZE);

    int OUTSIZE = 0;

    // the following procedure cancels the duplicated detection outputs
    for (int cpt=0; cpt<SIZE; cpt++)
    {
        sofa::core::collision::DetectionOutput* detectionOutput = &outputs[cpt];

        bool found = false;
        for (int i=0; i<cpt && !found; i++)
        {
            const sofa::core::collision::DetectionOutput* p = &outputs[i];
            if ((detectionOutput->point[0]-p->point[0]).norm2()+(detectionOutput->point[1]-p->point[1]).norm2() < minDist2)
                found = true;
        }

        if (found) continue;
        contacts.push_back(detectionOutput);
        ++OUTSIZE;
    }

    // DUPLICATED CONTACTS FOUND
    msg_info_when(OUTSIZE<SIZE) << "Removed " << (SIZE-OUTSIZE) <<" / " << SIZE << " collision points." ;

}

template < class TCollisionModel1, class TCollisionModel2 >
void StickContactConstraint<TCollisionModel1,TCollisionModel2>::activateMappers()
{
    if (!m_constraint)
    {
        msg_info() << "Creating StickContactConstraint bilateral constraints";
        MechanicalState1* mstate1 = mapper1.createMapping(mapper::GenerateStringID::generate().c_str());
        MechanicalState2* mstate2 = mapper2.createMapping(mapper::GenerateStringID::generate().c_str());
        m_constraint = sofa::core::objectmodel::New<constraint::lagrangian::model::BilateralLagrangianConstraint<defaulttype::Vec3Types> >(mstate1, mstate2);
        m_constraint->setName( getName() );
    }

    msg_info() << "activateMappers(" << contacts.size() << ")";

    int size = contacts.size();
    m_constraint->clear(size);
    mapper1.resize(size);
    mapper2.resize(size);

    int i = 0;
    const double d0 = intersectionMethod->getContactDistance() + model1->getProximity() + model2->getProximity(); // - 0.001;

    mappedContacts.resize(contacts.size());
    for (auto it = contacts.begin(); it!=contacts.end(); it++, i++)
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
        index2 = mapper2.addPointB(o->point[1], index2, r2);
        const double distance = d0 + r1 + r2;

        mappedContacts[i].first.first = index1;
        mappedContacts[i].first.second = index2;
        mappedContacts[i].second = distance;
    }

    // Update mappings
    mapper1.update();
    mapper1.updateXfree();
    mapper2.update();
    mapper2.updateXfree();

    msg_info() << contacts.size() << " StickContactConstraint created";
    msg_info() << "mstate1 size = " << m_constraint->getMState1()->getSize() << " x = " << m_constraint->getMState1()->getSize() << " xfree = " << m_constraint->getMState1()->read(core::vec_id::read_access::freePosition)->getValue().size();
    msg_info() << "mstate2 size = " << m_constraint->getMState2()->getSize() << " x = " << m_constraint->getMState2()->getSize() << " xfree = " << m_constraint->getMState2()->read(core::vec_id::read_access::freePosition)->getValue().size();

}

template < class TCollisionModel1, class TCollisionModel2 >
void StickContactConstraint<TCollisionModel1,TCollisionModel2>::doCreateResponse(core::objectmodel::BaseContext* group)
{
    msg_info() << "->createResponse(" << group->getName() << ")";
    if (!contacts.empty() || !keepAlive())
    {
        activateMappers();
        int i = 0;
        for (auto it = contacts.begin(); it!=contacts.end(); it++, i++)
        {
            const sofa::core::collision::DetectionOutput* o = *it;
            const int index1 = mappedContacts[i].first.first;
            const int index2 = mappedContacts[i].first.second;
            const double distance = mappedContacts[i].second;

            // Polynome de Cantor de NxN sur N bijectif f(x,y)=((x+y)^2+3x+y)/2
            //long index = cantorPolynomia(o->id /*cantorPolynomia(index1, index2)*/,id);

            // Add contact in unilateral constraint
            m_constraint->addContact(o->normal, o->point[0], o->point[1], distance, index1, index2, o->point[0], o->point[1], i, o->id);
        }
    }

    if (m_constraint!=nullptr)
    {
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

template < class TCollisionModel1, class TCollisionModel2 >
void StickContactConstraint<TCollisionModel1,TCollisionModel2>::doRemoveResponse()
{
    msg_info() << "->removeResponse()";
    if (m_constraint)
    {
        if (parent!=nullptr)
        {
            parent->removeObject(this);
            parent->removeObject(m_constraint);
        }
        parent = nullptr;
    }
}

} //namespace sofa::component::collision::response::contact
