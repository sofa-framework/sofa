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

#include <sofa/component/collision/response/contact/BarycentricStickContact.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa::component::collision::response::contact
{

template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
BarycentricStickContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::BarycentricStickContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
    : model1(model1), model2(model2), intersectionMethod(intersectionMethod), ff(nullptr), parent(nullptr)
    , d_keepAlive(initData(&d_keepAlive, true, "keepAlive", "set to true to keep this contact alive even after collisions are no longer detected"))
{
    mapper1.setCollisionModel(model1);
    mapper2.setCollisionModel(model2);
}

template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
BarycentricStickContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::~BarycentricStickContact()
{
}


template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
void BarycentricStickContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::cleanup()
{
    if (ff!=nullptr)
    {
        ff->cleanup();
        if (parent!=nullptr) parent->removeObject(ff);
        parent = nullptr;
        ff.reset();
        mapper1.cleanup();
        mapper2.cleanup();
    }
}

template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
void BarycentricStickContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::doSetDetectionOutputs(OutputVector* o)
{
    if (o==nullptr) return;
    TOutputVector& outputs = *static_cast<TOutputVector*>(o);

    if (ff==nullptr)
    {
        msg_info() << "Creating BarycentricStickContact springs" ;
        MechanicalState1* mstate1 = mapper1.createMapping(response::mapper::GenerateStringID::generate().c_str());
        MechanicalState2* mstate2 = mapper2.createMapping(response::mapper::GenerateStringID::generate().c_str());
        ff = sofa::core::objectmodel::New<ResponseForceField>(mstate1,mstate2);
        ff->setName( getName());
        setInteractionTags(mstate1, mstate2);
        ff->init();
    }

    int insize = outputs.size();

    // old index for each contact
    // >0 indicate preexisting contact
    // 0  indicate new contact
    // -1 indicate ignored duplicate contact
    std::vector<int> oldIndex(insize);

    int nbnew = 0;

    for (int i=0; i<insize; i++)
    {
        sofa::core::collision::DetectionOutput* o = &outputs[i];
        // find this contact in contactIndex, possibly creating a new entry initialized by 0
        int& index = contactIndex[o->id];
        if (index < 0) // duplicate contact
        {
            int i2 = -1-index;
            const sofa::core::collision::DetectionOutput* o2 = &outputs[i2];
            if (o2->value <= o->value)
            {
                // current contact is ignored
                oldIndex[i] = -1;
                continue;
            }
            else
            {
                // previous contact is replaced
                oldIndex[i] = oldIndex[i2];
                oldIndex[i2] = -1;
            }
        }
        else
        {
            oldIndex[i] = index;
            if (!index)
            {
                ++nbnew;
                msg_info() << "BarycentricStickContact: New contact "<<o->id ;
            }
        }
        index = -1-i; // save this index as a negative value in contactIndex map.
    }

    // compute new index of each contact
    std::vector<int> newIndex(insize);
    // number of final contacts used in the response
    int size = 0;
    for (int i=0; i<insize; i++)
    {
        if (oldIndex[i] >= 0)
        {
            ++size;
            newIndex[i] = size;
        }
    }

    // update contactMap
    for (ContactIndexMap::iterator it = contactIndex.begin(), itend = contactIndex.end(); it != itend; )
    {
        int& index = it->second;
        if (index >= 0)
        {
            msg_info() << "BarycentricStickContact: Removed contact "<<it->first;
            const ContactIndexMap::iterator oldit = it;
            ++it;
            contactIndex.erase(oldit);
        }
        else
        {
            index = newIndex[-1-index]; // write the final contact index
            ++it;
        }
    }
    msg_info() << "BarycentricStickContact: "<<insize<<" input contacts, "<<size<<" contacts used for response ("<<nbnew<<" new)." ;

    ff->clear(size);
    mapper1.resize(size);
    mapper2.resize(size);
    for (int i=0; i<insize; i++)
    {
        const int index = oldIndex[i];
        if (index < 0) continue; // this contact is ignored
        sofa::core::collision::DetectionOutput* o = &outputs[i];
        CollisionElement1 elem1(o->elem.first);
        CollisionElement2 elem2(o->elem.second);
        int index1 = elem1.getIndex();
        int index2 = elem2.getIndex();

        typename DataTypes1::Real r1 = 0.0;
        typename DataTypes2::Real r2 = 0.0;
        // Create mapping for first point
        index1 = mapper1.addPointB(o->point[0], index1, r1);
        // Create mapping for second point
        index2 = mapper2.addPointB(o->point[1], index2, r2);

        const double stiffness = (elem1.getContactStiffness() + elem2.getContactStiffness());
        ff->d_stiffness.setValue(stiffness);

        const double mu_v = (elem1.getContactFriction() + elem2.getContactFriction());

        ff->addSpring(index1, index2, stiffness, mu_v/* *distance */, o->point[1]-o->point[0]);
    }
    // Update mappings
    mapper1.update();
    mapper2.update();

    msg_info() << size << "BarycentricStickContact springs created";
}

template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
void BarycentricStickContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::doCreateResponse(core::objectmodel::BaseContext* group)
{
    if (ff!=nullptr)
    {
        if (parent!=nullptr)
        {
            parent->removeObject(this);
            parent->removeObject(ff);
        }
        parent = group;
        if (parent!=nullptr)
        {
            parent->addObject(this);
            parent->addObject(ff);
        }
    }
}

template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
void BarycentricStickContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::doRemoveResponse()
{
    if (ff!=nullptr)
    {
        if (parent!=nullptr)
        {
            parent->removeObject(this);
            parent->removeObject(ff);
        }
        parent = nullptr;
    }
}

template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
void BarycentricStickContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::draw(const core::visual::VisualParams* )
{
}

template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
void BarycentricStickContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::setInteractionTags(MechanicalState1* mstate1, MechanicalState2* mstate2)
{
    core::objectmodel::TagSet tagsm1 = mstate1->getTags();
    core::objectmodel::TagSet tagsm2 = mstate2->getTags();
    core::objectmodel::TagSet::iterator it;
    for(it=tagsm1.begin(); it != tagsm1.end(); it++)
        ff->addTag(*it);
    for(it=tagsm2.begin(); it!=tagsm2.end(); it++)
        ff->addTag(*it);
}

} // namespace sofa::component::collision::response::contact
