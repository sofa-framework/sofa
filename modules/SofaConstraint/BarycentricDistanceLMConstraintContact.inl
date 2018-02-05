/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_COLLISION_BarycentricDistanceLMConstraintContact_INL
#define SOFA_COMPONENT_COLLISION_BarycentricDistanceLMConstraintContact_INL

#include <SofaConstraint/BarycentricDistanceLMConstraintContact.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace collision
{



template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
BarycentricDistanceLMConstraintContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::BarycentricDistanceLMConstraintContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
    : model1(model1), model2(model2), intersectionMethod(intersectionMethod), ff(NULL), parent(NULL)
{
    mapper1.setCollisionModel(model1);
    mapper2.setCollisionModel(model2);
}

template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
BarycentricDistanceLMConstraintContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::~BarycentricDistanceLMConstraintContact()
{
}





template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
void BarycentricDistanceLMConstraintContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::cleanup()
{
    if (ff!=NULL)
    {
        ff->cleanup();
        if (parent!=NULL) parent->removeObject(ff);
        parent = NULL;
        ff.reset();
        mapper1.cleanup();
        mapper2.cleanup();
    }
}

template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
void BarycentricDistanceLMConstraintContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::setDetectionOutputs(OutputVector* o)
{
    TOutputVector& outputs = *static_cast<TOutputVector*>(o);
    if (ff==NULL)
    {

        MechanicalState1* mstate1 = mapper1.createMapping(GenerateStirngID::generate().c_str());
        MechanicalState2* mstate2 = mapper2.createMapping(GenerateStirngID::generate().c_str());
        ff = sofa::core::objectmodel::New<ResponseType>(mstate1,mstate2);
        ff->setName( getName() );
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
        core::collision::DetectionOutput* o = &outputs[i];
        // find this contact in contactIndex, possibly creating a new entry initialized by 0
        int& index = contactIndex[o->id];
        if (index < 0) // duplicate contact
        {
            int i2 = -1-index;
            core::collision::DetectionOutput* o2 = &outputs[i2];
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
                msg_info() << " new contact "<<o->id ;
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
            msg_info() << " removed contact "<<it->first;
            ContactIndexMap::iterator oldit = it;
            ++it;
            contactIndex.erase(oldit);
        }
        else
        {
            index = newIndex[-1-index]; // write the final contact index
            ++it;
        }
    }
    msg_info() << " "<<insize<<" input contacts, "<<size<<" contacts used for response ("<<nbnew<<" new)." ;

    ff->clear();
    mapper1.resize(size);
    mapper2.resize(size);
//    const double d0 = intersectionMethod->getContactDistance() + model1->getProximity() + model2->getProximity(); // - 0.001;
    for (int i=0; i<insize; i++)
    {
        int index = oldIndex[i];
        if (index < 0) continue; // this contact is ignored
        core::collision::DetectionOutput* o = &outputs[i];
        CollisionElement1 elem1(o->elem.first);
        CollisionElement2 elem2(o->elem.second);
        int index1 = elem1.getIndex();
        int index2 = elem2.getIndex();
        typename DataTypes1::Real r1 = 0.0;
        typename DataTypes2::Real r2 = 0.0;
        // Create mapping for first point
        index1 = mapper1.addPointB(o->point[0], index1, r1
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                , o->baryCoords[0]
#endif
                                  );
        // Create mapping for second point
        index2 = mapper2.addPointB(o->point[1], index2, r2
#ifdef DETECTIONOUTPUT_BARYCENTRICINFO
                , o->baryCoords[1]
#endif
                                  );

        ff->addContact(index1,index2);
        ff->contactFriction.setValue( elem1.getCollisionModel()->getContactFriction(0) * elem2.getCollisionModel()->getContactFriction(0) );
        // if all the elements belong to the same CollisionModels then ff->contactFriction should be done only once...
    }
    // Update mappings
    mapper1.update();
    mapper2.update();
}

template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
void BarycentricDistanceLMConstraintContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::createResponse(core::objectmodel::BaseContext* group)
{
    if (ff!=NULL)
    {
        if (parent!=NULL)
        {
            parent->removeObject(this);
            parent->removeObject(ff);
        }
        parent = group;
        if (parent!=NULL)
        {
            //msg_error() << "Attaching contact response to "<<parent->getName() ;
            parent->addObject(this);
            parent->addObject(ff);
        }
    }
}

template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
void BarycentricDistanceLMConstraintContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::removeResponse()
{
    if (ff!=NULL)
    {
        if (parent!=NULL)
        {
            //msg_error()<< "Removing contact response from "<<parent->getName();
            parent->removeObject(this);
            parent->removeObject(ff);
        }
        parent = NULL;
    }
}

template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
void BarycentricDistanceLMConstraintContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::draw(const core::visual::VisualParams* )
{
    //	if (ff!=NULL)
    //		ff->draw(vparams);
}

template < class TCollisionModel1, class TCollisionModel2, class ResponseDataTypes >
void BarycentricDistanceLMConstraintContact<TCollisionModel1,TCollisionModel2,ResponseDataTypes>::setInteractionTags(MechanicalState1* mstate1, MechanicalState2* mstate2)
{
    sofa::core::objectmodel::TagSet tagsm1 = mstate1->getTags();
    sofa::core::objectmodel::TagSet tagsm2 = mstate2->getTags();
    sofa::core::objectmodel::TagSet::iterator it;
    for(it=tagsm1.begin(); it != tagsm1.end(); ++it)
        ff->addTag(*it);
    for(it=tagsm2.begin(); it!=tagsm2.end(); ++it)
        ff->addTag(*it);
}


} // namespace collision

} // namespace component

} // namespace sofa

#endif
