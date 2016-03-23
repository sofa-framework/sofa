/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_COLLISION_BARYCENTRICLAGRANGIANMULTIPLIERCONTACT_INL
#define SOFA_COMPONENT_COLLISION_BARYCENTRICLAGRANGIANMULTIPLIERCONTACT_INL

#include <sofa/component/collision/BarycentricLagrangianMultiplierContact.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace core::collision;

template < class TCollisionModel1, class TCollisionModel2 >
BarycentricLagrangianMultiplierContact<TCollisionModel1,TCollisionModel2>::BarycentricLagrangianMultiplierContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
    : model1(model1), model2(model2), intersectionMethod(intersectionMethod), ff(NULL), parent(NULL)
{
    mapper1.setCollisionModel(model1);
    mapper2.setCollisionModel(model2);
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
void BarycentricLagrangianMultiplierContact<TCollisionModel1,TCollisionModel2>::setDetectionOutputs(OutputVector* o)
{
    TOutputVector& outputs = *static_cast<TOutputVector*>(o);
    // We need to remove duplicate contacts
    const double minDist2 = 0.01f;
    std::vector<DetectionOutput*> contacts;
    contacts.reserve(outputs.size());
    for (std::vector<DetectionOutput>::iterator it = outputs.begin(); it!=outputs.end(); it++)
    {
        DetectionOutput* o = &*it;
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
    if (contacts.size()<outputs.size() && this->f_printLog.getValue())
    {
        sout << "Removed " << (outputs.size()-contacts.size()) <<" / " << outputs.size() << " collision points." << sendl;
    }
    if (ff==NULL)
    {
        MechanicalState1* mstate1 = mapper1.createMapping();
        MechanicalState2* mstate2 = mapper2.createMapping();
        ff = new constraintset::LagrangianMultiplierContactConstraint<Vec3Types>(mstate1,mstate2);
        setInteractionTags(mstate1, mstate2);
    }

    int size = contacts.size();


    ff->clear(size);
    mapper1.resize(size);
    mapper2.resize(size);
    int i = 0;
    for (std::vector<DetectionOutput*>::iterator it = contacts.begin(); it!=contacts.end(); it++, i++)
    {
        DetectionOutput* o = *it;
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
        double distance = intersectionMethod->getContactDistance() + r1 + r2;
        if (!model1->isSimulated() || !model2->isSimulated()) // create stiffer springs for non-animated models as only half of the force is really applied
            ff->addContact(index1, index2, o->normal, distance, 300, 0.00f, 0.00f); /// \todo compute stiffness and damping
        else
            ff->addContact(index1, index2, o->normal, distance, 150, 0.00f, 0.00f); /// \todo compute stiffness and damping
    }
    // Update mappings
    mapper1.update();
    mapper2.update();
}

template < class TCollisionModel1, class TCollisionModel2 >
void BarycentricLagrangianMultiplierContact<TCollisionModel1,TCollisionModel2>::createResponse(core::objectmodel::BaseContext* group)
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
void BarycentricLagrangianMultiplierContact<TCollisionModel1,TCollisionModel2>::draw(const core::visual::VisualParams* vparams)
{
//	if (ff!=NULL)
//		ff->draw(vparams);
}

template < class TCollisionModel1, class TCollisionModel2 >
void BarycentricLagrangianMultiplierContact<TCollisionModel1,TCollisionModel2>::setInteractionTags(MechanicalState1* mstate1, MechanicalState2* mstate2)
{
    TagSet tagsm1 = mstate1->getTags();
    TagSet tagsm2 = mstate2->getTags();
    TagSet::iterator it;
    for(it=tagsm1.begin(); it != tagsm1.end(); it++)
        ff->addTag(*it);
    for(it=tagsm2.begin(); it!=tagsm2.end(); it++)
        ff->addTag(*it);
}


} // namespace collision

} // namespace component

} // namespace sofa

#endif
