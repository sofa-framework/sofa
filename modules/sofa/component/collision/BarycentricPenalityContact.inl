/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_COLLISION_BARYCENTRICPENALITYCONTACT_INL
#define SOFA_COMPONENT_COLLISION_BARYCENTRICPENALITYCONTACT_INL

#include <sofa/component/collision/BarycentricPenalityContact.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace core::componentmodel::collision;
using simulation::tree::GNode;

template < class TCollisionModel1, class TCollisionModel2 >
BarycentricPenalityContact<TCollisionModel1,TCollisionModel2>::BarycentricPenalityContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
    : model1(model1), model2(model2), intersectionMethod(intersectionMethod), mapper1(model1), mapper2(model2), ff(NULL), parent(NULL)
{
}

template < class TCollisionModel1, class TCollisionModel2 >
BarycentricPenalityContact<TCollisionModel1,TCollisionModel2>::~BarycentricPenalityContact()
{
    if (ff!=NULL)
    {
        if (parent!=NULL) parent->removeObject(ff);
        delete ff;
    }
}

template < class TCollisionModel1, class TCollisionModel2 >
void BarycentricPenalityContact<TCollisionModel1,TCollisionModel2>::setDetectionOutputs(std::vector<DetectionOutput>& outputs)
{
    // We need to remove duplicate contacts
    /*
    const double minDist2 = 0.000001f;
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
    if (contacts.size()<outputs.size())
    {
    	//std::cout << "Removed " << (outputs.size()-contacts.size()) <<" / " << outputs.size() << " collision points." << std::endl;
    }
    */

    if (ff==NULL)
    {
        MechanicalState1* mstate1 = mapper1.createMapping();
        MechanicalState2* mstate2 = mapper2.createMapping();
        ff = new forcefield::PenalityContactForceField<Vec3Types>(mstate1,mstate2);
    }

    //int size = contacts.size();
    int size = outputs.size();
    ff->clear(size);
    mapper1.resize(size);
    mapper2.resize(size);
    //int i = 0;
    const double d0 = intersectionMethod->getContactDistance() + model1->getProximity() + model2->getProximity();
    //for (std::vector<DetectionOutput*>::const_iterator it = contacts.begin(); it!=contacts.end(); it++)
    //{
    //	DetectionOutput* o = *it;
    for (std::vector<DetectionOutput>::iterator it = outputs.begin(); it!=outputs.end(); it++)
    {
        DetectionOutput* o = &*it;
        CollisionElement1 elem1(o->elem.first);
        CollisionElement2 elem2(o->elem.second);
        int index1 = elem1.getIndex();
        int index2 = elem2.getIndex();
        // Create mapping for first point
        index1 = mapper1.addPoint(o->point[0], index1);
        // Create mapping for second point
        index2 = mapper2.addPoint(o->point[1], index2);
        double distance = d0 + mapper1.radius(elem1) + mapper2.radius(elem2);
        double stiffness = (elem1.getContactStiffness() * elem2.getContactStiffness())/distance;
        double mu_v = (elem1.getContactFriction() + elem2.getContactFriction());
        ff->addContact(index1, index2, o->normal, distance, stiffness, mu_v*distance, mu_v);
        //if (model1->isStatic() || model2->isStatic()) // create stiffer springs for static models as only half of the force is really applied
        //	ff->addContact(index1, index2, o->normal, distance, 300, 0.00f, 0.00f); /// \todo compute stiffness and damping
        //else
        //	ff->addContact(index1, index2, o->normal, distance, 250, 0.00f, 0.00f); /// \todo compute stiffness and damping
    }
    // Update mappings
    mapper1.update();
    mapper2.update();
}

template < class TCollisionModel1, class TCollisionModel2 >
void BarycentricPenalityContact<TCollisionModel1,TCollisionModel2>::createResponse(core::objectmodel::BaseContext* group)
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
void BarycentricPenalityContact<TCollisionModel1,TCollisionModel2>::removeResponse()
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
void BarycentricPenalityContact<TCollisionModel1,TCollisionModel2>::draw()
{
//	if (dynamic_cast<core::VisualModel*>(ff)!=NULL)
//		dynamic_cast<core::VisualModel*>(ff)->draw();
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
