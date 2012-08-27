#ifndef SOFA_COMPONENT_COLLISION_COMPLIANTCONTACT_INL
#define SOFA_COMPONENT_COLLISION_COMPLIANTCONTACT_INL

#include "CompliantContact.h"


#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/DefaultContactManager.h>
#include <sofa/component/collision/BarycentricContactMapper.h>
#include <sofa/component/collision/IdentityContactMapper.h>
#include <sofa/component/collision/RigidContactMapper.h>
#include <sofa/simulation/common/Node.h>
#include <iostream>

// TODO why do i need to do this to get correct linkage ?
#include <sofa/component/collision/FrictionContact.inl>

namespace sofa
{

namespace component
{

namespace collision
{

template < class TCollisionModel1, class TCollisionModel2 >
CompliantContact<TCollisionModel1,TCollisionModel2>::~CompliantContact() { }

template < class TCollisionModel1, class TCollisionModel2 >
void CompliantContact<TCollisionModel1,TCollisionModel2>::createResponse(core::objectmodel::BaseContext* group)
{

    // this->activateMappers();
    // const double mu_ = this->mu.getValue();
    // // Checks if friction is considered
    // if (mu_ < 0.0 || mu_ > 1.0)
    //   serr << sendl << "Error: mu has to take values between 0.0 and 1.0" << sendl;

    // int i=0;
    // if (this->m_constraint)
    //   {
    //     for (std::vector<DetectionOutput*>::const_iterator it = contacts.begin(); it!=contacts.end(); it++, i++)
    //       {
    // 	DetectionOutput* o = *it;
    // 	int index1 = mappedContacts[i].first.first;
    // 	int index2 = mappedContacts[i].first.second;
    // 	double distance = mappedContacts[i].second;

    // 	// Polynome de Cantor de NxN sur N bijectif f(x,y)=((x+y)^2+3x+y)/2
    // 	long index = cantorPolynomia(o->id /*cantorPolynomia(index1, index2)*/,id);

    // 	// Add contact in unilateral constraint
    // 	m_constraint->addContact(mu_, o->normal, distance, index1, index2, index, o->id);
    //       }

    //     if (parent!=NULL)
    //       {
    // 	parent->removeObject(this);
    // 	parent->removeObject(m_constraint);
    //       }

    //     parent = group;
    //     if (parent!=NULL)
    //       {
    // 	//sout << "Attaching contact response to "<<parent->getName()<<sendl;
    // 	parent->addObject(this);
    // 	parent->addObject(m_constraint);
    //       }
    //   }
}

template < class TCollisionModel1, class TCollisionModel2 >
void CompliantContact<TCollisionModel1,TCollisionModel2>::removeResponse()
{
    // if (m_constraint)
    //   {
    //     mapper1.resize(0);
    //     mapper2.resize(0);
    //     if (parent!=NULL)
    //       {
    // 	//sout << "Removing contact response from "<<parent->getName()<<sendl;
    // 	parent->removeObject(this);
    // 	parent->removeObject(m_constraint);
    //       }
    //     parent = NULL;
    //   }
}

}
}
}


#endif


