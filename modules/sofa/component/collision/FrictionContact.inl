
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
    , mu (initData(&mu, 0.8, "mu", "friction coefficient (0 for frictionless contacts)"))
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
        if (parent!=NULL)
            parent->removeObject(m_constraint);
        delete m_constraint;
        parent = NULL;
        m_constraint = NULL;
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
        m_constraint = new constraintset::UnilateralInteractionConstraint<Vec3Types>(mmodel1, mmodel2);
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

            // Polynome de Cantor de N� sur N bijectif f(x,y)=((x+y)^2+3x+y)/2
            long index = cantorPolynomia(o->id /*cantorPolynomia(index1, index2)*/,id);

            // add contact in unilateral constraint
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


//////////////////////////////// TEST : Specific response for AdaptiveBeamModel (for now)



template < class TCollisionModel1, class TCollisionModel2 >
ContinuousFrictionContact<TCollisionModel1,TCollisionModel2>::ContinuousFrictionContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
    : FrictionContact<TCollisionModel1, TCollisionModel2>(model1, model2, intersectionMethod)
    , constraintModel1(NULL)
    , constraintModel2(NULL)
    , map1(NULL)
    , map2(NULL)
{
    mstate1 = model1->getMechanicalState();
    mstate2 = model2->getMechanicalState();

    use_mapper_for_state1 = true;
    use_mapper_for_state2 = true;

    std::cout<<" ContinuousFrictionContact created between  mstate1 named " << mstate1->getName() << " and mtate2 named" << mstate2->getName()<<std::endl;
}

template < class TCollisionModel1, class TCollisionModel2 >
ContinuousFrictionContact<TCollisionModel1,TCollisionModel2>::~ContinuousFrictionContact()
{
    std::cout<<"!!!! destructor of ContinuousFrictionContact is called !!!!"<<std::endl;
}

template < class TCollisionModel1, class TCollisionModel2 >
void ContinuousFrictionContact<TCollisionModel1,TCollisionModel2>::cleanup()
{
    std::cout<<"\n*******\n*******ContinuousFrictionContact : ENTERING CLEAN UP\n*******\n*******"<<std::endl;

    if (this->m_constraint)
    {
        this->m_constraint->cleanup();
        if (this->parent!=NULL)
            this->parent->removeObject(this->m_constraint);
        delete this->m_constraint;
        this->parent = NULL;
        this->m_constraint = NULL;

        if(constraintModel1!=NULL)
        {
            constraintModel1->resize(0);
            constraintModel1=NULL;
            map1->beginAddContactPoint();
            map1 = NULL;

        }

        if(constraintModel2!=NULL)
        {
            constraintModel2->resize(0);
            constraintModel2=NULL;
            map2->beginAddContactPoint();
            map2 = NULL;
        }

        this->mapper1.cleanup();
        if (!this->selfCollision) this->mapper2.cleanup();
    }

    this->contacts.clear();
    this->mappedContacts.clear();

    std::cout<<"ContinuousFrictionContact : OUT OF CLEAN UP"<<std::endl;
}


template < class TCollisionModel1, class TCollisionModel2 >
void ContinuousFrictionContact<TCollisionModel1,TCollisionModel2>::setDetectionOutputs(OutputVector* o)
{
    if(!use_mapper_for_state1)
    {
        if(map1 != NULL)
        {
            std::cout<<"  TODO : replace beginAddContactPoint by follow Contacts "<<std::endl;
            map1->beginAddContactPoint();
        }
        else
            serr<<"map1 is not defined in setDetectionOutputs"<<sendl;
    }
    if(!use_mapper_for_state2)
    {
        if(map2 != NULL)
        {
            std::cout<<"  TODO : replace beginAddContactPoint by follow Contacts "<<std::endl;
            map2->beginAddContactPoint();
        }
        else
            serr<<"map2 is not defined in setDetectionOutputs"<<sendl;
    }

    this->FrictionContact<TCollisionModel1,TCollisionModel2>::setDetectionOutputs(o);
}


template < class TCollisionModel1, class TCollisionModel2 >
bool ContinuousFrictionContact<TCollisionModel1,TCollisionModel2>::findMappingOrUseMapper(bool case1)
{
    sofa::core::BaseMapping* map = NULL;
    core::objectmodel::BaseContext *child, *parent = NULL;

    if (case1)
    {
        child = mstate1->getContext();
        if (constraintModel1!=NULL && map1!=NULL)
        {
            //std::cout<<"constraintModel1 already created / found"<<std::endl;
            return true;
        }
    }
    else
    {
        child = mstate2->getContext();
        if (constraintModel2!=NULL && map2!=NULL)
        {
            //std::cout<<"constraintModel2 already created / found"<<std::endl;
            return true;
        }
    }

    child->get(map);

    sofa::simulation::Node* childNode = NULL;
    sofa::simulation::Node* parentNode = NULL;

    if (map)
    {
        helper::vector<core::BaseState*> fromObjects = map->getFrom();

        if (fromObjects.size( ) == 0)
        {
            serr<<" problem with fromObjects size="<<fromObjects.size( )<<sendl;
            return false;
        }

        parent = fromObjects[0]->getContext();
        parentNode = dynamic_cast<simulation::Node*>(parent);
    }
    else
    {
        // specific case: the collision model is not mapped => it is directly put on the degrees of freedom
        parentNode = dynamic_cast<simulation::Node*>(child);
    }

    if (parentNode == NULL)
    {
        serr<<" the cast is not working for parentNode"<<sendl;
        return false;
    }

    childNode = parentNode->getChild("ContinuousFrictionResponse");

    if (childNode != NULL)
    {
        std::cout<<" THE CHILD ALREADY EXISTS !! => only resize MObject"<<std::endl;

        if (case1)
        {
            constraintModel1 = dynamic_cast<component::container::MechanicalObject<DataTypes1 >* > (childNode->getMechanicalState());
            childNode->get(map1);
            if(constraintModel1 != NULL && map1 != NULL)
            {
                //std::cout<<"MObject1 is named "<<constraintModel1->getName()<<" and mapping is named "<<map1->getName()<<std::endl;
                return true;
            }
            else
            {
                serr<<"WARNING dynamic Cast failed => use mapper and not continuousFrictionResponse"<<sendl;
                return false;
            }
        }
        else
        {
            constraintModel2 = dynamic_cast<component::container::MechanicalObject<DataTypes2 >* > (childNode->getMechanicalState());
            childNode->get(map2);
            if(constraintModel2 != NULL && map2 != NULL)
            {
                //std::cout<<"MObject2 is named "<<constraintModel2->getName()<<" and mapping is named "<<map2->getName()<<std::endl;
                return true;
            }
            else
            {
                serr<<"WARNING dynamic Cast failed => use mapper and not continuousFrictionResponse"<<sendl;
                return false;
            }
        }
    }
    else
    {
        //std::cout<<" no child Node named : ContinuousFrictionResponse found"<<std::endl;
        return false;
    }

    /*
    // CODE POUR CREER UN NOUVEAU NOEUD !!!
    child = simulation::getSimulation()->newNode("ContinuousFrictionResponse");
    parent_2->addChild(child);
    std::cout<<"add child node to parent named:"<<parent_2->getName()<<std::endl;

    constraintModel2 = new component::container::MechanicalObject<DataTypes2 >();

    child->addObject(constraintModel2);
    constraintModel2->init();
    constraintModel2->resize(0);
    child->updateSimulationContext();

    sofa::core::behavior::MechanicalState<Rigid3Types> * mstateParent = dynamic_cast< sofa::core::behavior::MechanicalState<Rigid3Types> * > (parent_2->getMechanicalState());

    sofa::component::mapping::AdaptiveBeamMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<Rigid3Types>, sofa::core::behavior::MechanicalState<Vec3Types> > >*
    mapTest = new sofa::component::mapping::AdaptiveBeamMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<Rigid3Types>, sofa::core::behavior::MechanicalState<Vec3Types> > >
    (mstateParent, constraintModel2 );

    child->addObject(mapTest);
    mapTest->init();
    */
}


template < class TCollisionModel1, class TCollisionModel2 >
void ContinuousFrictionContact<TCollisionModel1,TCollisionModel2>::activateConstraint()
{
    /*
      if(use_mapper_for_state1)
      {
          std::cerr<<"activateMappers call for state1"<<std::endl;
      }
      else
          std::cerr<<"*********\nthis->constraintModel1 ="<<this->constraintModel1->getName()<<std::endl;
      if(use_mapper_for_state2)
      {
          std::cerr<<"activateMappers call for state2"<<std::endl;
      }
      else
          std::cerr<<"*********\nthis->constraintModel2 ="<<this->constraintModel2->getName()<<std::endl;
    */


    ////////////////////////////////////// STEP 1 : creation de la contrainte et/ou
    if (!this->m_constraint)
    {
        // TODO : verify that the constraint do not already exists //

        // Get the mechanical model from mapper1 to fill the constraint vector
        MechanicalState1* mmodel1;
        MechanicalState2* mmodel2;

        if (use_mapper_for_state1)
        {
            mmodel1 = this->mapper1.createMapping();
        }
        else
        {
            mmodel1 = (MechanicalState1*) this->constraintModel1;
        }

        if(use_mapper_for_state2)
        {
            mmodel2 = this->selfCollision ? mmodel1 : this->mapper2.createMapping();
        }
        else
        {
            if(this->selfCollision && use_mapper_for_state1!=use_mapper_for_state2)
            {
                this->f_printLog.setValue(true);
                serr<<" problem : selfColision but not same targetting state => constraint not created"<<sendl;
                return;
            }
            if(this->selfCollision)
            {
                std::cout<<"selfCollision"<<std::endl;
                mmodel2 = mmodel1;
            }
            else
            {
                std::cout<<" get Mechanical Model mmodel2 from constraintModel"<<std::endl;
                mmodel2 = (MechanicalState2*) this->constraintModel2;
            }
        }

        this->m_constraint = new constraintset::UnilateralInteractionConstraint<Vec3Types>(mmodel1, mmodel2);
        std::cerr<<"constraint created"<<std::endl;
        this->m_constraint->setName( this->getName() );
    }

    int size = this->contacts.size();
    this->m_constraint->clear(size);

    if(use_mapper_for_state1)
    {
        if(this->selfCollision)
            this->mapper1.resize(2*size);
        else
            this->mapper1.resize(size);
    }

    if(use_mapper_for_state2 && !this->selfCollision)
        this->mapper2.resize(size);

//    std::cerr<<"step2"<<std::endl;
    ////////////////////////////////////// STEP 2  : creation des "mappedContacts" + corrections associées par rapport à la ddc

    int i = 0;
    const double d0 = this->intersectionMethod->getContactDistance() + this->model1->getProximity() + this->model2->getProximity(); // - 0.001;

    this->mappedContacts.resize(this->contacts.size());
    for (std::vector<DetectionOutput*>::const_iterator it = this->contacts.begin(); it!=this->contacts.end(); it++, i++)
    {
        DetectionOutput* o = *it;
        CollisionElement1 elem1(o->elem.first);
        CollisionElement2 elem2(o->elem.second);
        int index1 = elem1.getIndex();
        int index2 = elem2.getIndex();

        double distance = d0;

        typename DataTypes1::Real r1 = 0.0;
        typename DataTypes2::Real r2 = 0.0;
        //double constraintValue = ((o->point[1] - o->point[0]) * o->normal) - intersectionMethod->getContactDistance();

        if(use_mapper_for_state1)
        {
            // Create mapping for first point
            index1 = this->mapper1.addPoint(o->point[0], index1, r1);
            distance += r1;
        }
        else
        {
            Vector3 thickness = o->normal * this->model1->getProximity();
            Vector3 posColpoint=  o->point[0]+thickness;
            index1 = this->mapTheContinuousContact(o->baryCoords[0], index1, posColpoint, true);
            distance -= this->model1->getProximity() ;
        }
        if(use_mapper_for_state2)
        {
            // Create mapping for second point
            index2 = this->selfCollision ? this->mapper1.addPoint(o->point[1], index2, r2) : this->mapper2.addPoint(o->point[1], index2, r2);
            distance +=r2;
        }
        else
        {
            Vector3 thickness = o->normal * this->model2->getProximity();
            Vector3 posColpoint=  o->point[1]-thickness;
            index2 = this->mapTheContinuousContact(o->baryCoords[1], index2, posColpoint, false);
            distance -= this->model2->getProximity() ;
            //index2 = map2->addPointInGlobalCoord(o->point[1]);
        }

        //std::cout<<" mapped contacts : index1 = "<<index1<<"  - index2 = "<<index2<<std::endl;

        this->mappedContacts[i].first.first = index1;
        this->mappedContacts[i].first.second = index2;
        this->mappedContacts[i].second = distance;
    }

    // Update mappings
    if (use_mapper_for_state1)
    {
        this->mapper1.update();
        this->mapper1.updateXfree();
    }
    else
    {
        map1->apply();
        map1->apply(sofa::core::VecCoordId::freePosition(), sofa::core::ConstVecCoordId::freePosition());
    }

    if (use_mapper_for_state2)
    {
        if (!this->selfCollision) this->mapper2.update();
        if (!this->selfCollision) this->mapper2.updateXfree();
    }
    else
    {
        map2->apply();
        map2->apply(sofa::core::VecCoordId::freePosition(), sofa::core::ConstVecCoordId::freePosition());
    }

    //std::cout<<" ===================== "<<std::endl;
//    i = 0;
//    for (std::vector<DetectionOutput*>::const_iterator it = this->contacts.begin(); it!=this->contacts.end(); it++, i++)
//    {
//        DetectionOutput* o = *it;
//
//        if(!use_mapper_for_state2 && this->constraintModel2!=NULL)
//        {
//            Vector3 thickness = o->normal * this->model2->getProximity();
//            int i2 = this->mappedContacts[i].first.second;
//            //std::cout<<" i2 = "<<i2<<std::endl;
//            //std::cout<<" contact["<<i<<"] : xfree before :"<<o->freePoint[1]-thickness<<" after: "<<(*this->constraintModel2->getXfree())[i2]<<"  x before : "<<o->point[1]-thickness<<"  after : "<<(*this->constraintModel2->getX())[i2]<<std::endl;
//        }
//
//    }


    // std::cerr<<" end activateMappers call"<<std::endl;
}


template < class TCollisionModel1, class TCollisionModel2 >
void ContinuousFrictionContact<TCollisionModel1,TCollisionModel2>::createResponse(core::objectmodel::BaseContext* group)
{
    use_mapper_for_state1 = !findMappingOrUseMapper(true);
    use_mapper_for_state2 = !findMappingOrUseMapper(false);

    //std::cout<<" use_mapper_for_state1 :"<<use_mapper_for_state1<<" - use_mapper_for_state2"<<use_mapper_for_state2<<std::endl;

    activateConstraint();

    this->parent = group;

    double mu_ = this->mu.getValue();

    // mu_=0.8;

    if (this->m_constraint)
    {
        int i=0;
        for (std::vector<DetectionOutput*>::const_iterator it = this->contacts.begin(); it!=this->contacts.end(); it++, i++)
        {
            DetectionOutput* o = *it;
            int index1 = this->mappedContacts[i].first.first;
            int index2 = this->mappedContacts[i].first.second;
            double distance = this->mappedContacts[i].second;

            // Polynome de Cantor bijectif f(x,y)=((x+y)^2+3x+y)/2
            long index = cantorPolynomia(o->id /*cantorPolynomia(index1, index2)*/,this->id);

            // add contact in unilateral constraint
            this->m_constraint->addContact(mu_, o->normal, distance, index1, index2, index, o->id);
        }

        if (this->parent!=NULL)
        {
            this->parent->removeObject(this);

            if (!this->use_mapper_for_state1 && map1 != NULL)
                map1->getContext()->removeObject(this->m_constraint);
            else
            {
                if(!this->use_mapper_for_state2 && map2 != NULL )
                    map2->getContext()->removeObject(this->m_constraint);
                else
                    this->parent->removeObject(this->m_constraint);
            }
        }
        this->parent = group;
        if (this->parent!=NULL)
        {

            this->parent->addObject(this);

            if (!this->use_mapper_for_state1 && map1 != NULL )
                map1->getContext()->addObject(this->m_constraint);
            else
            {
                if(!this->use_mapper_for_state2 && map2 != NULL )
                    map2->getContext()->addObject(this->m_constraint);
                else
                    this->parent->addObject(this->m_constraint);
            }
        }
    }
}


template < class TCollisionModel1, class TCollisionModel2 >
void ContinuousFrictionContact<TCollisionModel1,TCollisionModel2>::removeResponse()
{
    if (this->m_constraint!=NULL)
    {
        this->mapper1.resize(0);
        this->mapper2.resize(0);
        if (this->parent!=NULL)
        {
            //sout << "Removing contact response from "<<this->parent->getName()<<sendl;
            this->parent->removeObject(this);
            if (!this->use_mapper_for_state1 && map1 != NULL)
                map1->getContext()->removeObject(this->m_constraint);
            else
            {
                if(!this->use_mapper_for_state2 && map2 != NULL )
                    map2->getContext()->removeObject(this->m_constraint);
                else
                    this->parent->removeObject(this->m_constraint);
            }
        }
        this->parent = NULL;
    }
}

} // namespace collision

} // namespace component

} // namespace sofa

#endif
