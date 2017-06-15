/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_COLLISION_PERSISTENTFRICTIONCONTACT_INL
#define SOFA_COMPONENT_COLLISION_PERSISTENTFRICTIONCONTACT_INL

#include "PersistentFrictionContact.h"
#include "PersistentUnilateralInteractionConstraint.inl"

#include <SofaConstraint/FrictionContact.inl>

#include <sofa/helper/gl/template.h>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace core::collision;


template < class TCollisionModel1, class TCollisionModel2 >
PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::PersistentFrictionContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod)
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

    if (this->f_printLog.getValue())
    {
        std::cout << " PersistentFrictionContact created between  mstate1 named " << model1->getName()
                << " and mtate2 named " << model2->getName() << std::endl;
    }
}


template < class TCollisionModel1, class TCollisionModel2 >
PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::~PersistentFrictionContact()
{
    if (this->f_printLog.getValue())
    {
        std::cout << " PersistentFrictionContact destructed between  mstate1 named " << this->model1->getName()
                << " and mtate2 named " << this->model2->getName() << std::endl;
    }
}


template < class TCollisionModel1, class TCollisionModel2 >
void PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::init()
{
    m_stickedContacts.clear();
    m_generatedContacts.clear();

    use_mapper_for_state1 = !findMappingOrUseMapper(mstate1, constraintModel1, map1);
    use_mapper_for_state2 = !findMappingOrUseMapper(mstate2, constraintModel2, map2);
}


template < class TCollisionModel1, class TCollisionModel2 >
void PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::cleanup()
{
    if (this->f_printLog.getValue())
    {
        std::cout << "--> PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::cleanup()\n" << std::endl;
    }

    if (constraintModel1)
    {
        constraintModel1->resize(0);
        constraintModel1 = NULL;
        map1->beginAddContactPoint();
        map1 = NULL;
    }

    if (constraintModel2)
    {
        constraintModel2->resize(0);
        constraintModel2 = NULL;
        map2->beginAddContactPoint();
        map2 = NULL;
    }

    Inherit::cleanup();
}


template < class TCollisionModel1, class TCollisionModel2 >
bool PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::haveSameId(const core::collision::DetectionOutput &input_do, const core::collision::DetectionOutput &output_do)
{
    if (input_do.id == output_do.id)
    {
        if (this->f_printLog.getValue())
        {
            std::cout << "haveSameId\n";
        }

        return true;
    }

    return false;
}


template < class TCollisionModel1, class TCollisionModel2 >
bool PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::areNear(const core::collision::DetectionOutput &input_do, const core::collision::DetectionOutput &output_do)
{
    const double MinDist2 = 0.00000001f;

    if (((input_do.point[0] - output_do.point[0]).norm2() < MinDist2) && ((input_do.point[1] - output_do.point[1]).norm2() < MinDist2))
    {
        if (this->f_printLog.getValue())
        {
            std::cout << "areNear\n";
        }

        return true;
    }

    return false;
}


template < class TCollisionModel1, class TCollisionModel2 >
void PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::filterDuplicatedDetectionOutputs(TOutputVector &input, DetectionOutputVector &output)
{
#ifdef DEBUG_INACTIVE_CONTACTS
    m_inactiveContacts.clear();
#endif

    int inputVectorSize = input.size();

    for (int cpt = 0; cpt < inputVectorSize; cpt++)
    {
        DetectionOutput* input_do = &input[cpt];

        bool filter = false;
        bool insertInput = true;

        for (unsigned int i = 0; i < output.size() && insertInput; i++)
        {
            DetectionOutput* output_do = output[i];

            filter = filter || haveSameId(*input_do, *output_do);
            filter = filter || areNear(*input_do, *output_do);

            if (filter)
            {
                if ((input_do->point[1] - input_do->point[0]).norm2() > (output_do->point[1] - output_do->point[0]).norm2())
                {
                    insertInput = false;
#ifdef DEBUG_INACTIVE_CONTACTS
                    m_inactiveContacts.push_back(input_do);
#endif
                }
                else
                {
#ifdef DEBUG_INACTIVE_CONTACTS
                    //	m_inactiveContacts.push_back(output_do);
#endif
                    output.erase(output.begin() + i);
                }

                filter = false;
            }
        }

        if (insertInput)
            output.push_back(input_do);
    }
}


template < class TCollisionModel1, class TCollisionModel2 >
void PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::keepStickyContacts(const DetectionOutputVector &input)
{
    using sofa::core::collision::DetectionOutput;

    typedef constraintset::PersistentUnilateralInteractionConstraint< Vec3Types > PersistentConstraint;

    m_stickedContacts.clear();
    m_slidingContacts.clear();

    if (this->m_constraint)
    {
        const PersistentConstraint *cc = static_cast< PersistentConstraint* >(this->m_constraint.get());

        if (this->f_printLog.getValue())
        {
            cc->debugContactStates();
        }

        DetectionOutputVector::const_iterator it = input.begin();
        DetectionOutputVector::const_iterator itEnd = input.end();

        while (it != itEnd)
        {
            MappedContactsMap::iterator itOld = m_generatedContacts.begin();
            MappedContactsMap::iterator itOldEnd = m_generatedContacts.end();

            while (itOld != itOldEnd)
            {
                const ContactInfo &oldContact = itOld->second;

                if (cc->isSticked(oldContact.m_contactId))
                {
                    if ((oldContact.getFirstPrimitive() == (*it)->elem.first)
                        && (oldContact.getSecondPrimitive() == (*it)->elem.second))
                    {
                        if (this->f_printLog.getValue())
                        {
                            std::cout << (*it)->id << " -> Found a persistent sticked contact between " << (*it)->elem.first.getCollisionModel()->getName()
                                    << " and " << (*it)->elem.second.getCollisionModel()->getName() << "\n";
                        }

                        m_stickedContacts.insert(std::make_pair(*it, ContactInfo()));

                        if (!oldContact.m_mapper1)
                        {
                            m_stickedContacts[*it].m_index1 = oldContact.m_index1;
                        }

                        if (!oldContact.m_mapper2)
                        {
                            m_stickedContacts[*it].m_index2 = oldContact.m_index2;
                        }

                        m_stickedContacts[*it].m_initForce = cc->getContactForce(oldContact.m_contactId);

                        // Remove related detection output info from old lists
                        m_generatedContacts.erase(itOld);

                        break;
                    }
                }
                else if (cc->isSliding(oldContact.m_contactId))
                {
                    if ((oldContact.getFirstPrimitive() == (*it)->elem.first)
                        && (oldContact.getSecondPrimitive() == (*it)->elem.second))
                    {
                        if (this->f_printLog.getValue())
                        {
                            std::cout << (*it)->id << " -> Found a sliding contact between " << (*it)->elem.first.getCollisionModel()->getName()
                                    << " and " << (*it)->elem.second.getCollisionModel()->getName() << "\n";
                        }

                        m_slidingContacts.insert(std::make_pair(*it, ContactInfo()));
                        m_slidingContacts[*it].m_initForce = cc->getContactForce(oldContact.m_contactId);

                        // Remove related detection output info from old lists
                        m_generatedContacts.erase(itOld);

                        break;
                    }
                }

                ++itOld;
            }

            ++it;
        }

        resetConstraintStoredData();
    }
}


template < class TCollisionModel1, class TCollisionModel2 >
void PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::resetConstraintStoredData()
{
    typedef constraintset::PersistentUnilateralInteractionConstraint< Vec3Types > PersistentConstraint;

    PersistentConstraint *cc = static_cast< PersistentConstraint* >(this->m_constraint.get());

    cc->clearContactStates();
    cc->clearContactForces();
    cc->clearInitForces();
}


template < class TCollisionModel1, class TCollisionModel2 >
void PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::setDetectionOutputs(OutputVector* o)
{
    DetectionOutputVector filteredOutputs;

    if (o != NULL)
    {
        TOutputVector& outputs = *static_cast< TOutputVector* >(o);

        if (this->f_printLog.getValue())
        {
            if (!m_generatedContacts.empty() && (m_generatedContacts.size() != outputs.size()))
            {
                std::cout << "Diff in contacts : " << (int)(outputs.size() - m_generatedContacts.size()) << std::endl;
            }
        }

        filterDuplicatedDetectionOutputs(outputs, filteredOutputs);

        if (this->f_printLog.getValue())
        {
            if (outputs.size() != filteredOutputs.size())
            {
                std::cout << outputs.size() - filteredOutputs.size() << " contact(s) filtered\n";
            }
        }

        keepStickyContacts(filteredOutputs);
    }

    if (this->f_printLog.getValue())
    {
        if (m_generatedContacts.size() > 0)
        {
            std::cout << "Lost " << m_generatedContacts.size() << " PersistentContact" << (m_generatedContacts.size() > 1 ? "s" : "") << "\n";
        }
    }

    // Replace contacts by new ones
    m_generatedContacts.clear();


    this->contacts.clear();
    this->contacts.reserve(filteredOutputs.size());

    for (unsigned int i = 0; i < filteredOutputs.size(); i++)
    {
        this->contacts.push_back(filteredOutputs[i]);
    }
}


template < class TCollisionModel1, class TCollisionModel2 >
void PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::resetPersistentContactMappings()
{
    if (!use_mapper_for_state1)
    {
        if (map1)
        {
            map1->beginAddContactPoint();
        }
        else
            serr << "map1 is not defined in setDetectionOutputs" << sendl;
    }

    if (!use_mapper_for_state2)
    {
        if (map2)
        {
            map2->beginAddContactPoint();
        }
        else
            serr << "map2 is not defined in setDetectionOutputs" << sendl;
    }
}


template< class TCollisionModel1, class TCollisionModel2 >
template< class T >
bool PersistentFrictionContact<TCollisionModel1, TCollisionModel2>::findMappingOrUseMapper(
    core::behavior::MechanicalState<T> *mState, container::MechanicalObject<T> *&constraintModel, component::mapping::PersistentContactMapping *&map)
{
    using sofa::core::objectmodel::BaseContext;
    using sofa::simulation::Node;

    if (constraintModel && map)
    {
        return true;
    }

    BaseContext *child = mState->getContext();

    sofa::core::BaseMapping* baseMap = NULL;
    child->get(baseMap);

    Node* parentNode = NULL;

    if (baseMap)
    {
        helper::vector< core::BaseState* > fromObjects = baseMap->getFrom();

        if (fromObjects.empty())
        {
            serr << "PersistentFrictionContact::Problem with fromObjects size = " << fromObjects.size() << sendl;
            return false;
        }

        BaseContext *parent = fromObjects[0]->getContext();
        parentNode = dynamic_cast< Node* >(parent);
    }
    else
    {
        // Specific case: the collision model is not mapped => it is directly put on the degrees of freedom
        parentNode = dynamic_cast< Node* >(child);
    }

    if (parentNode == NULL)
    {
        serr << "PersistentFrictionContact::Error 1 in findMappingOrUseMapper" << sendl;
        return false;
    }

    typedef helper::vector< component::mapping::PersistentContactMapping* > PersistentContactMappings;

    PersistentContactMappings persistentMappings;

    parentNode->getTreeObjects< component::mapping::PersistentContactMapping, PersistentContactMappings >(&persistentMappings);

    PersistentContactMappings::const_iterator it = persistentMappings.begin();
    PersistentContactMappings::const_iterator itEnd = persistentMappings.end();

    while (it != itEnd)
    {
        if ((*it)->m_nameOfInputMap.getValue() == baseMap->getName())
        {
            map = *it;
            constraintModel = dynamic_cast< container::MechanicalObject<T >* > ((*it)->getContext()->getMechanicalState());
            return (constraintModel && map);
        }

        ++it;
    }

    return false;
}


template< class TCollisionModel1, class TCollisionModel2 >
std::pair<bool, bool> PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::findMappingOrUseMapper()
{
    std::pair<bool, bool> retValue;

    retValue.first = findMappingOrUseMapper(mstate1, constraintModel1, map1);
    retValue.second = findMappingOrUseMapper(mstate2, constraintModel2, map2);

    return retValue;
}


template < class TCollisionModel1, class TCollisionModel2 >
void PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::activateConstraint()
{
    ////////////////////////////////////// STEP 1 : creation de la contrainte et/ou
    if (!this->m_constraint)
    {
        // TODO : verify that the constraint do not already exists //

        // Get the mechanical model from mapper1 to fill the constraint vector
        MechanicalState1 *mmodel1 = use_mapper_for_state1 ? this->mapper1.createMapping() : (MechanicalState1*)this->constraintModel1;

        MechanicalState2 *mmodel2;

        if (use_mapper_for_state2)
        {
            mmodel2 = this->selfCollision ? mmodel1 : this->mapper2.createMapping();
        }
        else
        {
            if (this->selfCollision && use_mapper_for_state1 != use_mapper_for_state2)
            {
                this->f_printLog.setValue(true);
                serr << "Problem : selfColision but not same targetting state => constraint not created" << sendl;
                return;
            }

            mmodel2 = this->selfCollision ? mmodel1 : (MechanicalState2*)this->constraintModel2;
        }

        this->m_constraint = new constraintset::PersistentUnilateralInteractionConstraint<Vec3Types>(mmodel1, mmodel2);

        if (this->f_printLog.getValue())
        {
            std::cerr << "Constraint created" << std::endl;
        }

        this->m_constraint->setName( this->getName() );
        static_cast< constraintset::PersistentUnilateralInteractionConstraint<Vec3Types> * >(this->m_constraint.get())->clearContactStates();
    }

    int size = this->contacts.size();
    this->m_constraint->clear(size);

    if (use_mapper_for_state1)
    {
        if (this->selfCollision)
            this->mapper1.resize(2*size);
        else
            this->mapper1.resize(size);
    }

    if (use_mapper_for_state2 && !this->selfCollision)
        this->mapper2.resize(size);

    ////////////////////////////////////// STEP 2  : creation des "mappedContacts" + corrections associées par rapport à la ddc

    const double d0 = this->intersectionMethod->getContactDistance() + this->model1->getProximity() + this->model2->getProximity(); // - 0.001;

    for (std::vector<DetectionOutput*>::const_iterator it = this->contacts.begin(); it!=this->contacts.end(); it++)
    {
        DetectionOutput* o = *it;
        CollisionElement1 elem1(o->elem.first);
        CollisionElement2 elem2(o->elem.second);
        int index1 = elem1.getIndex();
        int index2 = elem2.getIndex();
        bool m1 = false;
        bool m2 = false;

        double distance = d0;
        Vec3d f;

        if (isSticked(o))
            f = m_stickedContacts[o].m_initForce;
        else if (isSliding(o))
            f = m_slidingContacts[o].m_initForce;

        typename DataTypes1::Real r1 = 0.0;
        typename DataTypes2::Real r2 = 0.0;
        //double constraintValue = ((o->point[1] - o->point[0]) * o->normal) - intersectionMethod->getContactDistance();

        if (use_mapper_for_state1)
        {
            // Create mapping for first point
            index1 = this->mapper1.addPoint(o->point[0], index1, r1);
            distance += r1;
            m1 = true;
        }
        else
        {
            if (isSticked(o))
            {
                index1 = this->keepThePersistentContact(m_stickedContacts[o].m_index1, true);
            }
            else
            {
                Vector3 thickness = o->normal * this->model1->getProximity();
                Vector3 posColpoint = o->point[0] + thickness;
                index1 = this->mapThePersistentContact(o->baryCoords[0], index1, posColpoint, true);
            }

            distance -= this->model1->getProximity();
            m1 = false;
        }

        if (use_mapper_for_state2)
        {
            // Create mapping for second point
            index2 = this->selfCollision ? this->mapper1.addPoint(o->point[1], index2, r2) : this->mapper2.addPoint(o->point[1], index2, r2);
            distance += r2;
            m2 = true;
        }
        else
        {
            if (isSticked(o))
            {
                index2 = this->keepThePersistentContact(m_stickedContacts[o].m_index2, false);
            }
            else
            {
                Vector3 thickness = o->normal * this->model2->getProximity();
                Vector3 posColpoint = o->point[1] - thickness;
                index2 = this->mapThePersistentContact(o->baryCoords[1], index2, posColpoint, false);
            }

            distance -= this->model2->getProximity();
            m2 = false;
        }

        m_generatedContacts.insert(std::make_pair(*it, ContactInfo(index1, index2, m1, m2, distance, f)));
    }

    // Update mappings
    if (use_mapper_for_state1)
    {
        //	std::cout<<" Q pos comes from mapper"<<std::endl;
        this->mapper1.update();
        this->mapper1.updateXfree();
    }
    else
    {
        map1->applyPositionAndFreePosition();
    }

    if (use_mapper_for_state2)
    {
        //	std::cout<<" P pos comes from mapper"<<std::endl;
        if (!this->selfCollision)
            this->mapper2.update();

        if (!this->selfCollision)
            this->mapper2.updateXfree();
    }
    else
    {
        map2->applyPositionAndFreePosition();
    }
}


template < class TCollisionModel1, class TCollisionModel2 >
void PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::createResponse(core::objectmodel::BaseContext* group)
{
    use_mapper_for_state1 = !findMappingOrUseMapper(mstate1, constraintModel1, map1);
    use_mapper_for_state2 = !findMappingOrUseMapper(mstate2, constraintModel2, map2);

    resetPersistentContactMappings();

    activateConstraint();

    this->parent = group;

    double mu_ = this->mu.getValue();

    if (this->m_constraint)
    {
        for (std::vector<DetectionOutput*>::const_iterator it = this->contacts.begin(); it != this->contacts.end(); it++)
        {
            DetectionOutput *o = *it;

            MappedContactsMap::iterator genContactIt = m_generatedContacts.find(o);

            if (genContactIt != m_generatedContacts.end())
            {
                ContactInfo& newContact = genContactIt->second;

                int index1		= newContact.m_index1;
                int index2		= newContact.m_index2;
                double distance	= newContact.m_distance;
                Vec3d initForce = newContact.m_initForce;

                // Polynome de Cantor bijectif f(x,y)=((x+y)^2+3x+y)/2
                long index = cantorPolynomia(o->id, this->id);

                // Add contact in PersistentUnilateralInteractionConstraint
                typedef constraintset::PersistentUnilateralInteractionConstraint<Vec3Types> PersistentConstraint;
                PersistentConstraint *persistent_constraint = static_cast< PersistentConstraint * >(this->m_constraint.get());

                persistent_constraint->addContact(mu_, o->normal, distance, index1, index2, index, o->id);

                persistent_constraint->setInitForce(index, initForce);

                // Store generated contact detectionOutput data and contact id
                newContact.m_detectionOutputId = o->id;
                newContact.m_contactId = index;
                newContact.setFirstPrimitive(o->elem.first);
                newContact.setSecondPrimitive(o->elem.second);
            }
        }

        if (this->parent)
        {
            this->parent->removeObject(this);

            if (map1 && !use_mapper_for_state1)
                map1->getContext()->removeObject(this->m_constraint);
            else
            {
                if (map2 && !use_mapper_for_state2)
                    map2->getContext()->removeObject(this->m_constraint);
                else
                    this->parent->removeObject(this->m_constraint);
            }
        }

        this->parent = group;

        if (this->parent)
        {
            this->parent->addObject(this);

            if (map1 && !use_mapper_for_state1)
                map1->getContext()->addObject(this->m_constraint);
            else
            {
                if (map2 && !use_mapper_for_state2)
                    map2->getContext()->addObject(this->m_constraint);
                else
                    this->parent->addObject(this->m_constraint);
            }
        }
    }
}


template < class TCollisionModel1, class TCollisionModel2 >
void PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::removeResponse()
{
    if (this->m_constraint)
    {
        this->mapper1.resize(0);
        this->mapper2.resize(0);

        if (this->parent)
        {
            this->parent->removeObject(this);

            if (map1 && !use_mapper_for_state1)
                map1->getContext()->removeObject(this->m_constraint);
            else
            {
                if (map2 && !use_mapper_for_state2)
                    map2->getContext()->removeObject(this->m_constraint);
                else
                    this->parent->removeObject(this->m_constraint);
            }
        }
        this->parent = NULL;
    }
}


template < class TCollisionModel1, class TCollisionModel2 >
bool PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::isSticked(sofa::core::collision::DetectionOutput *o) const
{
    return m_stickedContacts.find(o) != m_stickedContacts.end();
}

template < class TCollisionModel1, class TCollisionModel2 >
bool PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::isSliding(sofa::core::collision::DetectionOutput *o) const
{
    return m_slidingContacts.find(o) != m_slidingContacts.end();
}


template < class TCollisionModel1, class TCollisionModel2 >
int PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::keepThePersistentContact(int index, bool case1)
{
    if (case1)
    {
        return map1->keepContactPointFromInputMapping(index);
    }
    else
    {
        return map2->keepContactPointFromInputMapping(index);
    }
}

#ifdef DEBUG_INACTIVE_CONTACTS
template < class TCollisionModel1, class TCollisionModel2 >
void PersistentFrictionContact<TCollisionModel1,TCollisionModel2>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowInteractionForceFields()) return;

    glDisable(GL_LIGHTING);

    for (unsigned int i=0; i< this->m_inactiveContacts.size(); i++)
    {
        glLineWidth(2);
        glBegin(GL_LINES);
        glColor4f(0.6f,0.2f,0.2f,1.f);

        helper::gl::glVertexT(m_inactiveContacts[i]->point[0]);
        helper::gl::glVertexT(m_inactiveContacts[i]->point[1]);

        glEnd();

        glLineWidth(1);
    }

    for (MappedContactsMap::iterator it=m_stickedContacts.begin(); it!=m_stickedContacts.end(); ++it)
    {
        glLineWidth(4);
        glBegin(GL_LINES);
        glColor4f(0.2f,0.6f,0.2f,1.f);

        helper::gl::glVertexT(it->first->point[0]);
        helper::gl::glVertexT(it->first->point[1]);

        glEnd();

        glLineWidth(1);
    }

    for (MappedContactsMap::iterator it=m_slidingContacts.begin(); it!=m_slidingContacts.end(); ++it)
    {
        glLineWidth(3);
        glBegin(GL_LINES);
        glColor4f(0.2f,0.2f,0.6f,1.f);

        helper::gl::glVertexT(it->first->point[0]);
        helper::gl::glVertexT(it->first->point[1]);

        glEnd();

        glLineWidth(1);
    }
}
#endif

} // namespace collision

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_COLLISION_PERSISTENTFRICTIONCONTACT_INL
