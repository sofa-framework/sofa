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
#ifndef SOFA_COMPONENT_COLLISION_PERSISTENTFRICTIONCONTACT_H
#define SOFA_COMPONENT_COLLISION_PERSISTENTFRICTIONCONTACT_H

#include <SofaConstraint/FrictionContact.h>

#include <sofa/core/collision/DetectionOutput.h>

#include "PersistentContactMapping.h"

#define DEBUG_INACTIVE_CONTACTS

namespace sofa
{

namespace component
{

namespace container { template< class T > class MechanicalObject; }

namespace collision
{

using namespace sofa::defaulttype;


struct ContactInfo
{
    ContactInfo()
        : m_index1(0)
        , m_index2(0)
        , m_mapper1(false)
        , m_mapper2(false)
        , m_distance(0.0)
        , m_initForce(Vec3d())
    {
    }

    ContactInfo(int id1, int id2, bool map1, bool map2, double dist, Vec3d f)
        : m_index1(id1)
        , m_index2(id2)
        , m_mapper1(map1)
        , m_mapper2(map2)
        , m_distance(dist)
        , m_initForce(f)
    {
    }

    // Contact data
    int m_index1;
    int m_index2;
    bool m_mapper1;
    bool m_mapper2;
    double m_distance;
    Vec3d m_initForce;

    // DetectionOutput data
    std::pair< core::CollisionElementIterator, core::CollisionElementIterator > m_elem;
    int64_t m_detectionOutputId;
    int m_contactId;

    core::CollisionElementIterator getFirstPrimitive() const
    {
        return m_elem.first;
    }

    core::CollisionElementIterator getSecondPrimitive() const
    {
        return m_elem.second;
    }

    void setFirstPrimitive(core::CollisionElementIterator c)
    {
        m_elem.first = c;
    }

    void setSecondPrimitive(core::CollisionElementIterator c)
    {
        m_elem.second = c;
    }
};


template <class TCollisionModel1, class TCollisionModel2>
class PersistentFrictionContact : public FrictionContact<TCollisionModel1, TCollisionModel2>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(PersistentFrictionContact, TCollisionModel1, TCollisionModel2), SOFA_TEMPLATE2(FrictionContact, TCollisionModel1, TCollisionModel2));

    typedef TCollisionModel1 CollisionModel1;
    typedef TCollisionModel2 CollisionModel2;
    typedef core::collision::Intersection Intersection;
    typedef core::collision::DetectionOutputVector OutputVector;
    typedef typename CollisionModel1::DataTypes DataTypes1;
    typedef typename CollisionModel2::DataTypes DataTypes2;
    typedef core::behavior::MechanicalState<DataTypes1> MechanicalState1;
    typedef core::behavior::MechanicalState<DataTypes2> MechanicalState2;
    typedef typename CollisionModel1::Element CollisionElement1;
    typedef typename CollisionModel2::Element CollisionElement2;
    typedef typename FrictionContact<TCollisionModel1, TCollisionModel2>::TOutputVector TOutputVector;
    typedef std::vector< sofa::core::collision::DetectionOutput* > DetectionOutputVector;
    typedef std::map< const sofa::core::collision::DetectionOutput*, ContactInfo > MappedContactsMap;

    typedef FrictionContact<TCollisionModel1, TCollisionModel2> Inherit;

    std::pair<core::CollisionModel*,core::CollisionModel*> getCollisionModels() { return std::make_pair(this->model1,this->model2); }

    PersistentFrictionContact() {}

    PersistentFrictionContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);

    ~PersistentFrictionContact();

    void cleanup();

    /// Set the generic description of a contact point
    void setDetectionOutputs(OutputVector* outputs);

    void createResponse(core::objectmodel::BaseContext* group);

    virtual void removeResponse();

    void init();

#ifdef DEBUG_INACTIVE_CONTACTS
    void draw(const core::visual::VisualParams* vparams);
#endif

protected:

    std::pair<bool,bool> findMappingOrUseMapper();

    template< class T >
    bool findMappingOrUseMapper(core::behavior::MechanicalState<T> *mState, container::MechanicalObject<T> *&constraintModel, component::mapping::PersistentContactMapping *&map);

    void activateConstraint();

    void resetPersistentContactMappings();

    int mapThePersistentContact(Vector3 &, int, Vector3 &, bool)
    {
        serr << "Warning: mapThePersistentContact is not defined for these collision elements" << sendl;
        return 0;
    }

    int keepThePersistentContact(int, bool);

    /// Removes duplicate contacts.
    void filterDuplicatedDetectionOutputs(TOutputVector &input, DetectionOutputVector &output);

    bool haveSameId(const core::collision::DetectionOutput &input_do, const core::collision::DetectionOutput &output_do);

    bool areNear(const core::collision::DetectionOutput &input_do, const core::collision::DetectionOutput &output_do);

    void keepStickyContacts(const DetectionOutputVector &output);

    /// Reset constraint stored forces, contact states...
    void resetConstraintStoredData();

    /// Checks if the DetectionOutput is corresponding to a sticked contact
    /// according to the previous contact resolution.
    bool isSticked(sofa::core::collision::DetectionOutput*) const;

    /// Checks if the DetectionOutput is corresponding to a sliding contact
    /// according to the previous contact resolution.
    bool isSliding(sofa::core::collision::DetectionOutput*) const;

    bool use_mapper_for_state1;
    bool use_mapper_for_state2;

    MechanicalState1 *mstate1;
    MechanicalState2 *mstate2;

    component::container::MechanicalObject< DataTypes1 >* constraintModel1;
    component::container::MechanicalObject< DataTypes2 >* constraintModel2;

    sofa::component::mapping::PersistentContactMapping *map1;
    sofa::component::mapping::PersistentContactMapping *map2;

    std::vector< Vector3 > barycentricValues1;
    std::vector< Vector3 > barycentricValues2;

    MappedContactsMap m_generatedContacts;
    MappedContactsMap m_stickedContacts;
    MappedContactsMap m_slidingContacts;

#ifdef DEBUG_INACTIVE_CONTACTS
    DetectionOutputVector m_inactiveContacts;
#endif
};


} // collision

} // component

} // sofa

#endif // SOFA_COMPONENT_COLLISION_PERSISTENTFRICTIONCONTACT_H
