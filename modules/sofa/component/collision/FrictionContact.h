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
#ifndef SOFA_COMPONENT_COLLISION_FRICTIONCONTACT_H
#define SOFA_COMPONENT_COLLISION_FRICTIONCONTACT_H

#include <sofa/core/collision/Contact.h>
#include <sofa/core/collision/Intersection.h>
#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/component/constraintset/UnilateralInteractionConstraint.h>
#include <sofa/helper/Factory.h>
#include <sofa/component/collision/BaseContactMapper.h>
#include <sofa/component/collision/RigidContactMapper.inl>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/BaseMapping.h>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

class Identifier
{
public:
    Identifier()
    {
        if (!availableId.empty())
        {
            id = availableId.front();
            availableId.pop_front();
        }
        else
            id = cpt++;

        //	sout << id << sendl;
    }

    virtual ~Identifier()
    {
        availableId.push_back(id);
    }

protected:
    static sofa::core::collision::DetectionOutput::ContactId cpt;
    sofa::core::collision::DetectionOutput::ContactId id;
    static std::list<sofa::core::collision::DetectionOutput::ContactId> availableId;
};


template <class TCollisionModel1, class TCollisionModel2>
class FrictionContact : public core::collision::Contact, public Identifier
{
public:
    typedef TCollisionModel1 CollisionModel1;
    typedef TCollisionModel2 CollisionModel2;
    typedef core::collision::Intersection Intersection;
    typedef typename CollisionModel1::DataTypes DataTypes1;
    typedef typename CollisionModel2::DataTypes DataTypes2;
    typedef core::behavior::MechanicalState<DataTypes1> MechanicalState1;
    typedef core::behavior::MechanicalState<DataTypes2> MechanicalState2;
    typedef typename CollisionModel1::Element CollisionElement1;
    typedef typename CollisionModel2::Element CollisionElement2;
    typedef core::collision::DetectionOutputVector OutputVector;
    typedef core::collision::TDetectionOutputVector<CollisionModel1,CollisionModel2> TOutputVector;

protected:
    CollisionModel1* model1;
    CollisionModel2* model2;
    Intersection* intersectionMethod;
    bool selfCollision; ///< true if model1==model2 (in this case, only mapper1 is used)
    ContactMapper<CollisionModel1,DataTypes1> mapper1;
    ContactMapper<CollisionModel2,DataTypes2> mapper2;

    constraintset::UnilateralInteractionConstraint<Vec3Types>* m_constraint;
    core::objectmodel::BaseContext* parent;

    Data<double> mu;
    std::vector<sofa::core::collision::DetectionOutput*> contacts;
    std::vector< std::pair< std::pair<int, int>, double > > mappedContacts;

    void activateMappers();


public:

    FrictionContact() {}

    FrictionContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);
    virtual ~FrictionContact();

    void cleanup();

    std::pair<core::CollisionModel*,core::CollisionModel*> getCollisionModels() { return std::make_pair(model1,model2); }

    void setDetectionOutputs(OutputVector* outputs);

    void createResponse(core::objectmodel::BaseContext* group);

    void removeResponse();
};

long cantorPolynomia(sofa::core::collision::DetectionOutput::ContactId x, sofa::core::collision::DetectionOutput::ContactId y)
{
    // Polynome de Cantor de NxN sur N bijectif f(x,y)=((x+y)^2+3x+y)/2
    return (long)(((x+y)*(x+y)+3*x+y)/2);
}


/////// TEST:
template <class TCollisionModel1, class TCollisionModel2>
class ContinuousFrictionContact : public FrictionContact<TCollisionModel1, TCollisionModel2>
{
    SOFA_CLASS(SOFA_TEMPLATE2(ContinuousFrictionContact,TCollisionModel1,TCollisionModel2), SOFA_TEMPLATE2(FrictionContact,TCollisionModel1,TCollisionModel2));

public:
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

    std::pair<core::CollisionModel*,core::CollisionModel*> getCollisionModels() { return std::make_pair(this->model1,this->model2); }

    ContinuousFrictionContact() {}

    ContinuousFrictionContact(CollisionModel1* model1, CollisionModel2* model2, Intersection* intersectionMethod);

    ~ContinuousFrictionContact();

    void cleanup();

    /// Set the generic description of a contact point
    void setDetectionOutputs(OutputVector* outputs);

    void createResponse(core::objectmodel::BaseContext* group);

    virtual void removeResponse();

protected:

    std::pair<bool,bool> findMappingOrUseMapper();

    template< class T >
    bool findMappingOrUseMapper(core::behavior::MechanicalState<T> *mState, container::MechanicalObject<T> *&constraintModel, core::BaseMapping *&map);

    void activateConstraint();

    int mapTheContinuousContact(Vector3 &, int, Vector3 &, bool)
    {
        serr << "Warning: mapTheContinuousContact is not defined for these collision elements" << sendl;
        return 0;
    }

    bool use_mapper_for_state1;
    bool use_mapper_for_state2;

    MechanicalState1 *mstate1;
    MechanicalState2 *mstate2;

    component::container::MechanicalObject< DataTypes1 >* constraintModel1;
    component::container::MechanicalObject< DataTypes2 >* constraintModel2;

    sofa::core::BaseMapping *map1;
    sofa::core::BaseMapping *map2;

    std::vector<Vector3> barycentricValues1;
    std::vector<Vector3> barycentricValues2;
};



} // collision

} // component

} // sofa

#endif // SOFA_COMPONENT_COLLISION_FRICTIONCONTACT_H
