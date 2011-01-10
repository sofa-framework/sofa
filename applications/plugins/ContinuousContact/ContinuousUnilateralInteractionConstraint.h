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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_CONTINUOUSUNILATERALINTERACTIONCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINTSET_CONTINUOUSUNILATERALINTERACTIONCONSTRAINT_H

#include <sofa/component/constraintset/UnilateralInteractionConstraint.h>

#include "ContinuousContact.h"

namespace sofa
{

namespace component
{

namespace constraintset
{

template<class DataTypes>
class ContinuousUnilateralInteractionConstraint;

template<class DataTypes>
class ContinuousUnilateralConstraintResolutionWithFriction : public core::behavior::ConstraintResolution
{
public:
    ContinuousUnilateralConstraintResolutionWithFriction(double mu, PreviousForcesContainer* prev=NULL, bool* active = NULL)
        : _mu(mu)
        , _prev(prev)
        , _active(active)
        , m_constraint(0)
    {
        nbLines=3;
    }

    virtual void init(int line, double** w, double* force);
    virtual void resolution(int line, double** w, double* d, double* force, double *dFree);
    virtual void store(int line, double* force, bool /*convergence*/);

    void setConstraint(ContinuousUnilateralInteractionConstraint<DataTypes> *c)
    {
        m_constraint = c;
    }

    enum ContactState { NONE=0, SLIDING, STICKY };

protected:
    double _mu;
    double _W[6];
    PreviousForcesContainer* _prev;
    bool* _active; // Will set this after the resolution
    ContinuousUnilateralInteractionConstraint<DataTypes> *m_constraint;
};


template<class DataTypes>
class ContinuousUnilateralInteractionConstraint : public UnilateralInteractionConstraint<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ContinuousUnilateralInteractionConstraint, DataTypes), SOFA_TEMPLATE(UnilateralInteractionConstraint, DataTypes));

    typedef UnilateralInteractionConstraint<DataTypes> Inherited;
    typedef typename Inherited::VecCoord VecCoord;
    typedef typename Inherited::VecDeriv VecDeriv;
    typedef typename Inherited::Coord Coord;
    typedef typename Inherited::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename core::behavior::MechanicalState<DataTypes> MechanicalState;
    typedef typename Inherited::PersistentID PersistentID;
    typedef typename Inherited::Contact Contact;
#ifdef SOFA_DEV
    typedef typename ContinuousUnilateralConstraintResolutionWithFriction<DataTypes>::ContactState ContactState;
#endif

    ContinuousUnilateralInteractionConstraint(MechanicalState* object1, MechanicalState* object2)
        : Inherited(object1, object2)
    {
    }

    ContinuousUnilateralInteractionConstraint(MechanicalState* object)
        : Inherited(object)
    {
    }

    ContinuousUnilateralInteractionConstraint()
        : Inherited()
    {
    }

    virtual ~ContinuousUnilateralInteractionConstraint()
    {
    }

    virtual void addContact(double mu, Deriv norm, Coord P, Coord Q, Real contactDistance, int m1, int m2, Coord Pfree, Coord Qfree, long id=0, PersistentID localid=0);

#ifdef SOFA_DEV
    void getConstraintResolution(std::vector< core::behavior::ConstraintResolution* >& resTab, unsigned int& offset);


protected:
    std::map< int, ContactState > contactStates;

public:

    /// @name Contact State API
    /// @{

    bool isSticked(int id);

    void setContactState(int id, ContactState s);

    void clearContactStates();

    void debugContactStates();

    // @}
#endif
};


#if defined(WIN32) && !defined(SOFA_COMPONENT_CONSTRAINTSET_UNILATERALINTERACTIONCONSTRAINT_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_CONTINUOUSCONTACT_API ContinuousUnilateralInteractionConstraint<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_CONTINUOUSCONTACT_API ContinuousUnilateralInteractionConstraint<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONSTRAINTSET_CONTINUOUSUNILATERALINTERACTIONCONSTRAINT_H
