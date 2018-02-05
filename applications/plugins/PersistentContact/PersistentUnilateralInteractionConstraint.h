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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_PERSISTENTUNILATERALINTERACTIONCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINTSET_PERSISTENTUNILATERALINTERACTIONCONSTRAINT_H

#include <SofaConstraint/UnilateralInteractionConstraint.h>

#include <PersistentContact/config.h>

namespace sofa
{

namespace component
{

namespace constraintset
{

template<class DataTypes>
class PersistentUnilateralInteractionConstraint;

template<class DataTypes>
class PersistentUnilateralConstraintResolutionWithFriction : public core::behavior::ConstraintResolution
{
public:

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    PersistentUnilateralConstraintResolutionWithFriction(double mu, bool* active = NULL)
        : _mu(mu)
        , _active(active)
        , m_constraint(0)
    {
        nbLines=3;
    }

    virtual void init(int line, double** w, double* force);
    virtual void resolution(int line, double** w, double* d, double* force, double *dFree);
    virtual void store(int line, double* force, bool /*convergence*/);

    void setConstraint(PersistentUnilateralInteractionConstraint<DataTypes> *c)
    {
        m_constraint = c;
    }

    void setInitForce(defaulttype::Vec3d f)
    {
        _f[0] = f.x();
        _f[1] = f.y();
        _f[2] = f.z();
    }

    enum ContactState { NONE=0, SLIDING, STICKY };

protected:
    double _mu;
    double _W[6];
    double _f[3];
    bool* _active; // Will set this after the resolution
    PersistentUnilateralInteractionConstraint<DataTypes> *m_constraint;
};


template<class DataTypes>
class PersistentUnilateralInteractionConstraint : public UnilateralInteractionConstraint<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PersistentUnilateralInteractionConstraint, DataTypes), SOFA_TEMPLATE(UnilateralInteractionConstraint, DataTypes));

    typedef UnilateralInteractionConstraint<DataTypes> Inherited;
    typedef typename Inherited::VecCoord VecCoord;
    typedef typename Inherited::VecDeriv VecDeriv;
    typedef typename Inherited::Coord Coord;
    typedef typename Inherited::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename core::behavior::MechanicalState<DataTypes> MechanicalState;
    typedef typename Inherited::PersistentID PersistentID;
    typedef typename Inherited::Contact Contact;
    typedef typename PersistentUnilateralConstraintResolutionWithFriction<DataTypes>::ContactState ContactState;
    typedef typename std::map< int, ContactState > ContactStateMap;
    typedef typename ContactStateMap::const_iterator contactStateIterator;


    PersistentUnilateralInteractionConstraint(MechanicalState* object1, MechanicalState* object2)
        : Inherited(object1, object2)
    {
    }

    PersistentUnilateralInteractionConstraint(MechanicalState* object)
        : Inherited(object)
    {
    }

    PersistentUnilateralInteractionConstraint()
        : Inherited()
    {
    }

    virtual ~PersistentUnilateralInteractionConstraint()
    {
    }

    virtual void addContact(double mu, Deriv norm, Real contactDistance, int m1, int m2, long id=0, PersistentID localid=0);

    void getConstraintResolution(std::vector< core::behavior::ConstraintResolution* >& resTab, unsigned int& offset);


protected:
    std::map< int, ContactState > contactStates;
    std::map< int, Deriv > contactForces;
    std::map< int, Deriv > initForces;

    /// Computes constraint violation in position and stores it into resolution global vector
    ///
    /// @param v Global resolution vector
    virtual void getPositionViolation(defaulttype::BaseVector *v);

    ///Computes constraint violation in velocity and stores it into resolution global vector
    ///
    /// @param v Global resolution vector
    virtual void getVelocityViolation(defaulttype::BaseVector *v);

public:

    /// @name Contact State API
    /// @{

    bool isSticked(int id) const;

    bool isSliding(int id) const;

    void setContactState(int id, ContactState s);

    void clearContactStates();

    void debugContactStates() const;

    // @}

    /// @name LCP Hot Start API
    /// @{

    void setContactForce(int id, Deriv f);

    Deriv getContactForce(int id) const;

    void clearContactForces();

    void setInitForce(int id, Deriv f);

    Deriv getInitForce(int id);

    void clearInitForces();

    // @}

    void draw(const core::visual::VisualParams* vparams);
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_CONSTRAINTSET_UNILATERALINTERACTIONCONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_PERSISTENTCONTACT_API PersistentUnilateralInteractionConstraint<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_PERSISTENTCONTACT_API PersistentUnilateralInteractionConstraint<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONSTRAINTSET_PERSISTENTUNILATERALINTERACTIONCONSTRAINT_H
