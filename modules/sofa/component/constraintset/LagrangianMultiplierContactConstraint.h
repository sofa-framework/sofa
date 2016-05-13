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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_LAGRANGIANMULTIPLIERCONTACTCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINTSET_LAGRANGIANMULTIPLIERCONTACTCONSTRAINT_H

#include <sofa/core/behavior/PairInteractionForceField.h>
#include <sofa/component/constraintset/LagrangianMultiplierConstraint.h>
#include <vector>



namespace sofa
{

namespace component
{

namespace constraintset
{

template<class DataTypes>
class LagrangianMultiplierContactConstraint : public LagrangianMultiplierConstraint<DataTypes>, public core::behavior::PairInteractionForceField<DataTypes>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE(LagrangianMultiplierContactConstraint,DataTypes),SOFA_TEMPLATE(LagrangianMultiplierConstraint, DataTypes), SOFA_TEMPLATE(core::behavior::PairInteractionForceField, DataTypes));

    typedef typename core::behavior::PairInteractionForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef defaulttype::StdVectorTypes<Real, Real, Real> LMTypes;
    typedef typename LMTypes::VecCoord LMVecCoord;
    typedef typename LMTypes::VecDeriv LMVecDeriv;
    typedef typename core::behavior::MechanicalState<DataTypes> MechanicalState;
protected:

    struct Contact
    {
        int m1, m2;   ///< the two extremities of the spring: masses m1 and m2
        Deriv norm;   ///< contact normal, from m1 to m2
        Real dist;    ///< minimum distance between the points
        Real ks;      ///< spring stiffness
        Real mu_s;    ///< coulomb friction coefficient (currently unused)
        Real mu_v;    ///< viscous friction coefficient
        Real pen0;     ///< penetration at start of timestep
        Real pen;     ///< current penetration
    };

    sofa::helper::vector<Contact> contacts;

public:

    LagrangianMultiplierContactConstraint(MechanicalState* m1=NULL, MechanicalState* m2=NULL)
        : Inherit(m1, m2)
    {
    }

    MechanicalState* getObject1() { return this->mstate1; }
    MechanicalState* getObject2() { return this->mstate2; }

    void clear(int reserve = 0)
    {
        contacts.clear();
        if (reserve)
            contacts.reserve(reserve);
        this->lambda->resize(0);
    }

    void addContact(int m1, int m2, const Deriv& norm, Real dist, Real ks, Real mu_s = 0.0f, Real mu_v = 0.0f);

    virtual void addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2);

    virtual void addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2);

    virtual double getPotentialEnergy(const VecCoord&, const VecCoord&)
    {
        std::cerr<<"LagrangianMultiplierContactConstraint::getPotentialEnergy-not-implemented !!!"<<std::endl;
        return 0;
    }

    void draw(const core::visual::VisualParams* vparams);


    /// The contact law is simplified during using the LagrangeMultiplier constraint to process the contact response:
    /// When the constraint is set, it can not be removed until the end of the time step
    /// Thus, during the time step, the constraint is holonomic
    bool isHolonomic() {return true;}

};

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
