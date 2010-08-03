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
#ifndef SOFA_COMPONENT_INTERACTION_LAGRANGEMULTIPLIERINTERACTION_H
#define SOFA_COMPONENT_INTERACTION_LAGRANGEMULTIPLIERINTERACTION_H

#include <sofa/core/behavior/MixedInteractionForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/InteractionConstraint.h>
#include <sofa/core/behavior/Constraint.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{
using namespace sofa::defaulttype;

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class DataTypes1, class DataTypes2>
class LagrangeMultiplierInteractionInternalData
{
public:
};

template<class DataTypes1, class DataTypes2>
class LagrangeMultiplierInteraction : public core::behavior::MixedInteractionForceField<DataTypes1, DataTypes2>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(LagrangeMultiplierInteraction, DataTypes1, DataTypes2), SOFA_TEMPLATE2(core::behavior::MixedInteractionForceField, DataTypes1, DataTypes2));

    typedef core::behavior::BaseConstraint baseConstraint;
    typedef core::behavior::Constraint<DataTypes2> SimpleConstraint;
    typedef core::behavior::InteractionConstraint InteractionConstraint;

    typedef core::behavior::MixedInteractionForceField<DataTypes1, DataTypes2> Inherit;

    typedef typename DataTypes1::VecCoord VecCoord1;
    typedef typename DataTypes1::VecDeriv VecDeriv1;
    typedef typename DataTypes1::Coord Coord1;
    typedef typename DataTypes1::Deriv Deriv1;
    typedef typename Coord1::value_type Real1;
    typedef typename DataTypes2::VecCoord VecCoord2;
    typedef typename DataTypes2::VecDeriv VecDeriv2;
    typedef typename DataTypes2::Coord Coord2;
    typedef typename DataTypes2::Deriv Deriv2;
    typedef typename DataTypes2::MatrixDeriv MatrixDeriv2;
    typedef typename DataTypes2::MatrixDeriv::RowConstIterator MatrixDeriv2RowConstIterator;
    typedef typename DataTypes2::MatrixDeriv::ColConstIterator MatrixDeriv2ColConstIterator;
    typedef typename DataTypes2::MatrixDeriv::RowIterator MatrixDeriv2RowIterator;
    typedef typename DataTypes2::MatrixDeriv::ColIterator MatrixDeriv2ColIterator;
    typedef typename Coord2::value_type Real2;

    Data < std::string > f_constraint;
    Data < std::string > pathObject1;
    Data < std::string > pathObject2;

protected:

    LagrangeMultiplierInteractionInternalData<DataTypes1, DataTypes2> data;
    baseConstraint* constraint;

    std::vector<baseConstraint*>  list_base_constraint;
    std::vector<SimpleConstraint*> list_constraint;
    std::vector<InteractionConstraint*> list_interaction_constraint;

public:

    LagrangeMultiplierInteraction()
        : f_constraint( initData(&f_constraint, "constraint", "constraint path"))
        , pathObject1(initData(&pathObject1,  "object1", "Mechanical State of the Lagrange Multiplier"))
        , pathObject2(initData(&pathObject2,  "object2", "Mechanical Object subject to constraints"))
    {

    }

    virtual void addForce(VecDeriv1& violation, VecDeriv2& f2, const VecCoord1& lambda , const VecCoord2& p2, const VecDeriv1& , const VecDeriv2& v2);

    virtual void addForce2(VecDeriv1& , VecDeriv2& , const VecCoord1& , const VecCoord2& , const VecDeriv1& , const VecDeriv2& ) {}

    virtual void addDForce(VecDeriv1& dViolation, VecDeriv2& df2, const VecDeriv1& dLambda, const VecDeriv2& dx2);

    virtual double getPotentialEnergy(const VecCoord1& , const VecCoord2& ) const { return 0.0;}

    void init();

    void reinit() { init(); }

    void draw() {}
};

} // namespace interactionforcefield

} // namespace component

} // namespace sofa


#endif //SOFA_COMPONENT_INTERACTION_LAGRANGEMULTIPLIERINTERACTION_H
