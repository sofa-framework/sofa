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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_LAGRANGIANMULTIPLIERFIXEDCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINTSET_LAGRANGIANMULTIPLIERFIXEDCONSTRAINT_H

#include <sofa/core/behavior/ForceField.h>
#include <sofa/component/constraintset/LagrangianMultiplierConstraint.h>
#include <vector>


namespace sofa
{

namespace component
{

namespace constraintset
{

template<class DataTypes>
class LagrangianMultiplierFixedConstraint : public LagrangianMultiplierConstraint<DataTypes>, public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS2(SOFA_TEMPLATE(LagrangianMultiplierFixedConstraint,DataTypes),SOFA_TEMPLATE(LagrangianMultiplierContactConstraint,DataTypes), SOFA_TEMPLATE(core::behavior::ForceField,DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef defaulttype::StdVectorTypes<Real, Real, Real> LMTypes;
    typedef typename LMTypes::VecCoord LMVecCoord;
    typedef typename LMTypes::VecDeriv LMVecDeriv;

protected:

    struct PointConstraint
    {
        int indice;   ///< index of the constrained point
        Coord pos;    ///< constrained position of the point
    };

    sofa::helper::vector<PointConstraint> constraints;

public:

    LagrangianMultiplierFixedConstraint()
    {
    }

    void clear(int reserve = 0)
    {
        constraints.clear();
        if (reserve)
            constraints.reserve(reserve);
        this->lambda->resize(0);
    }

    void addConstraint(int indice, const Coord& pos);

    virtual void init();

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    virtual void addDForce (VecDeriv& df, const VecDeriv& dx);

    virtual double getPotentialEnergy(const VecCoord& x);

    void draw(const core::visual::VisualParams* vparams);

    /// this constraint is holonomic
    bool isHolonomic() {return true;}
};

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
