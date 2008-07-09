/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_CONSTRAINT_BILATERALINTERACTIONCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_BILATERALINTERACTIONCONSTRAINT_H

#include <sofa/core/componentmodel/behavior/InteractionConstraint.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/VisualModel.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace constraint
{

template<class DataTypes>
class BilateralInteractionConstraint : public core::componentmodel::behavior::InteractionConstraint
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecConst VecConst;
    typedef typename DataTypes::SparseVecDeriv SparseVecDeriv;
    typedef typename DataTypes::SparseDeriv SparseDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename core::componentmodel::behavior::MechanicalState<DataTypes> MechanicalState;

protected:
    MechanicalState* object1;
    MechanicalState* object2;
    bool yetIntegrated;

    Coord dfree;
    unsigned int cid;

    Data<int> m1;
    Data<int> m2;

public:

    BilateralInteractionConstraint(MechanicalState* object1, MechanicalState* object2)
        : object1(object1), object2(object2), yetIntegrated(false)
        , m1(initData(&m1, 0, "first_point","index of the constraint on the first model"))
        , m2(initData(&m2, 0, "second_point","index of the constraint on the second model"))
    {
    }

    BilateralInteractionConstraint(MechanicalState* object)
        : object1(object), object2(object), yetIntegrated(false)
        , m1(initData(&m1, 0, "first_point","index of the constraint on the first model"))
        , m2(initData(&m2, 0, "second_point","index of the constraint on the second model"))
    {
    }

    BilateralInteractionConstraint()
        : object1(NULL), object2(NULL), yetIntegrated(false)
        , m1(initData(&m1, 0, "first_point","index of the constraint on the first model"))
        , m2(initData(&m2, 0, "second_point","index of the constraint on the second model"))
    {
    }

    virtual ~BilateralInteractionConstraint()
    {
    }

    MechanicalState* getObject1() { return object1; }
    MechanicalState* getObject2() { return object2; }
    core::componentmodel::behavior::BaseMechanicalState* getMechModel1() { return object1; }
    core::componentmodel::behavior::BaseMechanicalState* getMechModel2() { return object2; }

    virtual void init();

    virtual void applyConstraint(unsigned int & /*constraintId*/, double & /*unused*/);

    virtual void getConstraintValue(double* v /*, unsigned int &numContacts */);

    virtual void getConstraintId(long* id, unsigned int &offset);

    virtual void getConstraintType(bool* type, unsigned int &offset);

    // Previous Constraint Interface
    virtual void projectResponse() {}
    virtual void projectVelocity() {}
    virtual void projectPosition() {}
    virtual void projectFreeVelocity() {}
    virtual void projectFreePosition() {}

    /// Pre-construction check method called by ObjectFactory.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (arg->getAttribute("object1") || arg->getAttribute("object2"))
        {
            if (dynamic_cast<MechanicalState*>(arg->findObject(arg->getAttribute("object1",".."))) == NULL)
                return false;
            if (dynamic_cast<MechanicalState*>(arg->findObject(arg->getAttribute("object2",".."))) == NULL)
                return false;
        }
        else
        {
            if (dynamic_cast<MechanicalState*>(context->getMechanicalState()) == NULL)
                return false;
        }
        return core::componentmodel::behavior::InteractionConstraint::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static void create(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        core::componentmodel::behavior::InteractionConstraint::create(obj, context, arg);
        if (arg && (arg->getAttribute("object1") || arg->getAttribute("object2")))
        {
            obj->object1 = dynamic_cast<MechanicalState*>(arg->findObject(arg->getAttribute("object1","..")));
            obj->object2 = dynamic_cast<MechanicalState*>(arg->findObject(arg->getAttribute("object2","..")));
        }
        else if (context)
        {
            obj->object1 =
                obj->object2 =
                        dynamic_cast<MechanicalState*>(context->getMechanicalState());
        }
    }

    void draw();

    /// this constraint is holonomic
    bool isHolonomic() {return true;}
};
} // namespace constraint

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONSTRAINT_BILATERALINTERACTIONCONSTRAINT_H
