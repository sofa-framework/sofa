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
#ifndef SOFA_COMPONENT_CONSTRAINT_UNILATERALINTERACTIONCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_UNILATERALINTERACTIONCONSTRAINT_H

#include <sofa/core/behavior/InteractionConstraint.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/component/component.h>
#include <sofa/defaulttype/VecTypes.h>
#include <iostream>
#include <deque>


namespace sofa
{

namespace component
{

namespace constraint
{
#ifdef SOFA_DEV
class UnilateralConstraintResolution : public core::behavior::ConstraintResolution
{
public:
    virtual void resolution(int line, double** w, double* d, double* force)
    {
//		std::cout<< "UnilateralConstraintResolution (without friction): verify resolution ! "<<std::endl;
        if(d[line]<0)
            force[line] -= d[line] / w[line][line];
        else
            force[line] = 0.0;
    }
};

class UnilateralConstraintResolutionWithFriction : public core::behavior::ConstraintResolution
{
public:
    UnilateralConstraintResolutionWithFriction(double mu, std::deque<double>* vec=NULL) : _mu(mu), _vec(vec) { nbLines=3; }
    virtual void init(int line, double** w, double* force);
    virtual void resolution(int line, double** w, double* d, double* force);
    virtual void store(int line, double* force, bool /*convergence*/);

protected:
    double _mu;
    double _W[6];
    std::deque<double>* _vec;
};

class UnilateralConstraintResolutionSticky : public UnilateralConstraintResolutionWithFriction
{
public:
    UnilateralConstraintResolutionSticky(double mu, double delta) : UnilateralConstraintResolutionWithFriction(mu), _delta(delta)  { nbLines=3; }
    virtual void resolution(int line, double** w, double* d, double* force);

protected:
    double _delta;
};

#endif // SOFA_DEV	

template<class DataTypes>
class UnilateralInteractionConstraint : public core::behavior::InteractionConstraint
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(UnilateralInteractionConstraint,DataTypes), core::behavior::InteractionConstraint);

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecConst VecConst;
    typedef typename DataTypes::SparseVecDeriv SparseVecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename core::behavior::MechanicalState<DataTypes> MechanicalState;

    typedef core::behavior::BaseConstraint::ConstraintBlockInfo ConstraintBlockInfo;
    typedef core::behavior::BaseConstraint::PersistentID PersistentID;
    typedef core::behavior::BaseConstraint::ConstCoord ConstCoord;
    typedef core::behavior::BaseConstraint::ConstDeriv ConstDeriv;
    typedef core::behavior::BaseConstraint::ConstArea ConstArea;

    typedef core::behavior::BaseConstraint::VecConstraintBlockInfo VecConstraintBlockInfo;
    typedef core::behavior::BaseConstraint::VecPersistentID VecPersistentID;
    typedef core::behavior::BaseConstraint::VecConstCoord VecConstCoord;
    typedef core::behavior::BaseConstraint::VecConstDeriv VecConstDeriv;
    typedef core::behavior::BaseConstraint::VecConstArea VecConstArea;

protected:
    MechanicalState* object1;
    MechanicalState* object2;

    struct Contact
    {
        int m1, m2;		///< the two extremities of the spring: masses m1 and m2
        Deriv norm;		///< contact normal, from m1 to m2
        Deriv t;		///< added for friction
        Deriv s;		///< added for friction
        Real dt;
        Real delta;		///< QP * normal - contact distance
        Real dfree;		///< QPfree * normal - contact distance
        Real dfree_t;   ///< QPfree * t
        Real dfree_s;   ///< QPfree * s
        unsigned int id;
        long contactId;
        PersistentID localId;
        double mu;		///< angle for friction

        // for visu
        Coord P, Q;
        Coord Pfree, Qfree;
    };

    sofa::helper::vector<Contact> contacts;
    Real epsilon;
    bool yetIntegrated;

    std::deque<double> prevForces;

public:

    unsigned int constraintId;

    UnilateralInteractionConstraint(MechanicalState* object1, MechanicalState* object2)
        : object1(object1), object2(object2), epsilon(Real(0.001)),yetIntegrated(false)
    {
    }

    UnilateralInteractionConstraint(MechanicalState* object)
        : object1(object), object2(object), epsilon(Real(0.001)),yetIntegrated(false)
    {
    }

    UnilateralInteractionConstraint()
        : object1(NULL), object2(NULL), epsilon(Real(0.001)),yetIntegrated(false)
    {
    }

    virtual ~UnilateralInteractionConstraint()
    {
    }

    MechanicalState* getObject1() { return object1; }
    MechanicalState* getObject2() { return object2; }
    core::behavior::BaseMechanicalState* getMechModel1() { return object1; }
    core::behavior::BaseMechanicalState* getMechModel2() { return object2; }

    void clear(int reserve = 0)
    {
        contacts.clear();
        if (reserve)
            contacts.reserve(reserve);
    }

    virtual void applyConstraint(unsigned int & /*contactId*/);

    virtual void addContact(double mu, Deriv norm, Coord P, Coord Q, Real contactDistance, int m1, int m2, Coord Pfree, Coord Qfree, long id=0, PersistentID localid=0);

    void addContact(double mu, Deriv norm, Coord P, Coord Q, Real contactDistance, int m1, int m2, long id=0, PersistentID localid=0)
    {
        addContact(mu, norm, P, Q, contactDistance, m1, m2, (*getObject2()->getXfree())[m2], (*getObject1()->getXfree())[m1], id, localid);
    }

    void addContact(double mu, Deriv norm, Real contactDistance, int m1, int m2, long id=0, PersistentID localid=0)
    {
        addContact(mu, norm, (*getObject2()->getX())[m2], (*getObject1()->getX())[m1], contactDistance, m1, m2, (*getObject2()->getXfree())[m2], (*getObject1()->getXfree())[m1], id, localid);
    }

    virtual void getConstraintValue(defaulttype::BaseVector *, bool freeMotion);

    virtual void getConstraintId(long* id, unsigned int &offset);
    virtual void getConstraintInfo(VecConstraintBlockInfo& blocks, VecPersistentID& ids, VecConstCoord& positions, VecConstDeriv& directions, VecConstArea& areas);

#ifdef SOFA_DEV
    virtual void getConstraintResolution(std::vector<core::behavior::ConstraintResolution*>& resTab, unsigned int& offset);
#endif
    // Previous Constraint Interface
    virtual void projectResponse() {}
    virtual void projectJacobianMatrix() {}
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
        return core::behavior::InteractionConstraint::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static void create(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        core::behavior::InteractionConstraint::create(obj, context, arg);
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

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const UnilateralInteractionConstraint<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }
    void draw();

    /// this constraint is NOT holonomic
    bool isHolonomic() {return false;}
};


#if defined(WIN32) && !defined(SOFA_COMPONENT_CONSTRAINT_UNILATERALINTERACTIONCONSTRAINT_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_CONSTRAINT_API UnilateralInteractionConstraint<defaulttype::Vec3dTypes>;
//extern template class SOFA_COMPONENT_CONSTRAINT_API UnilateralInteractionConstraint<defaulttype::Vec2dTypes>;
//extern template class SOFA_COMPONENT_CONSTRAINT_API UnilateralInteractionConstraint<defaulttype::Vec1dTypes>;
//extern template class SOFA_COMPONENT_CONSTRAINT_API UnilateralInteractionConstraint<defaulttype::Vec6dTypes>;
//extern template class SOFA_COMPONENT_CONSTRAINT_API UnilateralInteractionConstraint<defaulttype::Rigid3dTypes>;
//extern template class SOFA_COMPONENT_CONSTRAINT_API UnilateralInteractionConstraint<defaulttype::Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_CONSTRAINT_API UnilateralInteractionConstraint<defaulttype::Vec3fTypes>;
//extern template class SOFA_COMPONENT_CONSTRAINT_API UnilateralInteractionConstraint<defaulttype::Vec2fTypes>;
//extern template class SOFA_COMPONENT_CONSTRAINT_API UnilateralInteractionConstraint<defaulttype::Vec1fTypes>;
//extern template class SOFA_COMPONENT_CONSTRAINT_API UnilateralInteractionConstraint<defaulttype::Vec6fTypes>;
//extern template class SOFA_COMPONENT_CONSTRAINT_API UnilateralInteractionConstraint<defaulttype::Rigid3fTypes>;
//extern template class SOFA_COMPONENT_CONSTRAINT_API UnilateralInteractionConstraint<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace constraint

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONSTRAINT_UNILATERALINTERACTIONCONSTRAINT_H
