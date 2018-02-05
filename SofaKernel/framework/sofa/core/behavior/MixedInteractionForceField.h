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
#ifndef SOFA_CORE_BEHAVIOR_MIXEDINTERACTIONFORCEFIELD_H
#define SOFA_CORE_BEHAVIOR_MIXEDINTERACTIONFORCEFIELD_H

#include <sofa/core/core.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/MechanicalParams.h>
namespace sofa
{

namespace core
{

namespace behavior
{

/**
 *  \brief Component computing forces between a pair of simulated body.
 *
 *  This class define the abstract API common to interaction force fields
 *  between a pair of bodies using a given type of DOFs.
 */
template<class TDataTypes1, class TDataTypes2>
class MixedInteractionForceField : public BaseInteractionForceField
{
public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE2(MixedInteractionForceField,TDataTypes1,TDataTypes2), BaseInteractionForceField);

    typedef TDataTypes1 DataTypes1;
    typedef typename DataTypes1::VecCoord VecCoord1;
    typedef typename DataTypes1::VecDeriv VecDeriv1;
    typedef typename DataTypes1::Coord    Coord1;
    typedef typename DataTypes1::Deriv    Deriv1;
    typedef typename DataTypes1::Real     Real1;
    typedef TDataTypes2 DataTypes2;
    typedef typename DataTypes2::VecCoord VecCoord2;
    typedef typename DataTypes2::VecDeriv VecDeriv2;
    typedef typename DataTypes2::Coord    Coord2;
    typedef typename DataTypes2::Deriv    Deriv2;
    typedef typename DataTypes2::Real     Real2;

    typedef core::objectmodel::Data<VecCoord1>    DataVecCoord1;
    typedef core::objectmodel::Data<VecDeriv1>    DataVecDeriv1;
    typedef core::objectmodel::Data<VecCoord2>    DataVecCoord2;
    typedef core::objectmodel::Data<VecDeriv2>    DataVecDeriv2;
protected:
    MixedInteractionForceField(MechanicalState<DataTypes1> *mm1 = NULL, MechanicalState<DataTypes2> *mm2 = NULL);

    virtual ~MixedInteractionForceField();
public:
    virtual void init() override;

    /// Retrieve the associated MechanicalState
    MechanicalState<DataTypes1>* getMState1() { return mstate1.get(); }
    BaseMechanicalState* getMechModel1() override { return mstate1.get(); }
    /// Retrieve the associated MechanicalState
    MechanicalState<DataTypes2>* getMState2() { return mstate2.get(); }
    BaseMechanicalState* getMechModel2() override { return mstate2.get(); }

    /// @name Vector operations
    /// @{

    /// Given the current position and velocity states, update the current force
    /// vector by computing and adding the forces associated with this
    /// ForceField.
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    /// $ f += B v + K x $
    ///
    /// This method retrieves the force, x and v vector from the two MechanicalState
    /// and call the internal addForce(VecDeriv&,VecDeriv&,const VecCoord&,const VecCoord&,const VecDeriv&,const VecDeriv&)
    /// method implemented by the component.
    virtual void addForce(const MechanicalParams* mparams, MultiVecDerivId fId ) override;

    /// Compute the force derivative given a small displacement from the
    /// position and velocity used in the previous call to addForce().
    ///
    /// The derivative should be directly derived from the computations
    /// done by addForce. Any forces neglected in addDForce will be integrated
    /// explicitly (i.e. using its value at the beginning of the timestep).
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    /// $ df += kFactor K dx + bFactor B dx $
    ///
    /// This method retrieves the force and dx vector from the two MechanicalState
    /// and call the internal addDForce(VecDeriv1&,VecDeriv2&,const VecDeriv1&,const VecDeriv2&,SReal,SReal)
    /// method implemented by the component.
    virtual void addDForce(const MechanicalParams* mparams, MultiVecDerivId dfId ) override;


    /// Get the potential energy associated to this ForceField.
    ///
    /// Used to extimate the total energy of the system by some
    /// post-stabilization techniques.
    ///
    /// This method retrieves the x vector from the MechanicalState and call
    /// the internal getPotentialEnergy(const VecCoord&,const VecCoord&) method implemented by
    /// the component.
    virtual SReal getPotentialEnergy(const MechanicalParams* mparams) const override;

    /// Given the current position and velocity states, update the current force
    /// vector by computing and adding the forces associated with this
    /// ForceField.
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    /// $ f += B v + K x $
    ///
    /// This method must be implemented by the component, and is usually called
    /// by the generic ForceField::addForce() method.

    virtual void addForce(const MechanicalParams* mparams, DataVecDeriv1& f1, DataVecDeriv2& f2, const DataVecCoord1& x1, const DataVecCoord2& x2, const DataVecDeriv1& v1, const DataVecDeriv2& v2)=0;

    /// Compute the force derivative given a small displacement from the
    /// position and velocity used in the previous call to addForce().
    ///
    /// The derivative should be directly derived from the computations
    /// done by addForce. Any forces neglected in addDForce will be integrated
    /// explicitly (i.e. using its value at the beginning of the timestep).
    ///
    /// If the ForceField can be represented as a matrix, this method computes
    /// $ df += K dx $
    ///
    /// This method must be implemented by the component, and is usually called
    /// by the generic MixedInteractionForceField::addDForce() method.

    virtual void addDForce(const MechanicalParams* mparams, DataVecDeriv1& df1, DataVecDeriv2& df2, const DataVecDeriv1& dx1, const DataVecDeriv2& dx2)=0;

    /// Get the potential energy associated to this ForceField.
    ///
    /// Used to extimate the total energy of the system by some
    /// post-stabilization techniques.
    ///
    /// This method must be implemented by the component, and is usually called
    /// by the generic MixedInteractionForceField::getPotentialEnergy() method.
    virtual SReal getPotentialEnergy(const MechanicalParams* mparams, const DataVecCoord1& x1, const DataVecCoord2& x2) const =0;

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {

        return BaseInteractionForceField::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T* /*p0*/, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        typename T::SPtr obj = sofa::core::objectmodel::New<T>();

        if (context)
            context->addObject(obj);

        if (arg)
        {
            obj->parse(arg);
        }

        return obj;
    }

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const MixedInteractionForceField<DataTypes1,DataTypes2>* = NULL)
    {
        return DataTypes1::Name()+std::string(",")+DataTypes2::Name();
    }

    template<class T>
    static std::string shortName(const T* ptr = NULL, objectmodel::BaseObjectDescription* arg = NULL)
    {
        std::string name = Inherit1::shortName(ptr, arg);
        sofa::helper::replaceAll(name, "InteractionForceField", "IFF");
        sofa::helper::replaceAll(name, "ForceField", "FF");
        return name;
    }


    /// Useful when the forcefield is applied only on a subset of dofs.
    /// It is automatically called by addForce.
    ///
    /// That way, we can optimize the time spent to transfer quantities through the mechanical mappings.
    /// Every Dofs are inserted by default. The forcefields using only a subset of dofs should only insert these dofs in the mask.
    virtual void updateForceMask() override;

protected:
    SingleLink<MixedInteractionForceField<DataTypes1,DataTypes2>, MechanicalState<DataTypes1>, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> mstate1;
    SingleLink<MixedInteractionForceField<DataTypes1,DataTypes2>, MechanicalState<DataTypes2>, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> mstate2;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_CORE_BEHAVIOR_MIXEDINTERACTIONFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec1dTypes, defaulttype::Vec3dTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec1dTypes, defaulttype::Rigid3dTypes>;

extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec3dTypes, defaulttype::Vec3dTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec2dTypes, defaulttype::Vec2dTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec1dTypes, defaulttype::Vec1dTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Rigid3dTypes, defaulttype::Rigid3dTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Rigid2dTypes, defaulttype::Rigid2dTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec3dTypes, defaulttype::Rigid3dTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec2dTypes, defaulttype::Rigid2dTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Rigid3dTypes, defaulttype::Vec3dTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Rigid2dTypes, defaulttype::Vec2dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec1fTypes, defaulttype::Vec3fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec1fTypes, defaulttype::Rigid3fTypes>;

extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec3fTypes, defaulttype::Vec3fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec2fTypes, defaulttype::Vec2fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec1fTypes, defaulttype::Vec1fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Rigid3fTypes, defaulttype::Rigid3fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Rigid2fTypes, defaulttype::Rigid2fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec3fTypes, defaulttype::Rigid3fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec2fTypes, defaulttype::Rigid2fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Rigid3fTypes, defaulttype::Vec3fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Rigid2fTypes, defaulttype::Vec2fTypes>;
#endif

#ifndef SOFA_DOUBLE
#ifndef SOFA_FLOAT
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec1dTypes, defaulttype::Vec3fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec1fTypes, defaulttype::Vec3dTypes>;

extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec1dTypes, defaulttype::Rigid3fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec1fTypes, defaulttype::Rigid3dTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec3dTypes, defaulttype::Vec3fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec2dTypes, defaulttype::Vec2fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec1dTypes, defaulttype::Vec1fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Rigid3dTypes, defaulttype::Rigid3fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Rigid2dTypes, defaulttype::Rigid2fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec3dTypes, defaulttype::Rigid3fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec2dTypes, defaulttype::Rigid2fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Rigid3dTypes, defaulttype::Vec3fTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Rigid2dTypes, defaulttype::Vec2fTypes>;

extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec3fTypes, defaulttype::Vec3dTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec2fTypes, defaulttype::Vec2dTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec1fTypes, defaulttype::Vec1dTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Rigid3fTypes, defaulttype::Rigid3dTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Rigid2fTypes, defaulttype::Rigid2dTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec3fTypes, defaulttype::Rigid3dTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Vec2fTypes, defaulttype::Rigid2dTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Rigid3fTypes, defaulttype::Vec3dTypes>;
extern template class SOFA_CORE_API MixedInteractionForceField<defaulttype::Rigid2fTypes, defaulttype::Vec2dTypes>;
#endif
#endif
#endif
} // namespace behavior

} // namespace core

} // namespace sofa

#endif
