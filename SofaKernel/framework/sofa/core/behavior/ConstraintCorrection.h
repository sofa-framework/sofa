/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_CORE_BEHAVIOR_CONSTRAINTCORRECTION_H
#define SOFA_CORE_BEHAVIOR_CONSTRAINTCORRECTION_H

#include <sofa/core/behavior/BaseConstraintCorrection.h>
#include <sofa/core/behavior/MechanicalState.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa
{

namespace core
{

namespace behavior
{

/**
 * Component computing contact forces within a simulated body using the compliance method.
 */
template<class TDataTypes>
class ConstraintCorrection : public BaseConstraintCorrection
{
public:
    SOFA_ABSTRACT_CLASS(SOFA_TEMPLATE(ConstraintCorrection, TDataTypes), BaseConstraintCorrection);

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::MatrixDeriv::RowConstIterator MatrixDerivRowConstIterator;
    typedef typename DataTypes::MatrixDeriv::ColConstIterator MatrixDerivColConstIterator;
    typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename DataTypes::MatrixDeriv::ColIterator MatrixDerivColIterator;

protected:
    /// Default Constructor
    ConstraintCorrection(MechanicalState< DataTypes > *ms = NULL)
        : mstate(ms)
    {
    };

    /// Default Destructor
    virtual ~ConstraintCorrection()
    {
    };
public:
    void init();

    virtual void cleanup();

    virtual void addConstraintSolver(core::behavior::ConstraintSolver *s);
    virtual void removeConstraintSolver(core::behavior::ConstraintSolver *s);
private:
    std::list<core::behavior::ConstraintSolver*> constraintsolvers;


public:
    /// Compute motion correction from the constraint resolution (LCP) calculated force
    ///
    /// @param cparams
    /// @param x is the position result VecId
    /// @param v is the velocity result VecId
    /// @param f is the motion space force vector
    /// @param lambda is the constraint space force vector
    virtual void computeAndApplyMotionCorrection(const core::ConstraintParams * cparams, core::MultiVecCoordId x, core::MultiVecDerivId v, core::MultiVecDerivId f, const defaulttype::BaseVector * lambda);

    virtual void computeAndApplyMotionCorrection(const core::ConstraintParams * cparams, Data< VecCoord > &x, Data< VecDeriv > &v, Data< VecDeriv > &f, const defaulttype::BaseVector * lambda) = 0;

    /// Compute position correction from the constraint resolution (LCP) calculated force
    ///
    /// @param cparams
    /// @param x is the position result VecId
    /// @param f is the motion space force vector
    /// @param lambda is the constraint space force vector
    virtual void computeAndApplyPositionCorrection(const core::ConstraintParams * cparams, core::MultiVecCoordId x, core::MultiVecDerivId f, const defaulttype::BaseVector * lambda);

    virtual void computeAndApplyPositionCorrection(const core::ConstraintParams * cparams, Data< VecCoord > &x, Data< VecDeriv > &f, const defaulttype::BaseVector * lambda)  = 0;

    /// Compute velocity correction from the constraint resolution (LCP) calculated force
    ///
    /// @param cparams
    /// @param v is the velocity result VecId
    /// @param f is the motion space force vector
    /// @param lambda is the constraint space force vector
    virtual void computeAndApplyVelocityCorrection(const core::ConstraintParams * cparams, core::MultiVecDerivId v, core::MultiVecDerivId f, const defaulttype::BaseVector * lambda);

    virtual void computeAndApplyVelocityCorrection(const core::ConstraintParams * cparams, Data< VecDeriv > &v, Data< VecDeriv > &f, const defaulttype::BaseVector * lambda) = 0;

    /// Apply predictive constraint force
    ///
    /// @param cparams
    /// @param f is the motion space force vector
    /// @param lambda is the constraint space force vector
    virtual void applyPredictiveConstraintForce(const core::ConstraintParams * cparams, core::MultiVecDerivId f, const defaulttype::BaseVector *lambda);

    virtual void applyPredictiveConstraintForce(const core::ConstraintParams * /*cparams*/, Data< VecDeriv > &/*f*/, const defaulttype::BaseVector * /*lambda*/) {};

    /// Converts constraint force from the constraints space to the motion space and stores it in f vector
    ///
    /// @param f is the motion space force vector
    /// @param lambda is the constraint space force vector
    void setConstraintForceInMotionSpace(core::MultiVecDerivId f, const defaulttype::BaseVector * lambda);

    void setConstraintForceInMotionSpace(Data< VecDeriv > &f, const defaulttype::BaseVector * lambda);

    /// Converts constraint force from the constraints space to the motion space and accumulates it in f vector
    ///
    /// @param f is the motion space force vector
    /// @param lambda is the constraint space force vector
    void addConstraintForceInMotionSpace(core::MultiVecDerivId f, const defaulttype::BaseVector * lambda);

    void addConstraintForceInMotionSpace(Data< VecDeriv > &f, const defaulttype::BaseVector * lambda);

    /// Converts constraint force from the constraints space to the motion space and stores it in f vector
    ///
    /// @param f is the motion space force vector
    /// @param lambda is the constraint space force vector
    /// @param activeDofs stores constrained dofs indices
    void setConstraintForceInMotionSpace(core::MultiVecDerivId f, const defaulttype::BaseVector * lambda, std::list< int > &activeDofs);

    void setConstraintForceInMotionSpace(Data< VecDeriv > &f, const defaulttype::BaseVector * lambda, std::list< int > &activeDofs);

    /// Converts constraint force from the constraints space to the motion space and accumulates it in f vector
    ///
    /// @param f is the motion space force vector
    /// @param lambda is the constraint space force vector
    /// @param activeDofs stores constrained dofs indices
    void addConstraintForceInMotionSpace(core::MultiVecDerivId f, const defaulttype::BaseVector * lambda, std::list< int > &activeDofs);

    void addConstraintForceInMotionSpace(Data< VecDeriv > &f, const defaulttype::BaseVector * lambda, std::list< int > &activeDofs);


    /// Pre-construction check method called by ObjectFactory.
    template< class T >
    static bool canCreate(T*& obj, objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast< MechanicalState<DataTypes>* >(context->getMechanicalState()) == NULL)
            return false;

        return BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const ConstraintCorrection<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    MechanicalState<DataTypes> *getMState() const
    {
        return mstate;
    }

    void setMState(MechanicalState<DataTypes> *_mstate)
    {
        mstate = _mstate;
    }

protected:
    MechanicalState<DataTypes> *mstate;
};


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_CORE_BEHAVIOR_CONSTRAINTCORRECTION_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_CORE_API ConstraintCorrection< sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API ConstraintCorrection< sofa::defaulttype::Vec2dTypes >;
extern template class SOFA_CORE_API ConstraintCorrection< sofa::defaulttype::Vec1dTypes >;
extern template class SOFA_CORE_API ConstraintCorrection< sofa::defaulttype::Rigid3dTypes >;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_CORE_API ConstraintCorrection< sofa::defaulttype::Vec1fTypes >;
extern template class SOFA_CORE_API ConstraintCorrection< sofa::defaulttype::Vec2fTypes >;
extern template class SOFA_CORE_API ConstraintCorrection< sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API ConstraintCorrection< sofa::defaulttype::Rigid3fTypes >;
#endif
#endif

} // namespace behavior

} // namespace core

} // namespace sofa

#endif // SOFA_CORE_BEHAVIOR_CONSTRAINTCORRECTION_H
