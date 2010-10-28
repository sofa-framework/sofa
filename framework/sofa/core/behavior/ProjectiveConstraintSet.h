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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_BEHAVIOR_PROJECTIVECONSTRAINTSET_H
#define SOFA_CORE_BEHAVIOR_PROJECTIVECONSTRAINTSET_H

#include <sofa/core/core.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>

#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace core
{

namespace behavior
{

/**
 *  \brief Component computing constraints within a simulated body.
 *
 *  This class define the abstract API common to constraints using a given type
 *  of DOFs.
 *  A ProjectiveConstraintSet computes constraints applied to one simulated body given its
 *  current position and velocity.
 *
 */
template<class DataTypes>
class ProjectiveConstraintSet : public BaseProjectiveConstraintSet
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ProjectiveConstraintSet,DataTypes), BaseProjectiveConstraintSet);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;
#ifndef SOFA_DEPRECATE_OLD_API
    typedef typename MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename MatrixDeriv::RowType MatrixDerivRowType;
#endif

    ProjectiveConstraintSet(MechanicalState<DataTypes> *mm = NULL);

    virtual ~ProjectiveConstraintSet();

    Data<Real> endTime;  ///< Time when the constraint becomes inactive (-1 for infinitely active)
    virtual bool isActive() const; ///< if false, the constraint does nothing

    virtual void init();

    /// Retrieve the associated MechanicalState
    MechanicalState<DataTypes>* getMState() { return mstate; }

    /// @name Vector operations
    /// @{

    /// Project dx to constrained space (dx models an acceleration).
    ///
    /// This method retrieves the dxId vector from the MechanicalState and call
    /// the internal projectResponse(VecDeriv&) method implemented by
    /// the component.
    virtual void projectResponse(MultiVecDerivId dxId, const MechanicalParams* mparams);

    /// Project the L matrix of the Lagrange Multiplier equation system.
    ///
    /// This method retrieves the lines of the Jacobian Matrix from the MechanicalState and call
    /// the internal projectResponse(MatrixDeriv&) method implemented by
    /// the component.
    virtual void projectJacobianMatrix(MultiMatrixDerivId cId, const MechanicalParams* mparams);

    /// Project v to constrained space (v models a velocity).
    ///
    /// This method retrieves the vId vector from the MechanicalState and call
    /// the internal projectVelocity(VecDeriv&) method implemented by
    /// the component.
    virtual void projectVelocity(MultiVecDerivId vId, const MechanicalParams* mparams);

    /// Project x to constrained space (x models a position).
    ///
    /// This method retrieves the xId vector from the MechanicalState and call
    /// the internal projectPosition(VecCoord&) method implemented by
    /// the component.
    virtual void projectPosition(MultiVecCoordId xId, const MechanicalParams* mparams);



    /// Project dx to constrained space (dx models an acceleration).
    ///
    /// This method must be implemented by the component, and is usually called
    /// by the generic ProjectiveConstraintSet::projectResponse() method.
    virtual void projectResponse(DataVecDeriv& dx, const MechanicalParams* /*mparams*/)
#ifdef SOFA_DEPRECATE_OLD_API
        = 0;
#else
    {
        projectResponse(*dx.beginEdit());
        dx.endEdit();
    }
    /// @deprecated use instead projectResponse(DataVecDeriv& dx, const MechanicalParams* mparams)
    virtual void projectResponse(VecDeriv& /*dx*/) {}
#endif

    /// Project v to constrained space (v models a velocity).
    ///
    /// This method must be implemented by the component, and is usually called
    /// by the generic ProjectiveConstraintSet::projectVelocity() method.
    virtual void projectVelocity(DataVecDeriv& v, const MechanicalParams* /*mparams*/)
#ifdef SOFA_DEPRECATE_OLD_API
        = 0;
#else
    {
        projectVelocity(*v.beginEdit());
        v.endEdit();
    }
    /// @deprecated use instead projectResponse(DataVecDeriv& v, const MechanicalParams* mparams)
    virtual void projectVelocity(VecDeriv& /*v*/) {}
#endif

    /// Project x to constrained space (x models a position).
    ///
    /// This method must be implemented by the component, and is usually called
    /// by the generic ProjectiveConstraintSet::projectPosition() method.
    virtual void projectPosition(DataVecCoord& x, const MechanicalParams* /*mparams*/)
#ifdef SOFA_DEPRECATE_OLD_API
        = 0;
#else
    {
        projectPosition(*x.beginEdit());
        x.endEdit();
    }
    /// @deprecated use instead  projectResponse(DataVecCoord& x, const MechanicalParams* mparams)
    virtual void projectPosition(VecCoord& /*x*/) {}
#endif

    /// Project c to constrained space (c models a constraint).
    ///
    /// This method must be implemented by the component to handle Lagrange Multiplier based constraint
    virtual void projectJacobianMatrix(DataMatrixDeriv& cData, const MechanicalParams* /*mparams*/)
#ifdef SOFA_DEPRECATE_OLD_API
        = 0;
#else
    {
        helper::WriteAccessor<DataMatrixDeriv> c = cData;

        MatrixDerivRowIterator rowIt = c->begin();
        MatrixDerivRowIterator rowItEnd = c->end();

        while (rowIt != rowItEnd)
        {
            projectResponse(rowIt.row());
            ++rowIt;
        }
    }
    /// @deprecated use instead projectResponse(Data< MatrixDeriv >& c, const MechanicalParams* mparams)
    virtual void projectResponse(MatrixDerivRowType& /*c*/) {}
#endif

    /// @}


    /// Project the global Mechanical Matrix to constrained space using offset parameter
    /// @deprecated
    virtual void applyConstraint(defaulttype::BaseMatrix* /*matrix*/, unsigned int & /*offset*/)
    {
    }

    /// Project the global Mechanical Matrix to constrained space using offset parameter
    virtual void applyConstraint(const sofa::core::behavior::MultiMatrixAccessor* matrix, const MechanicalParams* /*mparams*/)
    {
        MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
        if (r)
            applyConstraint(r.matrix, r.offset);
    }

    /// Project the global Mechanical Vector to constrained space using offset parameter
    /// @deprecated
    virtual void applyConstraint(defaulttype::BaseVector* /*vector*/, unsigned int & /*offset*/)
    {
    }

    /// Project the global Mechanical Vector to constrained space using offset parameter
    virtual void applyConstraint(defaulttype::BaseVector* vector, const sofa::core::behavior::MultiMatrixAccessor* matrix, const MechanicalParams* /*mparams*/)
    {
        int o = matrix->getGlobalOffset(this->mstate);
        if (o >= 0)
        {
            unsigned int offset = (unsigned int)o;
            applyConstraint(vector, offset);
        }
    }

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const ProjectiveConstraintSet<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:
    MechanicalState<DataTypes> *mstate;
};

#if defined(WIN32) && !defined(SOFA_BUILD_CORE)
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Vec3dTypes >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Vec2dTypes >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Vec1dTypes >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Rigid3dTypes >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Rigid2dTypes >;

extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Vec3fTypes >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Vec2fTypes >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Vec1fTypes >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Rigid3fTypes >;
extern template class SOFA_CORE_API ProjectiveConstraintSet< defaulttype::Rigid2fTypes >;
#endif

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
