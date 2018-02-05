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
#ifndef SOFA_CORE_COLLISION_GENERICCONTACTCORRECTION_H
#define SOFA_CORE_COLLISION_GENERICCONTACTCORRECTION_H
#include "config.h"

#include <sofa/core/behavior/BaseConstraintCorrection.h>

namespace sofa
{

    namespace core { namespace behavior { class OdeSolver; class LinearSolver; } }


namespace component
{

namespace constraintset
{

class GenericConstraintCorrection : public core::behavior::BaseConstraintCorrection
{
public:
    SOFA_CLASS(GenericConstraintCorrection, core::behavior::BaseConstraintCorrection);

protected:
    GenericConstraintCorrection();
    virtual ~GenericConstraintCorrection();

public:
    virtual void bwdInit() override;

    virtual void cleanup() override;

    virtual void addConstraintSolver(core::behavior::ConstraintSolver *s) override;
    virtual void removeConstraintSolver(core::behavior::ConstraintSolver *s) override;
private:
    std::list<core::behavior::ConstraintSolver*> constraintsolvers;

public:
    virtual void addComplianceInConstraintSpace(const core::ConstraintParams *cparams, defaulttype::BaseMatrix* W) override;

    virtual void getComplianceMatrix(defaulttype::BaseMatrix* ) const override;

    virtual void computeAndApplyMotionCorrection(const core::ConstraintParams *cparams, core::MultiVecCoordId x, core::MultiVecDerivId v, core::MultiVecDerivId f, const defaulttype::BaseVector * lambda) override;

    virtual void computeAndApplyPositionCorrection(const core::ConstraintParams *cparams, core::MultiVecCoordId xId, core::MultiVecDerivId fId, const defaulttype::BaseVector *lambda) override;

    virtual void computeAndApplyVelocityCorrection(const core::ConstraintParams *cparams, core::MultiVecDerivId vId, core::MultiVecDerivId fId, const defaulttype::BaseVector *lambda) override;

    virtual void applyPredictiveConstraintForce(const core::ConstraintParams *cparams, core::MultiVecDerivId f, const defaulttype::BaseVector *lambda) override;

    virtual void rebuildSystem(double massFactor, double forceFactor) override;

    virtual void applyContactForce(const defaulttype::BaseVector *f) override;

    virtual void resetContactForce() override;

    virtual void computeResidual(const core::ExecParams* params, defaulttype::BaseVector *lambda) override;

    Data< helper::vector< std::string > >  d_linearSolversName;
    Data< std::string >                    d_ODESolverName;

    /// Pre-construction check method called by ObjectFactory.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        return BaseConstraintCorrection::canCreate(obj, context, arg);
    }

protected:

    core::behavior::OdeSolver* m_ODESolver;
    std::vector< core::behavior::LinearSolver* > m_linearSolvers;
};

} // namespace collision

} // namespace component

} // namespace sofa

#endif // SOFA_CORE_COLLISION_GENERICCONTACTCORRECTION_H
