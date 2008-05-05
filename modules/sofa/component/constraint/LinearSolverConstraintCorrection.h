/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_COMPONENTMODEL_COLLISION_LINEARSOLVERCONTACTCORRECTION_H
#define SOFA_CORE_COMPONENTMODEL_COLLISION_LINEARSOLVERCONTACTCORRECTION_H

#include <sofa/core/componentmodel/behavior/BaseConstraintCorrection.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/core/componentmodel/behavior/LinearSolver.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>

namespace sofa
{

namespace component
{

namespace constraint
{

using namespace sofa::core;
using namespace sofa::core::componentmodel;
using namespace sofa::defaulttype;



/// to avoid compilation problem under gcc3.3
extern inline behavior::OdeSolver* getOdeSolver(objectmodel::BaseContext* context)
{
    return context->get<behavior::OdeSolver>();
}
extern inline behavior::LinearSolver* getLinearSolver(objectmodel::BaseContext* context)
{
    return context->get<behavior::LinearSolver>();
}


/**
 *  \brief Component computing contact forces within a simulated body using the compliance method.
 */
template<class TDataTypes>
class LinearSolverConstraintCorrection : public componentmodel::behavior::BaseConstraintCorrection
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real Real;
    typedef typename componentmodel::behavior::BaseConstraintCorrection Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::VecConst VecConst;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::SparseVecDeriv Const;

    LinearSolverConstraintCorrection(behavior::MechanicalState<DataTypes> *mm = NULL);

    virtual ~LinearSolverConstraintCorrection();

    virtual void init();

    /// Retrieve the associated MechanicalState
    behavior::MechanicalState<DataTypes>* getMState() { return mstate; }

    virtual void getCompliance(defaulttype::BaseMatrix* W);
    virtual void applyContactForce(const defaulttype::BaseVector *f);
    virtual void resetContactForce();

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        if( getOdeSolver(context)==NULL )
            return false;
        if( getLinearSolver(context)==NULL )
            return false;
//         if (context->get<behavior::OdeSolver>() == NULL)
//             return false;
// 		if (context->get<behavior::LinearSolver>() == NULL)
//             return false;
        return BaseObject::canCreate(obj, context, arg);
    }



    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const LinearSolverConstraintCorrection<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:
    behavior::MechanicalState<DataTypes> *mstate;
    behavior::OdeSolver* odesolver;
    behavior::LinearSolver* linearsolver;

    linearsolver::SparseMatrix<SReal> J; ///< constraint matrix
    linearsolver::FullVector<SReal> F; ///< forces computed from the constraints
    linearsolver::FullMatrix<SReal> refMinv; ///< reference inverse matrix
};



} // namespace collision

} // namespace component

} // namespace sofa

#endif
