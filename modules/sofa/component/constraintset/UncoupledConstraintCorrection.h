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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_UNCOUPLEDCONSTRAINTCORRECTION_H
#define SOFA_COMPONENT_CONSTRAINTSET_UNCOUPLEDCONSTRAINTCORRECTION_H

#include <sofa/core/behavior/BaseConstraintCorrection.h>
#include <sofa/core/behavior/MechanicalState.h>

#include <sofa/component/component.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa
{

namespace component
{

namespace constraintset
{

using namespace sofa::core;
/**
 *  \brief Component computing contact forces within a simulated body using the compliance method.
 */
template<class TDataTypes>
class UncoupledConstraintCorrection : public behavior::BaseConstraintCorrection
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(UncoupledConstraintCorrection,TDataTypes), core::behavior::BaseConstraintCorrection);

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::MatrixDeriv::RowConstIterator MatrixDerivRowConstIterator;
    typedef typename DataTypes::MatrixDeriv::ColConstIterator MatrixDerivColConstIterator;
    typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename DataTypes::MatrixDeriv::ColIterator MatrixDerivColIterator;

    typedef typename Coord::value_type Real;
    typedef helper::vector<Real> VecReal;

    UncoupledConstraintCorrection(behavior::MechanicalState<DataTypes> *mm = NULL);

    virtual ~UncoupledConstraintCorrection();

    virtual void init();

    /// Handle Topological Changes.
    void handleTopologyChange();

    /// Retrieve the associated MechanicalState
    behavior::MechanicalState<DataTypes>* getMState() { return mstate; }

    virtual void getCompliance(defaulttype::BaseMatrix *W);
    virtual void getComplianceMatrix(defaulttype::BaseMatrix* ) const;

    // for multigrid approach => constraints are merged
    virtual void  getComplianceWithConstraintMerge(defaulttype::BaseMatrix* Wmerged, std::vector<int> &constraint_merge);

    virtual void applyContactForce(const defaulttype::BaseVector *f);

    virtual void applyPredictiveConstraintForce(const defaulttype::BaseVector *f);

    virtual void resetContactForce();


    // new API for non building the constraint system during solving process //

    virtual bool hasConstraintNumber(int index) ;  // virtual ???

    virtual void resetForUnbuiltResolution(double * f, std::list<int>& /*renumbering*/)  ;

    virtual void addConstraintDisplacement(double *d, int begin,int end) ;

    virtual void setConstraintDForce(double *df, int begin, int end, bool update) ;

    virtual void getBlockDiagonalCompliance(defaulttype::BaseMatrix* W, int begin, int end) ;
    /////////////////////////////////////////////////////////////////////////////////


    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const UncoupledConstraintCorrection<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:
    behavior::MechanicalState<DataTypes> *mstate;

public:
    Data< VecReal > compliance;

private:
    // new :  for non building the constraint system during solving process //
    VecDeriv constraint_disp, constraint_force;
    std::list<int> constraint_dofs;		// list of indices of each point which is involve with constraint

    //std::vector< std::vector<int> >  dof_constraint_table;   // table of indices of each point involved with each constraint
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_CONSTRAINTSET_UNCOUPLEDCONSTRAINTCORRECTION_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection<defaulttype::Vec3dTypes>;
//extern template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection<defaulttype::Vec2dTypes>;
extern template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection<defaulttype::Vec1dTypes>;
//extern template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection<defaulttype::Vec6dTypes>;
extern template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection<defaulttype::Rigid3dTypes>;
//extern template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection<defaulttype::Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection<defaulttype::Vec3fTypes>;
//extern template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection<defaulttype::Vec2fTypes>;
extern template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection<defaulttype::Vec1fTypes>;
//extern template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection<defaulttype::Vec6fTypes>;
extern template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection<defaulttype::Rigid3fTypes>;
//extern template class SOFA_COMPONENT_CONSTRAINTSET_API UncoupledConstraintCorrection<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif
