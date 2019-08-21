/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_CONSTRAINTSET_SLIDINGCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINTSET_SLIDINGCONSTRAINT_H
#include "config.h"

#include <sofa/core/behavior/PairInteractionConstraint.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <iostream>

namespace sofa
{

namespace component
{

namespace constraintset
{

using sofa::core::ConstraintParams;

template<class DataTypes>
class SlidingConstraint : public core::behavior::PairInteractionConstraint<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SlidingConstraint,DataTypes), SOFA_TEMPLATE(core::behavior::PairInteractionConstraint,DataTypes));

    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename core::behavior::MechanicalState<DataTypes> MechanicalState;
    typedef typename core::behavior::PairInteractionConstraint<DataTypes> Inherit;

    typedef core::objectmodel::Data<VecCoord>		DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv>		DataVecDeriv;
    typedef core::objectmodel::Data<MatrixDeriv>    DataMatrixDeriv;

protected:

    Data<int> d_m1; ///< index of the spliding point on the first model
    Data<int> d_m2a; ///< index of one end of the sliding axis
    Data<int> d_m2b; ///< index of the other end of the sliding axis
    Data<Deriv> d_force; ///< interaction force

    Real m_dist;	// constraint violation
    Real m_thirdConstraint; // 0 if A<proj<B, -1 if proj<A, 1 if B<proj
    bool m_yetIntegrated;
    unsigned int m_cid;


    SlidingConstraint();
    SlidingConstraint(MechanicalState* object);
    SlidingConstraint(MechanicalState* object1, MechanicalState* object2);

    virtual ~SlidingConstraint(){}

public:
    void init() override;

    void buildConstraintMatrix(const core::ConstraintParams* cParams, DataMatrixDeriv &c1, DataMatrixDeriv &c2, unsigned int &cIndex
            , const DataVecCoord &x1, const DataVecCoord &x2) override;

    void getConstraintViolation(const core::ConstraintParams* cParams, defaulttype::BaseVector *v, const DataVecCoord &x1, const DataVecCoord &x2
            , const DataVecDeriv &v1, const DataVecDeriv &v2) override;

    void getConstraintResolution(const core::ConstraintParams*,
                                         std::vector<core::behavior::ConstraintResolution*>& resTab,
                                         unsigned int& offset) override;
    void storeLambda(const ConstraintParams* cParams, sofa::core::MultiVecDerivId res, const sofa::defaulttype::BaseVector* lambda) override;

    void draw(const core::visual::VisualParams* vparams) override;

private:
    // storage of force
    Deriv  m_dirAxe, m_dirProj, m_dirOrtho;



};

#if  !defined(SOFA_COMPONENT_CONSTRAINTSET_SLIDINGCONSTRAINT_CPP)
extern template class SOFA_CONSTRAINT_API SlidingConstraint< defaulttype::Vec3Types >;

#endif

} // namespace constraintset

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONSTRAINTSET_SLIDINGCONSTRAINT_H
