/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <LMConstraint/config.h>
#include <SofaConstraint/PrecomputedConstraintCorrection.h>


namespace sofa::component::constraintset
{

/**
 *  \brief Component computing constraint forces within a simulated body using the compliance method.
 */
template<class TDataTypes>
class PrecomputedLMConstraintCorrection : public sofa::component::constraintset::PrecomputedConstraintCorrection< TDataTypes >
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PrecomputedLMConstraintCorrection,TDataTypes), SOFA_TEMPLATE(sofa::component::constraintset::PrecomputedConstraintCorrection, TDataTypes));

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

    typedef sofa::core::behavior::ConstraintCorrection< TDataTypes > Inherit;

    typedef typename Coord::value_type Real;
    typedef sofa::defaulttype::MatNoInit<3, 3, Real> Transformation;

    Data<bool> m_rotations;
    Data<bool> m_restRotations;

    Data<bool> recompute; ///< if true, always recompute the compliance
	Data<double> debugViewFrameScale; ///< Scale on computed node's frame
	sofa::core::objectmodel::DataFileName f_fileCompliance; ///< Precomputed compliance matrix data file
	Data<std::string> fileDir; ///< If not empty, the compliance will be saved in this repertory
    
protected:
    PrecomputedLMConstraintCorrection(sofa::core::behavior::MechanicalState<DataTypes> *mm = nullptr){};

public:
    void bwdInit() override;
};





#if  !defined(LMCONSTRAINT_PRECOMPUTEDLMCONSTRAINTCORRECTION_CPP)
extern template class LMCONSTRAINT_API PrecomputedLMConstraintCorrection<defaulttype::Vec3Types>;
extern template class LMCONSTRAINT_API PrecomputedLMConstraintCorrection<defaulttype::Vec1Types>;
extern template class LMCONSTRAINT_API PrecomputedLMConstraintCorrection<defaulttype::Rigid3Types>;

#endif


} //namespace sofa::component::constraintset
