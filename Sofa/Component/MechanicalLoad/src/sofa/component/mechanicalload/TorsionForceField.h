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
#include <sofa/component/mechanicalload/config.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/Vec.h>
#include <sofa/type/Mat.h>
#include <sofa/type/vector.h>

namespace sofa::component::mechanicalload
{

using sofa::type::Vec;
using sofa::type::Mat;
using sofa::core::behavior::ForceField;
using sofa::core::MechanicalParams;

using sofa::defaulttype::Vec3Types;
using sofa::defaulttype::Rigid3Types;

///
///	\brief TorsionForceField
///
///	This forcefield applies a torque to a set of selected nodes. The force is applied from a specified axis (origin and direction)
///	and a torque value.
///

template<typename DataTypes>
class TorsionForceField : public ForceField<DataTypes>
{
public:
	SOFA_CLASS(SOFA_TEMPLATE(TorsionForceField,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ForceField,DataTypes));

	typedef ForceField<DataTypes> Inherit;
	typedef typename DataTypes::Real Real;
	typedef typename DataTypes::Coord Coord;
	typedef typename DataTypes::Deriv Deriv;
	typedef typename DataTypes::VecCoord VecCoord;
	typedef typename DataTypes::VecDeriv VecDeriv;
	typedef typename DataTypes::CPos Pos;
//	typedef TorsionForceFieldUtility<DataTypes> FFUtil;
	typedef Data<VecCoord> DataVecCoord;
	typedef Data<VecDeriv> DataVecDeriv;

	typedef unsigned int PointId;
	typedef type::vector<PointId> VecId;
	typedef Mat<3, 3, Real> Mat3;

public:
	TorsionForceField();
	virtual ~TorsionForceField();

	void addForce(const MechanicalParams *, DataVecDeriv &f, const DataVecCoord &x, const DataVecDeriv &v) override;
	void addDForce(const MechanicalParams *mparams, DataVecDeriv &df, const DataVecDeriv &dx) override;
	void addKToMatrix(linearalgebra::BaseMatrix *matrix, SReal kFact, unsigned int &offset) override;
    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;
	void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    SReal getPotentialEnergy(const core::MechanicalParams* /*mparams*/, const DataVecCoord&  /* x */) const override;

public :
	Data<VecId> m_indices;		///< indices of the selected nodes.
	Data<Real> m_torque;		///< torque to be applied.
	Data<Pos> m_axis;			///< direction of the axis.
	Data<Pos> m_origin;			///< origin of the axis.

protected :
	Pos m_u;					///< normalized axis
};

template<>
void TorsionForceField<Rigid3Types>::addForce(const core::MechanicalParams *, DataVecDeriv &f, const DataVecCoord &x, const DataVecDeriv &v);

template<>
void TorsionForceField<Rigid3Types>::addDForce(const core::MechanicalParams *mparams, DataVecDeriv &df, const DataVecDeriv &dx);


#if !defined(SOFA_COMPONENT_FORCEFIELD_TORSIONFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_MECHANICALLOAD_API TorsionForceField<Vec3Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API TorsionForceField<Rigid3Types>;

#endif

} // namespace sofa::component::mechanicalload
