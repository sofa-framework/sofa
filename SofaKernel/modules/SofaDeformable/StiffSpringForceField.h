/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_INTERACTIONFORCEFIELD_STIFFSPRINGFORCEFIELD_H
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_STIFFSPRINGFORCEFIELD_H
#include "config.h"

#include <SofaDeformable/SpringForceField.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/core/MechanicalParams.h>


namespace sofa
{

namespace component
{

namespace interactionforcefield
{

/** SpringForceField able to evaluate and apply its stiffness.
This allows to perform implicit integration.
Stiffness is evaluated and stored by the addForce method.
When explicit integration is used, SpringForceField is slightly more efficient.
*/

template<class DataTypes>
class StiffSpringForceField : public sofa::component::interactionforcefield::SpringForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(StiffSpringForceField,DataTypes), SOFA_TEMPLATE(SpringForceField,DataTypes));

    typedef SpringForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;

    typedef core::objectmodel::Data<VecDeriv>    DataVecDeriv;
    typedef core::objectmodel::Data<VecCoord>    DataVecCoord;


    typedef typename Inherit::Spring Spring;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalState;
    enum { N=DataTypes::spatial_dimensions };
    typedef defaulttype::Mat<N,N,Real> Mat;

protected:
    sofa::helper::vector<Mat>  dfdx;

    /// Accumulate the spring force and compute and store its stiffness
    virtual void addSpringForce(Real& potentialEnergy, VecDeriv& f1,const  VecCoord& p1,const VecDeriv& v1, VecDeriv& f2,const  VecCoord& p2,const  VecDeriv& v2, int i, const Spring& spring) override;

    /// Apply the stiffness, i.e. accumulate df given dx
    virtual void addSpringDForce(VecDeriv& df1,const  VecDeriv& dx1, VecDeriv& df2,const  VecDeriv& dx2, int i, const Spring& spring, double kFactor, double bFactor);


    StiffSpringForceField(MechanicalState* object1, MechanicalState* object2, double ks=100.0, double kd=5.0)
        : SpringForceField<DataTypes>(object1, object2, ks, kd)
    {
    }

    StiffSpringForceField(double ks=100.0, double kd=5.0)
        : SpringForceField<DataTypes>(ks, kd)
    {
    }
public:
    virtual void init() override;

    /// Accumulate f corresponding to x,v
    virtual void addForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2 ) override;
    ///SOFA_DEPRECATED_ForceField <<<virtual void addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2);

    /// Accumulate df corresponding to dx
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2) override;
    ///SOFA_DEPRECATED_ForceField <<<virtual void addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2, double kFactor, double bFactor);

    // getPotentialEnergy of base class SpringForceField.
   ///SOFA_DEPRECATED_ForceField <<<virtual void addKToMatrix(const sofa::core::behavior::MultiMatrixAccessor* matrix, double kFact);

    using Inherit::addKToMatrix;
    virtual void addKToMatrix(const sofa::core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix) override;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_STIFFSPRINGFORCEFIELD_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_DEFORMABLE_API StiffSpringForceField<defaulttype::Vec3dTypes>;
extern template class SOFA_DEFORMABLE_API StiffSpringForceField<defaulttype::Vec2dTypes>;
extern template class SOFA_DEFORMABLE_API StiffSpringForceField<defaulttype::Vec1dTypes>;
extern template class SOFA_DEFORMABLE_API StiffSpringForceField<defaulttype::Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_DEFORMABLE_API StiffSpringForceField<defaulttype::Vec3fTypes>;
extern template class SOFA_DEFORMABLE_API StiffSpringForceField<defaulttype::Vec2fTypes>;
extern template class SOFA_DEFORMABLE_API StiffSpringForceField<defaulttype::Vec1fTypes>;
extern template class SOFA_DEFORMABLE_API StiffSpringForceField<defaulttype::Vec6fTypes>;
#endif
#endif

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_INTERACTIONFORCEFIELD_STIFFSPRINGFORCEFIELD_H */
