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
#include <sofa/component/mapping/linear/config.h>
#include <sofa/component/mapping/linear/LinearMapping.h>

#include <sofa/linearalgebra/EigenSparseMatrix.h>
#include <sofa/core/Mapping.h>
#include <sofa/core/Mapping.inl>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/type/vector.h>
#include <sofa/core/trait/DataTypes.h>


namespace sofa::component::mapping::linear
{

template <class TIn>
class DistanceToPlaneMapping : public LinearMapping<TIn, defaulttype::Vec1dTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(DistanceToPlaneMapping,TIn), SOFA_TEMPLATE2(LinearMapping,TIn, defaulttype::Vec1dTypes));
    typedef LinearMapping<TIn,  defaulttype::Vec1dTypes> Inherit;
    typedef  defaulttype::Vec1dTypes TOut;

    void init() override;

    void apply(const core::MechanicalParams *mparams, Data<VecCoord_t<TOut>>& out, const Data<VecCoord_t<TIn>>& in) override;

    void applyJ(const core::MechanicalParams *mparams, Data<VecDeriv_t<TOut>>& out, const Data<VecDeriv_t<TIn>>& in) override;

    void applyJT(const core::MechanicalParams *mparams, Data<VecDeriv_t<TIn>>& out, const Data<VecDeriv_t<TOut>>& in) override;

    void applyJT(const core::ConstraintParams *cparams, Data<MatrixDeriv_t<TIn>>& out, const Data<MatrixDeriv_t<TOut>>& in) override;

    const linearalgebra::BaseMatrix* getJ() override;

    void handleTopologyChange() override;


    Data<type::Vec<Deriv_t<TIn>::spatial_dimensions,typename Deriv_t<TIn>::value_type>> d_planeNormal; ///< Normal of the plane to compute the distance to
    Data<type::Vec<Coord_t<TIn>::spatial_dimensions,typename Coord_t<TIn>::value_type>> d_planePoint; ///< A point belonging to the plane

protected:

    DistanceToPlaneMapping();
    virtual ~DistanceToPlaneMapping() {};

    linearalgebra::EigenSparseMatrix<TIn, TOut> J;
};


}
