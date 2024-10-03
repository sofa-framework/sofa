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

#include <sofa/component/mapping/nonlinear/config.h>

#include <sofa/component/mapping/nonlinear/BaseNonLinearMapping.h>
#include <sofa/component/mapping/nonlinear/NonLinearMappingData.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>
#include <sofa/type/RGBAColor.h>


namespace sofa::component::mapping::nonlinear
{

/** Maps point positions to square distances.
  Type TOut corresponds to a scalar value.
  The pairs are given in an EdgeSetTopologyContainer in the same node.

    In: parent point positions
    Out: square distance between point pairs, minus a square rest distance.

    No restLength (imposed null rest length) for now
    TODO: compute Jacobians for non null restLength

@author Matthieu Nesme
  */


// If the rest lengths are not defined, they are set using the initial values.
// If computeDistance is set to true, the rest lengths are set to 0.
template <class TIn, class TOut>
class SquareDistanceMapping : public BaseNonLinearMapping<TIn, TOut, true>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(SquareDistanceMapping,TIn,TOut), SOFA_TEMPLATE3(BaseNonLinearMapping,TIn,TOut,true));

    using In = TIn;
    using Out = TOut;

    using Real = Real_t<Out>;

    static constexpr auto Nin = In::deriv_total_size;

    typedef sofa::core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef type::Vec<In::spatial_dimensions,Real> Direction;

    Data<Real> d_showObjectScale; ///< Scale for object display
    Data<sofa::type::RGBAColor> d_color; ///< Color for object display. (default=[1.0,1.0,0.0,1.0])

    /// Link to be set to the topology container in the component graph. 
    SingleLink<SquareDistanceMapping<TIn, TOut>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    void init() override;

    void apply(const core::MechanicalParams *mparams, DataVecCoord_t<Out>& out, const DataVecCoord_t<In>& in) override;
    void buildGeometricStiffnessMatrix(sofa::core::GeometricStiffnessMatrix* matrices) override;

    void draw(const core::visual::VisualParams* vparams) override;

protected:
    SquareDistanceMapping();
    ~SquareDistanceMapping() override;

    void matrixFreeApplyDJT(const core::MechanicalParams* mparams, Real kFactor,
                            Data<VecDeriv_t<In> >& parentForce,
                            const Data<VecDeriv_t<In> >& parentDisplacement,
                            const Data<VecDeriv_t<Out> >& childForce) override;

    using typename Inherit1::SparseKMatrixEigen;

    void doUpdateK(
        const core::MechanicalParams* mparams, const Data<VecDeriv_t<Out> >& childForce,
        SparseKMatrixEigen& matrix) override;

    /// r=b-a only for position (eventual rotation, affine transform... remains null)
    void computeCoordPositionDifference( Direction& r, const Coord_t<In>& a, const Coord_t<In>& b );
};




#if !defined(SOFA_COMPONENT_MAPPING_SquareDistanceMapping_CPP)
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API SquareDistanceMapping< defaulttype::Vec3Types, defaulttype::Vec1Types >;
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API SquareDistanceMapping< defaulttype::Rigid3Types, defaulttype::Vec1Types >;
#endif

} // namespace sofa::component::mapping::nonlinear
