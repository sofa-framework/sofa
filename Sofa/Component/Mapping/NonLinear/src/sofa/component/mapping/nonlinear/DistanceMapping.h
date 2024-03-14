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

#include <sofa/core/Mapping.h>
#include <sofa/component/mapping/nonlinear/NonLinearMappingData.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>
#include <sofa/type/RGBAColor.h>

// The following header is included just for compatibility (see #3707). To remove in v23.12
#include <sofa/component/mapping/nonlinear/DistanceMultiMapping.h>


namespace sofa::component::mapping::nonlinear
{


/** Maps point positions to distances (in distance unit).
 * @tparam TIn parent point positions
 * @tparam TOut corresponds to a scalar value: distance between point pairs, minus a rest distance.
 * The pairs are given in an EdgeSetTopologyContainer in the same node.
 * If the rest lengths are not defined, they are set using the initial values.
 * If computeDistance is set to true, the rest lengths are set to 0.
 * (Changed class name on Feb. 4, 2014, previous name was ExtensionMapping)

 * @author Francois Faure
 */
template <class TIn, class TOut>
class DistanceMapping : public core::Mapping<TIn, TOut>, public NonLinearMappingData<true>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(DistanceMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename Out::Real Real;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef linearalgebra::EigenSparseMatrix<TIn,TOut>   SparseMatrixEigen;
    typedef linearalgebra::EigenSparseMatrix<TIn,TIn>    SparseKMatrixEigen;
    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
    typedef Data<InMatrixDeriv> InDataMatrixDeriv;
    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;
    enum {Nin = In::deriv_total_size, Nout = Out::deriv_total_size };
    typedef sofa::core::topology::BaseMeshTopology::SeqEdges SeqEdges;
    typedef type::Vec<In::spatial_dimensions,Real> Direction;


    Data<bool> f_computeDistance;                    ///< if 'computeDistance = true', then rest length of each element equal 0, otherwise rest length is the initial lenght of each of them
    Data<type::vector<Real>> f_restLengths;          ///< Rest lengths of the connections
    Data<Real> d_showObjectScale;                    ///< Scale for object display
    Data<sofa::type::RGBAColor> d_color;             ///< Color for object display. (default=[1.0,1.0,0.0,1.0])

    /// Link to be set to the topology container in the component graph. 
    SingleLink<DistanceMapping<TIn, TOut>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;



    void init() override;

    using Inherit::apply;

    void apply(const core::MechanicalParams *mparams, Data<OutVecCoord>& out, const Data<InVecCoord>& in) override;

    void applyJ(const core::MechanicalParams *mparams, Data<OutVecDeriv>& out, const Data<InVecDeriv>& in) override;

    void applyJT(const core::MechanicalParams *mparams, Data<InVecDeriv>& out, const Data<OutVecDeriv>& in) override;

    void applyJT(const core::ConstraintParams *cparams, Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in) override;

    void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentForce, core::ConstMultiVecDerivId  childForce ) override;

    const sofa::linearalgebra::BaseMatrix* getJ() override;
    virtual const type::vector<sofa::linearalgebra::BaseMatrix*>* getJs() override;

    void updateK( const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForce ) override;
    const linearalgebra::BaseMatrix* getK() override;
    void buildGeometricStiffnessMatrix(sofa::core::GeometricStiffnessMatrix* matrices) override;

    void draw(const core::visual::VisualParams* vparams) override;

protected:
    DistanceMapping();
    ~DistanceMapping() override;

    SparseMatrixEigen jacobian;                         ///< Jacobian of the mapping
    type::vector<linearalgebra::BaseMatrix*> baseMatrices;      ///< Jacobian of the mapping, in a vector
    SparseKMatrixEigen K;                               ///< Assembled geometric stiffness matrix
    type::vector<Direction> directions;                         ///< Unit vectors in the directions of the lines
    type::vector< Real > invlengths;                          ///< inverse of current distances. Null represents the infinity (null distance)

    /// r=b-a only for position (eventual rotation, affine transform... remains null)
    void computeCoordPositionDifference( Direction& r, const InCoord& a, const InCoord& b );
};





#if !defined(SOFA_COMPONENT_MAPPING_DistanceMapping_CPP)
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API DistanceMapping< defaulttype::Vec3Types, defaulttype::Vec1Types >;
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API DistanceMapping< defaulttype::Rigid3Types, defaulttype::Vec1Types >;
#endif

} // namespace sofa::component::mapping::nonlinear
