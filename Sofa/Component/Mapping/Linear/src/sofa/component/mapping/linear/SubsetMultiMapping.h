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

#include <sofa/core/MultiMapping.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::mapping::linear
{

/**
 * @class SubsetMapping
 * @brief Compute a subset of input points
 */
template <class TIn, class TOut>
class SubsetMultiMapping : public core::MultiMapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(SubsetMultiMapping, TIn, TOut), SOFA_TEMPLATE2(core::MultiMapping, TIn, TOut));

    typedef core::MultiMapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename InCoord::value_type Real;
    typedef typename type::vector<const InVecCoord*> vecConstInVecCoord;
    typedef typename type::vector<OutVecCoord*> vecOutVecCoord;

    typedef Data<InVecCoord> InDataVecCoord;
    typedef Data<InVecDeriv> InDataVecDeriv;
    typedef Data<InMatrixDeriv> InDataMatrixDeriv;

    typedef Data<OutVecCoord> OutDataVecCoord;
    typedef Data<OutVecDeriv> OutDataVecDeriv;
    typedef Data<OutMatrixDeriv> OutDataMatrixDeriv;

    /// Correspondance array
    typedef core::topology::BaseMeshTopology::SetIndex IndexArray;

    void init() override;

    // Append particle of given index within the given model to the subset.
    void addPoint(const core::BaseState* fromModel, int index);
    // Append particle of given index within the given model to the subset.
    void addPoint(int fromModel, int index);

    // usual Mapping API
    void apply(const core::MechanicalParams* mparams, const type::vector<OutDataVecCoord*>& dataVecOutPos, const type::vector<const InDataVecCoord*>& dataVecInPos) override;
    void applyJ(const core::MechanicalParams* mparams, const type::vector<OutDataVecDeriv*>& dataVecOutVel, const type::vector<const InDataVecDeriv*>& dataVecInVel) override;
    void applyJT(const core::MechanicalParams* mparams, const type::vector<InDataVecDeriv*>& dataVecOutForce, const type::vector<const OutDataVecDeriv*>& dataVecInForce) override;
    void applyDJT(const core::MechanicalParams* /*mparams*/, core::MultiVecDerivId /*inForce*/, core::ConstMultiVecDerivId /*outForce*/) override {}

    void applyJT( const core::ConstraintParams* cparams, const type::vector< InDataMatrixDeriv* >& dataMatOutConst, const type::vector< const OutDataMatrixDeriv* >& dataMatInConst ) override;

    /// Experimental API used to handle multimappings in matrix assembly. Returns pointers to matrices associated with parent states, consistently with  getFrom().
    virtual const type::vector<sofa::linearalgebra::BaseMatrix*>* getJs() override;


    Data< type::vector<unsigned> > indexPairs;                     ///< Two indices per child: the parent, and the index within the parent

protected :

    SubsetMultiMapping();
    virtual ~SubsetMultiMapping();

    type::vector<linearalgebra::BaseMatrix*> baseMatrices;      ///< Jacobian of the mapping, in a vector

};


#if !defined(SOFA_COMPONENT_MAPPING_SUBSETMULTIMAPPING_CPP)
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API SubsetMultiMapping< defaulttype::Vec3Types, defaulttype::Vec3Types >;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API SubsetMultiMapping< defaulttype::Vec2Types, defaulttype::Vec2Types >;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API SubsetMultiMapping< defaulttype::Vec1Types, defaulttype::Vec1Types >;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API SubsetMultiMapping< defaulttype::Rigid3Types, defaulttype::Rigid3Types >;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API SubsetMultiMapping< defaulttype::Rigid3Types, defaulttype::Vec3Types >;

#endif

} // namespace sofa::component::mapping::linear
