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

#include <sofa/core/MultiMapping.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/core/objectmodel/lifecycle/RenamedData.h>

namespace sofa::component::mapping::linear
{

template <class TIn, class TOut>
class SubsetMultiMapping : public LinearMultiMapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(SubsetMultiMapping, TIn, TOut), SOFA_TEMPLATE2(LinearMultiMapping, TIn, TOut));

    typedef LinearMultiMapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    /// Correspondence array
    typedef core::topology::BaseMeshTopology::SetIndex IndexArray;

    void init() override;

    // Append particle of given index within the given model to the subset.
    void addPoint(const core::BaseState* fromModel, int index);
    // Append particle of given index within the given model to the subset.
    void addPoint(int fromModel, int index);

    // usual Mapping API
    void apply(const core::MechanicalParams* mparams, const type::vector<DataVecCoord_t<Out>*>& dataVecOutPos, const type::vector<const DataVecCoord_t<In>*>& dataVecInPos) override;
    void applyJ(const core::MechanicalParams* mparams, const type::vector<DataVecDeriv_t<Out>*>& dataVecOutVel, const type::vector<const DataVecDeriv_t<In>*>& dataVecInVel) override;
    void applyJT(const core::MechanicalParams* mparams, const type::vector<DataVecDeriv_t<In>*>& dataVecOutForce, const type::vector<const DataVecDeriv_t<Out>*>& dataVecInForce) override;
    void applyDJT(const core::MechanicalParams* /*mparams*/, core::MultiVecDerivId /*inForce*/, core::ConstMultiVecDerivId /*outForce*/) override {}

    void applyJT( const core::ConstraintParams* cparams, const type::vector< DataMatrixDeriv_t<In>* >& dataMatOutConst, const type::vector< const DataMatrixDeriv_t<Out>* >& dataMatInConst ) override;

    /// Experimental API used to handle multimappings in matrix assembly. Returns pointers to matrices associated with parent states, consistently with  getFrom().
    virtual const type::vector<sofa::linearalgebra::BaseMatrix*>* getJs() override;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_MAPPING_LINEAR()
    sofa::core::objectmodel::lifecycle::RenamedData< type::vector<unsigned> > indexPairs;

    Data< type::vector<unsigned> > d_indexPairs; ///< list of couples (parent index + index in the parent)

protected :

    SubsetMultiMapping();
    virtual ~SubsetMultiMapping() override;

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
