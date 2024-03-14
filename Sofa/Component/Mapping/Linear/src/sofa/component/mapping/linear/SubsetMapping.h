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

#include <sofa/core/topology/TopologySubsetData.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>

#include <sofa/core/Mapping.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/vector.h>
#include <sofa/type/trait/Rebind.h>


namespace sofa::component::mapping::linear
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class InDataTypes, class OutDataTypes>
class SubsetMappingInternalData
{
public:
};

/**
 * @class SubsetMapping
 * @brief Compute a subset of input points
 */
template <class TIn, class TOut>
class SubsetMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(SubsetMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    typedef typename In::Real         Real;
    typedef typename In::VecCoord     InVecCoord;
    typedef typename In::VecDeriv     InVecDeriv;
    typedef typename In::MatrixDeriv  InMatrixDeriv;
    typedef Data<InVecCoord>          InDataVecCoord;
    typedef Data<InVecDeriv>          InDataVecDeriv;
    typedef Data<InMatrixDeriv>       InDataMatrixDeriv;
    typedef typename In::Coord        InCoord;
    typedef typename In::Deriv        InDeriv;

    typedef typename Out::VecCoord    OutVecCoord;
    typedef typename Out::VecDeriv    OutVecDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef Data<OutVecCoord>         OutDataVecCoord;
    typedef Data<OutVecDeriv>         OutDataVecDeriv;
    typedef Data<OutMatrixDeriv>      OutDataMatrixDeriv;
    typedef typename Out::Coord       OutCoord;
    typedef typename Out::Deriv       OutDeriv;

    enum { NIn = sofa::defaulttype::DataTypeInfo<InDeriv>::Size };
    enum { NOut = sofa::defaulttype::DataTypeInfo<OutDeriv>::Size };
    typedef type::Mat<NOut, NIn, Real> MBloc;
    typedef sofa::linearalgebra::CompressedRowSparseMatrix<MBloc> MatrixType;

    /// Correspondance array
    using IndexArray = sofa::type::rebind_to<InVecCoord, Index>;
    typedef sofa::core::topology::PointSubsetData< IndexArray > SetIndex;
    SetIndex f_indices;

    Data < Index > f_first; ///< first index (use if indices are sequential)
    Data < Index > f_last; ///< last index (use if indices are sequential)
    Data < Real > f_radius; ///< search radius to find corresponding points in case no indices are given
    Data < bool > f_handleTopologyChange; ///< Enable support of topological changes for indices (disable if it is linked from SubsetTopologicalMapping::pointD2S)
    Data < bool > f_ignoreNotFound; ///< True to ignore points that are not found in the input model, they will be treated as fixed points
    Data < bool > f_resizeToModel; ///< True to resize the output MechanicalState to match the size of indices
    SubsetMappingInternalData<In, Out> data;
    void postInit();
    /// Link to be set to the topology container in the component graph. 
    SingleLink<SubsetMapping<In, Out>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected:
    SubsetMapping();
public:
    void clear(Size reserve);

    int addPoint(Index index);

    void init() override;

    virtual ~SubsetMapping();

    void apply ( const core::MechanicalParams* mparams, OutDataVecCoord& dOut, const InDataVecCoord& dIn ) override;

    void applyJ( const core::MechanicalParams* mparams, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn ) override;

    void applyJT ( const core::MechanicalParams* mparams, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn ) override;

    void applyJT ( const core::ConstraintParams* /*cparams*/, InDataMatrixDeriv& dOut, const OutDataMatrixDeriv& dIn) override;

    const sofa::linearalgebra::BaseMatrix* getJ() override;

public:
    typedef type::vector< linearalgebra::BaseMatrix* > js_type;
    const js_type* getJs() override;

protected:
    typedef linearalgebra::EigenSparseMatrix<In, Out> eigen_type;
    eigen_type eigen;
    js_type js;
public:

protected:
    std::unique_ptr<MatrixType> matrixJ;
    bool updateJ;
};

#if !defined(SOFA_COMPONENT_MAPPING_SUBSETMAPPING_CPP)

extern template class SOFA_COMPONENT_MAPPING_LINEAR_API SubsetMapping< sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types >;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API SubsetMapping< sofa::defaulttype::Vec1Types, sofa::defaulttype::Vec1Types >;
extern template class SOFA_COMPONENT_MAPPING_LINEAR_API SubsetMapping< sofa::defaulttype::Rigid3Types, sofa::defaulttype::Rigid3Types >;

#endif

} // namespace sofa::component::mapping::linear
