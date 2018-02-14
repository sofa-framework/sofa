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
#ifndef SOFA_COMPONENT_MAPPING_SUBSETMAPPING_H
#define SOFA_COMPONENT_MAPPING_SUBSETMAPPING_H
#include "config.h"


#include <sofa/core/Mapping.h>

#include <SofaBaseTopology/TopologySubsetData.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>

#include <sofa/helper/vector.h>
#include <sofa/core/topology/BaseMeshTopology.h>

#include <memory>

#include <SofaEigen2Solver/EigenSparseMatrix.h>

namespace sofa
{

namespace component
{

namespace mapping
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
    typedef defaulttype::Mat<NOut, NIn, Real> MBloc;
    typedef sofa::component::linearsolver::CompressedRowSparseMatrix<MBloc> MatrixType;

    /// Correspondance array
    typedef typename InVecCoord::template rebind<unsigned int>::other IndexArray;
    typedef sofa::component::topology::PointSubsetData< IndexArray > SetIndex;
    SetIndex f_indices;

    Data < int > f_first; ///< first index (use if indices are sequential)
    Data < int > f_last; ///< last index (use if indices are sequential)
    Data < Real > f_radius; ///< search radius to find corresponding points in case no indices are given
    Data < bool > f_handleTopologyChange; ///< Enable support of topological changes for indices (disable if it is linked from SubsetTopologicalMapping::pointD2S)
    Data < bool > f_ignoreNotFound; ///< True to ignore points that are not found in the input model, they will be treated as fixed points
    Data < bool > f_resizeToModel; ///< True to resize the output MechanicalState to match the size of indices
    SubsetMappingInternalData<In, Out> data;
    void postInit();
protected:
    SubsetMapping();
public:
    void clear(int reserve);

    int addPoint(int index);

    void init() override;

    // handle topology changes depending on the topology
    //void handleTopologyChange(core::topology::Topology* t);

    virtual ~SubsetMapping();

    virtual void apply ( const core::MechanicalParams* mparams, OutDataVecCoord& dOut, const InDataVecCoord& dIn ) override;

    virtual void applyJ( const core::MechanicalParams* mparams, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn ) override;

    virtual void applyJT ( const core::MechanicalParams* mparams, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn ) override;

    virtual void applyJT ( const core::ConstraintParams* /*cparams*/, InDataMatrixDeriv& dOut, const OutDataMatrixDeriv& dIn) override;

    const sofa::defaulttype::BaseMatrix* getJ() override;

public:
    typedef helper::vector< defaulttype::BaseMatrix* > js_type;
    virtual const js_type* getJs() override;

protected:
    typedef linearsolver::EigenSparseMatrix<In, Out> eigen_type;
    eigen_type eigen;
    js_type js;
public:

protected:
    std::unique_ptr<MatrixType> matrixJ;
    bool updateJ;

    /// Pointer to the current topology
    sofa::core::topology::BaseMeshTopology* topology;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_SUBSETMAPPING_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_BASE_MECHANICS_API SubsetMapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API SubsetMapping< sofa::defaulttype::Vec1dTypes, sofa::defaulttype::Vec1dTypes >;
extern template class SOFA_BASE_MECHANICS_API SubsetMapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API SubsetMapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Rigid3dTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_MECHANICS_API SubsetMapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API SubsetMapping< sofa::defaulttype::Vec1fTypes, sofa::defaulttype::Vec1fTypes >;
extern template class SOFA_BASE_MECHANICS_API SubsetMapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::ExtVec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API SubsetMapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Rigid3fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_MECHANICS_API SubsetMapping< sofa::defaulttype::Vec3dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_BASE_MECHANICS_API SubsetMapping< sofa::defaulttype::Vec3fTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_BASE_MECHANICS_API SubsetMapping< sofa::defaulttype::Vec1dTypes, sofa::defaulttype::Vec1fTypes >;
extern template class SOFA_BASE_MECHANICS_API SubsetMapping< sofa::defaulttype::Vec1fTypes, sofa::defaulttype::Vec1dTypes >;
extern template class SOFA_BASE_MECHANICS_API SubsetMapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Rigid3fTypes >;
extern template class SOFA_BASE_MECHANICS_API SubsetMapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Rigid3dTypes >;
#endif
#endif

#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
