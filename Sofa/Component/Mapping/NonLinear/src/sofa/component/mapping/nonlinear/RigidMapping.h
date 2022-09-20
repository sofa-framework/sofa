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
#include <sofa/core/objectmodel/DataFileName.h>

#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/EigenSparseMatrix.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/type/vector.h>

namespace sofa::component::mapping::nonlinear
{

/// This class can be overridden if needed for additionnal storage within template specializations.
template<class InDataTypes, class OutDataTypes>
class RigidMappingInternalData
{
public:
};

template <class TIn, class TOut>
class RigidMapping : public core::Mapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(RigidMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef Out OutDataTypes;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;
    typedef typename In::Real InReal;
    typedef typename In::Deriv InDeriv;
    typedef typename InDeriv::Pos DPos;
    typedef typename InDeriv::Rot DRot;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;
    typedef typename Coord::value_type Real;
    enum
    {
        N = OutDataTypes::spatial_dimensions
    };
    enum
    {
        NIn = sofa::defaulttype::DataTypeInfo<InDeriv>::Size
    };
    enum
    {
        NOut = sofa::defaulttype::DataTypeInfo<Deriv>::Size
    };
    typedef type::Mat<N, N, Real> Mat;
    typedef type::Vec<N, Real> Vector;
    typedef type::Mat<NOut, NIn, Real> MBloc;
    typedef sofa::linearalgebra::CompressedRowSparseMatrix<MBloc> MatrixType;

    Data<VecCoord> points;    ///< mapped points in local coordinates
    VecCoord rotatedPoints;   ///< vectors from frame origin to mapped points, projected to world coordinates
    RigidMappingInternalData<In, Out> data;
    Data<sofa::Index> index; ///< input DOF index
    sofa::core::objectmodel::DataFileName fileRigidMapping; ///< Filename
    Data<bool> useX0; ///< Use x0 instead of local copy of initial positions (to support topo changes)
    Data<bool> indexFromEnd; ///< input DOF index starts from the end of input DOFs vector

    Data< type::vector<unsigned int> > rigidIndexPerPoint; ///< For each mapped point, the index of the Rigid it is mapped from
    Data<bool> globalToLocalCoords; ///< are the output DOFs initially expressed in global coordinates

    Data<int> geometricStiffness; ///< assemble (and use) geometric stiffness (0=no GS, 1=non symmetric, 2=symmetrized)

protected:
    RigidMapping();
    virtual ~RigidMapping() {}

    unsigned int getRigidIndex( unsigned int pointIndex ) const;

public:
    sofa::Size addPoint(const Coord& c);
    sofa::Size addPoint(const Coord& c, sofa::Index indexFrom);

    void doBaseObjectInit() override;

    /// Compute the local coordinates based on the current output coordinates.
    void reinit() override;

    void apply(const core::MechanicalParams *mparams, Data<VecCoord>& out, const Data<InVecCoord>& in) override;

    void applyJ(const core::MechanicalParams *mparams, Data<VecDeriv>& out, const Data<InVecDeriv>& in) override;

    void applyJT(const core::MechanicalParams *mparams, Data<InVecDeriv>& out, const Data<VecDeriv>& in) override;

    void applyJT(const core::ConstraintParams *cparams, Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in) override;

    void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentForce, core::ConstMultiVecDerivId  childForce ) override;

    const sofa::linearalgebra::BaseMatrix* getJ() override;

    virtual const type::vector<sofa::linearalgebra::BaseMatrix*>* getJs() override;

    void updateK( const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForceId ) override;
    const linearalgebra::BaseMatrix* getK() override;


    void draw(const core::visual::VisualParams* vparams) override;

    void clear(sofa::Size reserve = 0);

    /// to give the number of mapped points attached to each rigid frame
    /// @warning the mapped points must be sorted by their parent frame indices
    /// for backward compatibility with previous data structure
    void setRepartition(sofa::Size value);
    void setRepartition(sofa::type::vector<sofa::Size> values);

    void parse(core::objectmodel::BaseObjectDescription* arg) override;

protected:
    class Loader;

    void load(const char* filename);
    const VecCoord& getPoints();
    void setJMatrixBlock(sofa::Index outIdx, sofa::Index inIdx);

    std::unique_ptr<MatrixType> matrixJ;
    bool updateJ;

    typedef linearalgebra::EigenSparseMatrix<In,Out> SparseMatrixEigen;
    SparseMatrixEigen eigenJacobian;                      ///< Jacobian of the mapping used by getJs
    type::vector<sofa::linearalgebra::BaseMatrix*> eigenJacobians; /// used by getJs

    typedef linearalgebra::EigenSparseMatrix<In,In> StiffnessSparseMatrixEigen;
    StiffnessSparseMatrixEigen geometricStiffnessMatrix;
};

template <std::size_t N, class Real>
struct RigidMappingMatrixHelper;

template<>
void RigidMapping< sofa::defaulttype::Rigid2Types, sofa::defaulttype::Vec2Types >::updateK( const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForceId );
template<>
const linearalgebra::BaseMatrix* RigidMapping< sofa::defaulttype::Rigid2Types, sofa::defaulttype::Vec2Types >::getK();

#if  !defined(SOFA_COMPONENT_MAPPING_RIGIDMAPPING_CPP)
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API RigidMapping< sofa::defaulttype::Rigid3Types, sofa::defaulttype::Vec3Types >;
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API RigidMapping< sofa::defaulttype::Rigid2Types, sofa::defaulttype::Vec2Types >;
#endif

} // namespace sofa::component::mapping::nonlinear
