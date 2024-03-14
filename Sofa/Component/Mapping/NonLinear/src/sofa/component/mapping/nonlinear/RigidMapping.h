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
class RigidMapping : public core::Mapping<TIn, TOut>, public NonLinearMappingData<true>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(RigidMapping,TIn,TOut), SOFA_TEMPLATE2(core::Mapping,TIn,TOut));

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;

    typedef typename Out::Real OutReal;
    typedef typename Out::Coord OutCoord;
    typedef typename Out::Deriv OutDeriv;
    typedef typename Out::VecCoord OutVecCoord;
    typedef typename Out::VecDeriv OutVecDeriv;
    typedef typename Out::MatrixDeriv OutMatrixDeriv;

    typedef typename In::Real InReal;
    typedef typename In::Coord InCoord;
    typedef typename In::Deriv InDeriv;
    typedef typename In::VecCoord InVecCoord;
    typedef typename In::VecDeriv InVecDeriv;
    typedef typename In::MatrixDeriv InMatrixDeriv;

    enum
    {
        N = Out::spatial_dimensions
    };
    enum
    {
        NIn = sofa::defaulttype::DataTypeInfo<InDeriv>::Size
    };
    enum
    {
        NOut = sofa::defaulttype::DataTypeInfo<OutDeriv>::Size
    };
    typedef type::Mat<N, N, OutReal> Mat;
    typedef type::Vec<N, OutReal> Vector;
    typedef type::Mat<NOut, NIn, OutReal> MBloc;
    typedef sofa::linearalgebra::CompressedRowSparseMatrix<MBloc> MatrixType;

    Data<OutVecCoord> d_points;    ///< mapped points in local coordinates
    SOFA_ATTRIBUTE_DISABLED("v23.06", "v23.12", "Use d_points instead") DeprecatedAndRemoved points;

    OutVecCoord m_rotatedPoints;   ///< vectors from frame origin to mapped points, projected to world coordinates
    SOFA_ATTRIBUTE_DISABLED("v23.06", "v23.12", "Use m_rotatedPoints instead") DeprecatedAndRemoved rotatedPoints;

    RigidMappingInternalData<In, Out> m_data;
    SOFA_ATTRIBUTE_DISABLED("v23.06", "v23.12", "Use m_data instead") DeprecatedAndRemoved data;

    Data<sofa::Index> d_index; ///< input DOF index
    SOFA_ATTRIBUTE_DISABLED("v23.06", "v23.12", "Use d_index instead") DeprecatedAndRemoved index;

    sofa::core::objectmodel::DataFileName d_fileRigidMapping; ///< Filename
    SOFA_ATTRIBUTE_DISABLED("v23.06", "v23.12", "Use d_fileRigidMapping instead") DeprecatedAndRemoved fileRigidMapping;

    Data<bool> d_useX0; ///< Use x0 instead of local copy of initial positions (to support topo changes)
    SOFA_ATTRIBUTE_DISABLED("v23.06", "v23.12", "Use d_useX0 instead") DeprecatedAndRemoved useX0;

    Data<bool> d_indexFromEnd; ///< input DOF index starts from the end of input DOFs vector
    SOFA_ATTRIBUTE_DISABLED("v23.06", "v23.12", "Use d_indexFromEnd instead") DeprecatedAndRemoved indexFromEnd;

    Data< type::vector<unsigned int> > d_rigidIndexPerPoint; ///< For each mapped point, the index of the Rigid it is mapped from
    SOFA_ATTRIBUTE_DISABLED("v23.06", "v23.12", "Use d_rigidIndexPerPoint instead") DeprecatedAndRemoved rigidIndexPerPoint;

    Data<bool> d_globalToLocalCoords; ///< are the output DOFs initially expressed in global coordinates
    SOFA_ATTRIBUTE_DISABLED("v23.06", "v23.12", "Use d_globalToLocalCoords instead") DeprecatedAndRemoved globalToLocalCoords;

    SOFA_ATTRIBUTE_DISABLED("v23.06", "v23.06", "Use d_geometricStiffness instead") DeprecatedAndRemoved geometricStiffness;

protected:
    RigidMapping();
    virtual ~RigidMapping() {}

    unsigned int getRigidIndex( unsigned int pointIndex ) const;

public:
    sofa::Size addPoint(const OutCoord& c);
    sofa::Size addPoint(const OutCoord& c, sofa::Index indexFrom);

    void init() override;

    /// Compute the local coordinates based on the current output coordinates.
    void reinit() override;

    void apply(const core::MechanicalParams *mparams, Data<OutVecCoord>& out, const Data<InVecCoord>& in) override;

    void applyJ(const core::MechanicalParams *mparams, Data<OutVecDeriv>& out, const Data<InVecDeriv>& in) override;

    void applyJT(const core::MechanicalParams *mparams, Data<InVecDeriv>& out, const Data<OutVecDeriv>& in) override;

    void applyJT(const core::ConstraintParams *cparams, Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in) override;

    void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentForce, core::ConstMultiVecDerivId  childForce ) override;

    const sofa::linearalgebra::BaseMatrix* getJ() override;

    virtual const type::vector<sofa::linearalgebra::BaseMatrix*>* getJs() override;

    void updateK( const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForceId ) override;
    const linearalgebra::BaseMatrix* getK() override;
    void buildGeometricStiffnessMatrix(sofa::core::GeometricStiffnessMatrix* matrices) override;


    void draw(const core::visual::VisualParams* vparams) override;

    void clear(sofa::Size reserve = 0);

    /// to give the number of mapped points attached to each rigid frame
    /// @warning the mapped points must be sorted by their parent frame indices
    /// for backward compatibility with previous data structure
    void setRepartition(sofa::Size value);
    void setRepartition(sofa::type::vector<sofa::Size> values);

    void parse(core::objectmodel::BaseObjectDescription* arg) override;

    void getGlobalToLocalCoords(OutCoord& result, const InCoord& xfrom, const OutCoord& xto);
    void updateOmega(typename InDeriv::Rot& omega, const OutDeriv& out, const OutCoord& rotatedpoint);

protected:
    class Loader;

    void load(const char* filename);
    const OutVecCoord& getPoints();
    void setJMatrixBlock(sofa::Index outIdx, sofa::Index inIdx);

    std::unique_ptr<MatrixType> m_matrixJ;
    SOFA_ATTRIBUTE_DISABLED("v23.06", "v23.12", "Use m_matrixJ instead") DeprecatedAndRemoved matrixJ;
    bool m_updateJ;
    SOFA_ATTRIBUTE_DISABLED("v23.06", "v23.12", "Use m_updateJ instead") DeprecatedAndRemoved updateJ;

    typedef linearalgebra::EigenSparseMatrix<In,Out> SparseMatrixEigen;
    SparseMatrixEigen m_eigenJacobian;                      ///< Jacobian of the mapping used by getJs
    SOFA_ATTRIBUTE_DISABLED("v23.06", "v23.12", "Use m_eigenJacobian instead") DeprecatedAndRemoved eigenJacobian;
    type::vector<sofa::linearalgebra::BaseMatrix*> m_eigenJacobians; /// used by getJs
    SOFA_ATTRIBUTE_DISABLED("v23.06", "v23.12", "Use m_eigenJacobians instead") DeprecatedAndRemoved eigenJacobians;

    typedef linearalgebra::EigenSparseMatrix<In,In> StiffnessSparseMatrixEigen;
    StiffnessSparseMatrixEigen m_geometricStiffnessMatrix;
    SOFA_ATTRIBUTE_DISABLED("v23.06", "v23.12", "Use m_geometricStiffnessMatrix instead") DeprecatedAndRemoved geometricStiffnessMatrix;
};

template <std::size_t N, class Real>
struct RigidMappingMatrixHelper;

template<>
void RigidMapping< sofa::defaulttype::Rigid2Types, sofa::defaulttype::Vec2Types >::updateK( const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForceId );
template<>
const linearalgebra::BaseMatrix* RigidMapping< sofa::defaulttype::Rigid2Types, sofa::defaulttype::Vec2Types >::getK();

#if !defined(SOFA_COMPONENT_MAPPING_RIGIDMAPPING_CPP)
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API RigidMapping< sofa::defaulttype::Rigid3Types, sofa::defaulttype::Vec3Types >;
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API RigidMapping< sofa::defaulttype::Rigid3Types, sofa::defaulttype::Rigid3Types >;
extern template class SOFA_COMPONENT_MAPPING_NONLINEAR_API RigidMapping< sofa::defaulttype::Rigid2Types, sofa::defaulttype::Vec2Types >;
#endif

} // namespace sofa::component::mapping::nonlinear
