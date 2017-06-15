/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_RIGIDMAPPING_H
#define SOFA_COMPONENT_MAPPING_RIGIDMAPPING_H
#include "config.h"

#include <sofa/core/Mapping.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>
#include <SofaEigen2Solver/EigenSparseMatrix.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <vector>
#include <memory>

namespace sofa
{

namespace component
{

namespace mapping
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
    typedef defaulttype::Mat<N, N, Real> Mat;
    typedef defaulttype::Vec<N, Real> Vector;
    typedef defaulttype::Mat<NOut, NIn, Real> MBloc;
    typedef sofa::component::linearsolver::CompressedRowSparseMatrix<MBloc> MatrixType;
    typedef typename Inherit::ForceMask ForceMask;

    Data<VecCoord> points;    ///< mapped points in local coordinates
    VecCoord rotatedPoints;   ///< vectors from frame origin to mapped points, projected to world coordinates
    RigidMappingInternalData<In, Out> data;
    Data<unsigned int> index;
    sofa::core::objectmodel::DataFileName fileRigidMapping;
    Data<bool> useX0;
    Data<bool> indexFromEnd;

    Data< helper::vector<unsigned int> > rigidIndexPerPoint;
    Data<bool> globalToLocalCoords;

    Data<int> geometricStiffness;

protected:
    RigidMapping();
    virtual ~RigidMapping() {}

    unsigned int getRigidIndex( unsigned int pointIndex ) const;

public:
    int addPoint(const Coord& c);
    int addPoint(const Coord& c, unsigned int indexFrom);

    virtual void init();

    /// Compute the local coordinates based on the current output coordinates.
    virtual void reinit();

    virtual void apply(const core::MechanicalParams *mparams, Data<VecCoord>& out, const Data<InVecCoord>& in);

    virtual void applyJ(const core::MechanicalParams *mparams, Data<VecDeriv>& out, const Data<InVecDeriv>& in);

    virtual void applyJT(const core::MechanicalParams *mparams, Data<InVecDeriv>& out, const Data<VecDeriv>& in);

    virtual void applyJT(const core::ConstraintParams *cparams, Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in);

    virtual void applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentForce, core::ConstMultiVecDerivId  childForce );

    virtual const sofa::defaulttype::BaseMatrix* getJ();

    virtual const helper::vector<sofa::defaulttype::BaseMatrix*>* getJs();

    virtual void updateK( const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForceId );
    virtual const defaulttype::BaseMatrix* getK();


    virtual void draw(const core::visual::VisualParams* vparams);

    void clear(int reserve = 0);

    /// to give the number of mapped points attached to each rigid frame
    /// @warning the mapped points must be sorted by their parent frame indices
    /// for backward compatibility with previous data structure
    void setRepartition(unsigned int value);
    void setRepartition(sofa::helper::vector<unsigned int> values);

    void parse(core::objectmodel::BaseObjectDescription* arg);

protected:
    class Loader;

    void load(const char* filename);
    const VecCoord& getPoints();
    void setJMatrixBlock(unsigned outIdx, unsigned inIdx);

    std::unique_ptr<MatrixType> matrixJ;
    bool updateJ;

    typedef linearsolver::EigenSparseMatrix<In,Out> SparseMatrixEigen;
    SparseMatrixEigen eigenJacobian;                      ///< Jacobian of the mapping used by getJs
    helper::vector<sofa::defaulttype::BaseMatrix*> eigenJacobians; /// used by getJs

    typedef linearsolver::EigenSparseMatrix<In,In> StiffnessSparseMatrixEigen;
    StiffnessSparseMatrixEigen geometricStiffnessMatrix;
};

template <int N, class Real>
struct RigidMappingMatrixHelper;



#ifndef SOFA_FLOAT
template<>
void RigidMapping< sofa::defaulttype::Rigid2dTypes, sofa::defaulttype::Vec2dTypes >::updateK( const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForceId );
template<>
const defaulttype::BaseMatrix* RigidMapping< sofa::defaulttype::Rigid2dTypes, sofa::defaulttype::Vec2dTypes >::getK();
#endif
#ifndef SOFA_DOUBLE
template<>
void RigidMapping< sofa::defaulttype::Rigid2fTypes, sofa::defaulttype::Vec2fTypes >::updateK( const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForceId );
template<>
const defaulttype::BaseMatrix* RigidMapping< sofa::defaulttype::Rigid2fTypes, sofa::defaulttype::Vec2fTypes >::getK();
#endif



#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MAPPING_RIGIDMAPPING_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_RIGID_API RigidMapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Vec3dTypes >;
extern template class SOFA_RIGID_API RigidMapping< sofa::defaulttype::Rigid2dTypes, sofa::defaulttype::Vec2dTypes >;
extern template class SOFA_RIGID_API RigidMapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::ExtVec3fTypes >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_RIGID_API RigidMapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_RIGID_API RigidMapping< sofa::defaulttype::Rigid2fTypes, sofa::defaulttype::Vec2fTypes >;
extern template class SOFA_RIGID_API RigidMapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::ExtVec3fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_RIGID_API RigidMapping< sofa::defaulttype::Rigid3dTypes, sofa::defaulttype::Vec3fTypes >;
extern template class SOFA_RIGID_API RigidMapping< sofa::defaulttype::Rigid3fTypes, sofa::defaulttype::Vec3dTypes >;
#endif
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
