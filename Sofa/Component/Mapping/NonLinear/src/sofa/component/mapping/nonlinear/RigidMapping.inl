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
#include <sofa/component/mapping/nonlinear/RigidMapping.h>
#include <sofa/core/BaseLocalMappingMatrix.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/State.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/helper/io/XspLoader.h>
#include <sofa/helper/io/SphereLoader.h>
#include <sofa/helper/io/Mesh.h>
#include <sofa/helper/decompose.h>
#include <sofa/core/MechanicalParams.h>

#include <Eigen/Dense>

#include <cstring>
#include <istream>

namespace sofa::component::mapping::nonlinear
{

template <class TIn, class TOut>
class RigidMapping<TIn, TOut>::Loader :
        public helper::io::XspLoaderDataHook,
        public helper::io::SphereLoaderDataHook
{
public:

    RigidMapping<TIn, TOut>* dest;
    helper::WriteAccessor<Data<OutVecCoord> > points;

    Loader(RigidMapping<TIn, TOut>* dest) :
        dest(dest),
        points(dest->d_points)
    {
    }
    void addMass(SReal px, SReal py, SReal pz, SReal, SReal, SReal,
                         SReal, SReal, bool, bool) override
    {
        OutCoord c;
        Out::set(c, px, py, pz);
        points.push_back(c); //Coord((Real)px,(Real)py,(Real)pz));
    }
    void addSphere(SReal px, SReal py, SReal pz, SReal) override
    {
        OutCoord c;
        Out::set(c, px, py, pz);
        points.push_back(c); //Coord((Real)px,(Real)py,(Real)pz));
    }
};

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::load(const char *filename)
{
    d_points.beginEdit()->resize(0);
    d_points.endEdit();

    if (strlen(filename) > 4
            && !strcmp(filename + strlen(filename) - 4, ".xs3"))
    {
        Loader loader(this);
        helper::io::XspLoader::Load(filename, loader);
    }
    else if (strlen(filename) > 4
             && !strcmp(filename + strlen(filename) - 4, ".sph"))
    {
        Loader loader(this);
        helper::io::SphereLoader::Load(filename, loader);
    }
    else if (strlen(filename) > 0)
    {
        // Default to mesh loader
        helper::io::Mesh* mesh = helper::io::Mesh::Create(filename);
        if (mesh != nullptr)
        {
            helper::WriteAccessor<Data<OutVecCoord> > points = d_points;

            points.resize(mesh->getVertices().size());
            for (unsigned int i = 0; i < mesh->getVertices().size(); i++)
            {
                Out::set(points[i],
                         mesh->getVertices()[i][0],
                         mesh->getVertices()[i][1],
                         mesh->getVertices()[i][2]);
            }
            delete mesh;
        }
    }
}


template <class TIn, class TOut>
RigidMapping<TIn, TOut>::RigidMapping()
    : Inherit()
    , d_points(initData(&d_points, "initialPoints", "Local Coordinates of the points"))
    , d_index(initData(&d_index, (unsigned)0, "index", "input DOF index"))
    , d_fileRigidMapping(initData(&d_fileRigidMapping, "filename", "Xsp file where rigid mapping information can be loaded from."))
    , d_useX0(initData(&d_useX0, false, "useX0", "Use x0 instead of local copy of initial positions (to support topo changes)"))
    , d_indexFromEnd(initData(&d_indexFromEnd, false, "indexFromEnd", "input DOF index starts from the end of input DOFs vector"))
    , d_rigidIndexPerPoint(initData(&d_rigidIndexPerPoint, "rigidIndexPerPoint", "For each mapped point, the index of the Rigid it is mapped from"))
    , d_globalToLocalCoords(initData(&d_globalToLocalCoords, "globalToLocalCoords", "are the output DOFs initially expressed in global coordinates"))
    , m_matrixJ()
    , m_updateJ(false)
{
    this->addAlias(&d_fileRigidMapping, "fileRigidMapping");
    sofa::helper::getWriteAccessor(this->d_geometricStiffness)->setSelectedItem(0);
}

template <class TIn, class TOut>
sofa::Index RigidMapping<TIn, TOut>::getRigidIndex(sofa::Index pointIndex ) const
{
    // do we really need this crap?
    if( d_points.getValue().size() == d_rigidIndexPerPoint.getValue().size() ) return d_rigidIndexPerPoint.getValue()[pointIndex];
    else
    {
        if( !d_indexFromEnd.getValue() ) return d_index.getValue();
        else return this->fromModel->getSize()-1-d_index.getValue();
    }
}

template <class TIn, class TOut>
sofa::Size RigidMapping<TIn, TOut>::addPoint(const OutCoord& c)
{
    helper::WriteAccessor<Data<OutVecCoord> > points = d_points;
    const sofa::Size i = sofa::Size(points.size());
    points.push_back(c);
    return i;
}

template <class TIn, class TOut>
sofa::Size RigidMapping<TIn, TOut>::addPoint(const OutCoord& c, sofa::Index indexFrom)
{
    OutVecCoord& points = *d_points.beginEdit();
    const sofa::Size i = sofa::Size(points.size());
    points.push_back(c);
    d_points.endEdit();

    type::vector<unsigned int>& rigidIndexPerPoint = *d_rigidIndexPerPoint.beginEdit();

    if( i && rigidIndexPerPoint.size()!=i )
    {
        rigidIndexPerPoint.resize( i+1 );
        std::fill( rigidIndexPerPoint.begin(), rigidIndexPerPoint.end()-1, getRigidIndex(0) );
    }
    else rigidIndexPerPoint.push_back(indexFrom);

    d_rigidIndexPerPoint.endEdit();
    return i;
}


template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::reinit()
{
    if (d_points.getValue().empty() && this->toModel != nullptr && !d_useX0.getValue())
    {
        const OutVecCoord& xTo =this->toModel->read(core::ConstVecCoordId::position())->getValue();
        helper::WriteOnlyAccessor<Data<OutVecCoord> > points = d_points;
        sofa::Size toModelSize = xTo.size();
        points.resize(toModelSize);
        unsigned int i = 0;
        if (d_globalToLocalCoords.getValue())
        {
            unsigned int i = 0;
            const InVecCoord& xFrom =this->fromModel->read(core::ConstVecCoordId::position())->getValue();

            for (i = 0; i < toModelSize; i++)
            {
                unsigned int rigidIndex = getRigidIndex(i);
                getGlobalToLocalCoords(points[i], xFrom[rigidIndex], xTo[i]);
            }
        }
        else
        {
            for (i = 0; i < toModelSize; i++)
            {
                points[i] = xTo[i];
            }
        }
    }
}


template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::getGlobalToLocalCoords(OutCoord& result, const InCoord& xFrom, const OutCoord& xTo)
{
    result = xFrom.inverseRotate(Out::getCPos(xTo) - In::getCPos(xFrom));
}


template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::init()
{
    if (!d_fileRigidMapping.getValue().empty())
        this->load(d_fileRigidMapping.getFullPath().c_str());

    m_eigenJacobians.resize( 1 );
    m_eigenJacobians[0] = &m_eigenJacobian;

    this->reinit();

    this->Inherit::init();
}

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::clear(sofa::Size reserve)
{
    helper::WriteOnlyAccessor<Data<OutVecCoord> > points = d_points;
    points.clear();
    if (reserve)
        points.reserve(reserve);
    d_rigidIndexPerPoint.beginWriteOnly()->clear();
    d_rigidIndexPerPoint.endEdit();
}

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::setRepartition(sofa::Size value)
{
    msg_deprecated()<<"setRepartition function. Fill rigidIndexPerPoint instead.";

    type::vector<unsigned int>& rigidIndexPerPoint = *d_rigidIndexPerPoint.beginWriteOnly();

    const size_t size = this->toModel->getSize();

    rigidIndexPerPoint.resize( size );

    unsigned int idx = 0;
    for( size_t i=0 ; i<size ; )
    {
         for( size_t j=0; j<value && i<size ; ++j, ++i )
         {
            rigidIndexPerPoint[i] = idx;
         }
         ++idx;
    }

    d_rigidIndexPerPoint.endEdit();
}

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::setRepartition(sofa::type::vector<sofa::Size> values)
{
    msg_deprecated()<<"setRepartition function. Fill rigidIndexPerPoint instead.";

    type::vector<unsigned int>& rigidIndexPerPoint = *d_rigidIndexPerPoint.beginWriteOnly();

    const size_t size = this->toModel->getSize();

    rigidIndexPerPoint.resize( size );

    size_t i = 0;
    for( unsigned int idx=0 ; idx<values.size() ; ++idx )
    {
         for( size_t j=0, jend=values[idx]; j<jend ; ++j, ++i )
         {
            rigidIndexPerPoint[i] = idx;
         }
    }

    d_rigidIndexPerPoint.endEdit();
}

template <class TIn, class TOut>
const typename RigidMapping<TIn, TOut>::OutVecCoord & RigidMapping<TIn, TOut>::getPoints()
{
    if (d_useX0.getValue())
    {
        const Data<OutVecCoord>* v = this->toModel.get()->read(core::VecCoordId::restPosition());
        if (v)
        {
            return v->getValue();
        }
        else
        {
            msg_error()<< "RigidMapping: ERROR useX0 can only be used in MechanicalMappings.";
        }
    }
    return d_points.getValue();
}

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/, Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteOnlyAccessor< Data<OutVecCoord> > out = dOut;
    helper::ReadAccessor< Data<InVecCoord> > in = dIn;
    const OutVecCoord& pts = this->getPoints();

    m_updateJ = true;
    m_eigenJacobian.resizeBlocks(out.size(),in.size());

    m_rotatedPoints.resize(pts.size());
    out.resize(pts.size());

    for (sofa::Index i = 0; i < sofa::Size(pts.size()); i++)
    {
        sofa::Index rigidIndex = getRigidIndex(i);

        m_rotatedPoints[i] = in[rigidIndex].rotate( Out::getCPos(pts[i]) );
        out[i] = in[rigidIndex].mult( pts[i]) ;
    }
}

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/, Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    helper::WriteOnlyAccessor< Data<OutVecDeriv> > out = dOut;
    helper::ReadAccessor< Data<InVecDeriv> > in = dIn;

    const OutVecCoord& pts = this->getPoints();
    out.resize(pts.size());

    for(sofa::Index i=0 ; i<out.size() ; ++i)
    {
        sofa::Index rigidIndex = getRigidIndex(i);
        out[i] = velocityAtRotatedPoint( in[rigidIndex], m_rotatedPoints[i] );
    }
}

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/, Data<InVecDeriv>& dOut, const Data<OutVecDeriv>& dIn)
{
    helper::WriteAccessor< Data<InVecDeriv> > out = dOut;
    helper::ReadAccessor< Data<OutVecDeriv> > in = dIn;

    for(sofa::Index i=0 ; i<in.size() ; ++i)
    {
        sofa::Index rigidIndex = getRigidIndex(i);

        getVCenter(out[rigidIndex]) += Out::getDPos(in[i]);
        updateOmega(getVOrientation(out[rigidIndex]), in[i], m_rotatedPoints[i]);
    }
}

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentForceChangeId, core::ConstMultiVecDerivId childForceId)
{
    if( !d_geometricStiffness.getValue().getSelectedId() )
        return;

    if( m_geometricStiffnessMatrix.compressedMatrix.nonZeros() ) // assembled version
    {
        auto InF = sofa::helper::getWriteOnlyAccessor(*parentForceChangeId[this->fromModel.get()].write());
        auto inDx = sofa::helper::getReadAccessor(*mparams->readDx(this->fromModel.get()));
        m_geometricStiffnessMatrix.addMult( InF.wref(), inDx.ref(), (InReal)mparams->kFactor() );
    }
    else
    {
        // if symmetrized version, force local assembly
        if( d_geometricStiffness.getValue().getSelectedId() == 2 )
        {
            updateK( mparams, childForceId );
            auto InF = sofa::helper::getWriteOnlyAccessor(*parentForceChangeId[this->fromModel.get()].write());
            auto inDx = sofa::helper::getReadAccessor(*mparams->readDx(this->fromModel.get()));

            m_geometricStiffnessMatrix.addMult( InF.wref(), inDx.ref(), (InReal)mparams->kFactor() );
            m_geometricStiffnessMatrix.resize(0,0); // forgot about this matrix
        }
        else
        {
            helper::ReadAccessor<Data<OutVecDeriv> > childForces (*mparams->readF(this->toModel.get()));
            helper::WriteAccessor<Data<InVecDeriv> > parentForces (*parentForceChangeId[this->fromModel.get()].write());
            helper::ReadAccessor<Data<InVecDeriv> > parentDisplacements (*mparams->readDx(this->fromModel.get()));
            InReal kfactor = (InReal)mparams->kFactor();

            for(sofa::Index i=0 ; i< childForces.size() ; ++i)
            {
                sofa::Index rigidIndex = getRigidIndex(i);

                typename TIn::AngularVector& parentTorque = getVOrientation(parentForces[rigidIndex]);
                const typename TIn::AngularVector& parentRotation = getVOrientation(parentDisplacements[rigidIndex]);
                const typename TIn::AngularVector& torqueDecrement = TIn::crosscross( Out::getDPos(childForces[i]), parentRotation, Out::getCPos(m_rotatedPoints[i])) * kfactor;
                parentTorque -=  torqueDecrement;
            }
        }
    }

}


// RigidMapping::applyJT( InMatrixDeriv& out, const OutMatrixDeriv& in ) //
// this function propagate the constraint through the rigid mapping :
// if one constraint along (vector n) with a value (v) is applied on the childModel (like collision model)
// then this constraint is transformed by (Jt.n) with value (v) for the rigid model
// There is a specificity of this propagateConstraint: we have to find the application point on the childModel
// in order to compute the right constaint on the rigidModel.
template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::applyJT(const core::ConstraintParams * /*cparams*/, Data<InMatrixDeriv>& dOut, const Data<OutMatrixDeriv>& dIn)
{
    InMatrixDeriv& out = *dOut.beginEdit();
    const OutMatrixDeriv& in = dIn.getValue();

    dmsg_info() << "J on mapped DOFs == " << in << msgendl
                << "J on input  DOFs == " << out ;

    const unsigned int numDofs = this->getFromModel()->getSize();

    // TODO the implementation on the new data structure could maybe be optimized
    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        for (unsigned int ito = 0; ito < numDofs; ito++)
        {
            typename InDeriv::Pos v;
            typename InDeriv::Rot omega = typename InDeriv::Rot();
            bool needToInsert = false;

            for (typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin(); colIt != rowIt.end(); ++colIt)
            {
                const unsigned int rigidIndex = getRigidIndex( colIt.index() );
                if(rigidIndex != ito)
                    continue;

                needToInsert = true;
                const OutDeriv f = colIt.val();
                v += Out::getDPos(f);
                updateOmega(omega, f, m_rotatedPoints[colIt.index()]);
            }

            if (needToInsert)
            {
                const InDeriv result(v, omega);

                typename InMatrixDeriv::RowIterator o = out.writeLine(rowIt.index());
                o.addCol(ito, result);
            }
        }
    }

    dmsg_info() << "new J on input  DOFs = " << out ;

    dOut.endEdit();
}


template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::updateOmega(typename InDeriv::Rot& omega, const OutDeriv& out, const OutCoord& rotatedpoint)
{
    omega += (typename InDeriv::Rot)cross(Out::getCPos(rotatedpoint), Out::getDPos(out));
}

namespace impl {

template<class U, class Coord>
static void fill_block(Eigen::Matrix<U, 3, 6>& block, const Coord& v) {
    U x = v[0];
    U y = v[1];
    U z = v[2];

    // note: this is -hat(v)
    block.template rightCols<3>() <<

        0,   z,  -y,
        -z,  0,   x,
        y,  -x,   0;
}

template<class U, class Coord>
static void fill_block(Eigen::Matrix<U, 6, 6>& block, const Coord& v) {
    U x = v[0];
    U y = v[1];
    U z = v[2];

    // note: this is -hat(v)
    block.template topRightCorner<3, 3>() <<

        0,   z,  -y,
        -z,  0,   x,
        y,  -x,   0;
}

template<class U, class Coord>
void fill_block(Eigen::Matrix<U, 2, 3>& block, const Coord& v) {
    U x = v[0];
    U y = v[1];

    // note: this is -hat(v)
    block.template rightCols<1>() <<
        -y,
        x;
}


}

template <class TIn, class TOut>
const type::vector<sofa::linearalgebra::BaseMatrix*>* RigidMapping<TIn, TOut>::getJs()
{
    const OutVecCoord& out =this->toModel->read(core::ConstVecCoordId::position())->getValue();
    const InVecCoord& in =this->fromModel->read(core::ConstVecCoordId::position())->getValue();

    typename SparseMatrixEigen::CompressedMatrix& J = m_eigenJacobian.compressedMatrix;

    if( m_updateJ || J.size() == 0 )
    {

        m_updateJ = false;

        J.resize(out.size() * NOut, in.size() * NIn);
        J.setZero();

        // matrix chunk
        typedef typename TOut::Real real;
        typedef Eigen::Matrix<real, NOut, NIn> block_type;
        block_type block;

        // translation part
        block.template leftCols<NOut>().setIdentity();


        for(sofa::Index outIdx=0 ; outIdx< m_rotatedPoints.size() ; ++outIdx)
        {
            sofa::Index inIdx = getRigidIndex(outIdx);

            const OutCoord& v = m_rotatedPoints[outIdx];

            impl::fill_block(block, v);

            // block is set, now insert it in sparse matrix
            for(unsigned i = 0; i < NOut; ++i){
                unsigned row = outIdx * NOut + i;

                J.startVec( row );

                // TODO optimize identity off-diagonal and
                // skew-symmetric diagonal
                for(unsigned j = 0; j < NIn; ++j) {
                    unsigned col = inIdx * NIn + j;

                    if( block(i, j) != 0 ) {

                        J.insertBack(row, col) = block(i, j);
                    }
                }
            }
        }

        J.finalize();
    }

    return &m_eigenJacobians;
}



template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::updateK( const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForceId )
{
    SOFA_UNUSED(mparams);
    const unsigned geomStiff = d_geometricStiffness.getValue().getSelectedId();

    if( !geomStiff ) { m_geometricStiffnessMatrix.resize(0,0); return; }

    m_geometricStiffnessMatrix.resizeBlocks( this->fromModel->getSize(), this->fromModel->getSize() );

    const OutVecDeriv& childForces = childForceId[this->toModel.get()].read()->getValue();

    // sorted in-out
    typedef std::map<unsigned, type::vector<unsigned> > in_out_type;
    in_out_type in_out;

    // wahoo it is heavy, can't we find lighter?
    for(sofa::Index i = 0, n = m_rotatedPoints.size(); i < n; ++i)
        in_out[ getRigidIndex(i) ].push_back(i);

    for( in_out_type::const_iterator it = in_out.begin(), end = in_out.end() ; it != end; ++it )
    {
        const unsigned rigidIdx = it->first;

        static const unsigned rotation_dimension = TIn::deriv_total_size - TIn::spatial_dimensions;
        type::Mat<rotation_dimension,rotation_dimension,OutReal> block;

        for( unsigned int w=0 ; w<it->second.size() ; ++w )
        {
            const unsigned pointIdx = it->second[w];
            block += type::crossProductMatrix<OutReal>( Out::getDPos(childForces[pointIdx]) )
                    * type::crossProductMatrix<OutReal>( Out::getCPos(m_rotatedPoints[pointIdx]) );
        }

        if( geomStiff == 2 )
        {
            block.symmetrize(); // symmetrization
            helper::Decompose<OutReal>::NSDProjection( block ); // negative, semi-definite projection
        }

        for(unsigned j = 0; j < rotation_dimension; ++j) {

            const unsigned row = TIn::deriv_total_size * rigidIdx + TIn::spatial_dimensions + j;

            for(unsigned k = 0; k < rotation_dimension; ++k)
            {
                const unsigned col = TIn::deriv_total_size * rigidIdx + TIn::spatial_dimensions + k;

                if( block(j, k) != static_cast<OutReal>(0))
                {
                    m_geometricStiffnessMatrix.add(row, col, (InReal)block[j][k]);
                }
            }
        }
    }

    m_geometricStiffnessMatrix.compress();
}


template <class TIn, class TOut>
const sofa::linearalgebra::BaseMatrix* RigidMapping<TIn, TOut>::getK()
{
    if( m_geometricStiffnessMatrix.compressedMatrix.nonZeros() ) return &m_geometricStiffnessMatrix;
    else return nullptr;
}

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::buildGeometricStiffnessMatrix(
    sofa::core::GeometricStiffnessMatrix* matrices)
{
    const unsigned int geomStiff = d_geometricStiffness.getValue().getSelectedId();

    if( !geomStiff )
    {
        return;
    }

    if constexpr (TOut::spatial_dimensions != 3)
    {
        static std::set<RigidMapping<TIn, TOut>*> hasShownError;
        msg_warning_when(hasShownError.insert(this).second) << "Geometric stiffness is not supported in " << TOut::spatial_dimensions << "d";
    }
    else
    {
        if (geomStiff == 1)
        {
            // This method corresponds to a non-symmetric matrix, due to the non-commutativity of the group of rotations.
            checkLinearSolverSymmetry(matrices->getMechanicalParams());
        }

        const auto dJdx = matrices->getMappingDerivativeIn(this->fromModel).withRespectToPositionsIn(this->fromModel);

        const auto childForces = this->toModel->readTotalForces();

        std::map<unsigned, sofa::type::vector<unsigned> > in_out;
        for(sofa::Index i = 0; i < m_rotatedPoints.size(); ++i)
        {
            in_out[ getRigidIndex(i) ].push_back(i);
        }

        for (auto& [fst, snd] : in_out)
        {
            const unsigned rigidIdx = fst;

            static constexpr unsigned rotation_dimension = TIn::deriv_total_size - TIn::spatial_dimensions;

            type::Mat<rotation_dimension,rotation_dimension,OutReal> block;

            for (const auto pointIdx : snd)
            {
                block += type::crossProductMatrix<OutReal>( Out::getDPos(childForces[pointIdx]) )
                        * type::crossProductMatrix<OutReal>( Out::getCPos(m_rotatedPoints[pointIdx]) );
            }

            if( geomStiff == 2 )
            {
                block.symmetrize(); // symmetrization
                helper::Decompose<OutReal>::NSDProjection( block ); // negative, semi-definite projection
            }

            const auto matrixIndex = TIn::deriv_total_size * rigidIdx + TIn::spatial_dimensions;

            dJdx(matrixIndex, matrixIndex) += block;
        }
    }
}


template <class TIn, class TOut>
const sofa::linearalgebra::BaseMatrix* RigidMapping<TIn, TOut>::getJ()
{
    const OutVecCoord& out =this->toModel->read(core::ConstVecCoordId::position())->getValue();
    const InVecCoord& in =this->fromModel->read(core::ConstVecCoordId::position())->getValue();
    const OutVecCoord& pts = this->getPoints();
    assert(pts.size() == out.size());

    if (m_matrixJ.get() == 0 || m_updateJ)
    {
        m_updateJ = false;
        if (m_matrixJ.get() == 0 ||
                (unsigned int)m_matrixJ->rowBSize() != out.size() ||
                (unsigned int)m_matrixJ->colBSize() != in.size())
        {
            m_matrixJ.reset(new MatrixType(out.size() * NOut, in.size() * NIn));
        }
        else
        {
            m_matrixJ->clear();
        }


        for (unsigned int outIdx = 0; outIdx < pts.size() ; outIdx++)
        {
            const unsigned int inIdx = getRigidIndex(outIdx);

            setJMatrixBlock(outIdx, inIdx);
        }
    }
    m_matrixJ->compress();
    return m_matrixJ.get();
}

template<class Real>
struct RigidMappingMatrixHelper<2, Real>
{
    template <class Matrix, class Vector>
    static void setMatrix(Matrix& mat,
                          const Vector& vec)
    {
        mat[0][0] = (Real) 1     ;    mat[1][0] = (Real) 0     ;
        mat[0][1] = (Real) 0     ;    mat[1][1] = (Real) 1     ;
        mat[0][2] = (Real)-vec[1];    mat[1][2] = (Real) vec[0];
    }
};

template<class Real>
struct RigidMappingMatrixHelper<3, Real>
{
    template <class Matrix, class Vector>
    static void setMatrix(Matrix& mat,
                          const Vector& vec)
    {
        // out = J in
        // J = [ I -OM^ ]
        mat[0][0] = (Real) 1     ;    mat[1][0] = (Real) 0     ;    mat[2][0] = (Real) 0     ;
        mat[0][1] = (Real) 0     ;    mat[1][1] = (Real) 1     ;    mat[2][1] = (Real) 0     ;
        mat[0][2] = (Real) 0     ;    mat[1][2] = (Real) 0     ;    mat[2][2] = (Real) 1     ;
        mat[0][3] = (Real) 0     ;    mat[1][3] = (Real)-vec[2];    mat[2][3] = (Real) vec[1];
        mat[0][4] = (Real) vec[2];    mat[1][4] = (Real) 0     ;    mat[2][4] = (Real)-vec[0];
        mat[0][5] = (Real)-vec[1];    mat[1][5] = (Real) vec[0];    mat[2][5] = (Real) 0     ;
    }
};

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::setJMatrixBlock(unsigned outIdx, unsigned inIdx)
{
    MBloc& block = *m_matrixJ->wblock(outIdx, inIdx, true);
    RigidMappingMatrixHelper<N, OutReal>::setMatrix(block, m_rotatedPoints[outIdx]);
}

template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowMappings() || this->toModel==nullptr )
        return;
    std::vector<type::Vec3> points;
    type::Vec3 point;

    const OutVecCoord& x =this->toModel->read(core::ConstVecCoordId::position())->getValue();
    for (unsigned int i = 0; i < x.size(); i++)
    {
        point = Out::getCPos(x[i]);
        points.push_back(point);
    }
    vparams->drawTool()->drawPoints(points, 7, sofa::type::RGBAColor::yellow() );
}


template <class TIn, class TOut>
void RigidMapping<TIn, TOut>::parse(core::objectmodel::BaseObjectDescription* arg)
{
    Inherit::parse(arg);

    // to be backward compatible with previous data structure
    const char* repartitionChar = arg->getAttribute("repartition");
    if( repartitionChar )
    {
        msg_deprecated() << "parse: You are using a deprecated Data 'repartition', please use the new structure data rigidIndexPerPoint";

        type::vector< unsigned int > repartition;
        std::istringstream ss( repartitionChar );
        repartition.read( ss );
        setRepartition( repartition );
    }
}

} // namespace sofa::component::mapping::nonlinear
