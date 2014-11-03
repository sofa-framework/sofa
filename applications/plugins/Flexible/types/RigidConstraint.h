/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef FLEXIBLE_RigidConstraint_H
#define FLEXIBLE_RigidConstraint_H

#include "../initFlexible.h"
#include "AffineTypes.h"
#include "QuadraticTypes.h"

#include <sofa/core/behavior/ProjectiveConstraintSet.inl>
#include <SofaEigen2Solver/EigenSparseMatrix.h>

#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using helper::vector;

using namespace sofa::defaulttype;
/** Make non-rigid frames rigid.
*/
template <class DataTypes>
class RigidConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(RigidConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, DataTypes));

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef Data<typename DataTypes::VecCoord> DataVecCoord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef Data<typename DataTypes::VecDeriv> DataVecDeriv;
    typedef Data<typename DataTypes::MatrixDeriv> DataMatrixDeriv;
    typedef typename DataTypes::MatrixDeriv::RowIterator MatrixDerivRowIterator;
    typedef typename DataTypes::MatrixDeriv::RowType MatrixDerivRowType;

    typedef helper::vector<unsigned int> Indices;

    typedef linearsolver::EigenBaseSparseMatrix<SReal> BaseSparseMatrix;
    typedef linearsolver::EigenSparseMatrix<DataTypes,DataTypes> SparseMatrix;
    typedef typename SparseMatrix::Block Block;                                       ///< projection matrix of a particle displacement to the plane
    enum {bsize=SparseMatrix::Nin};                                                   ///< size of a block

    Data<Indices> f_index;   ///< Indices of the constrained frames
    Data<double> _drawSize;

    VecCoord oldPos;
    Block bIdentity; // precomputed identity block for unconstrained dofs

    // -- Constraint interface
    virtual void init()
    {
        Inherit1::init();
        for(unsigned i=0; i<bsize; i++) for(unsigned j=0; j<bsize; j++) bIdentity[i][j]=(i==j)?1.:0;
        reinit();
    }

    virtual void reinit()
    {
        // sort indices to fill the jacobian in ascending order
        Indices tmp = f_index.getValue();
        std::sort(tmp.begin(),tmp.end());
        f_index.setValue(tmp);

        // store positions to compute jacobian
        const vector<unsigned> & indices = f_index.getValue();
        oldPos.resize(indices.size());
        helper::ReadAccessor< Data< VecCoord > > pos(*this->getMState()->read(core::ConstVecCoordId::position()));
        for(unsigned i=0; i<indices.size(); i++)       oldPos[i]=pos[indices[i]];
    }

    template <class VecDerivType>
    void projectResponseT( VecDerivType& res)
    {
        const vector<unsigned> & indices = f_index.getValue();
        for(unsigned ind=0; ind<indices.size(); ind++) res[indices[ind]].setRigid( oldPos[ind]);
    }

    virtual void projectResponse(const core::MechanicalParams* /*mparams*/, DataVecDeriv& resData)
    {
        helper::WriteAccessor<DataVecDeriv> res = resData;
        projectResponseT<VecDeriv>( res.wref());
    }

    virtual void projectVelocity(const core::MechanicalParams* /*mparams*/, DataVecDeriv& resData)
    {
        helper::WriteAccessor<DataVecDeriv> res = resData;
        projectResponseT<VecDeriv>( res.wref());
    }

    virtual void projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& xData)
    {
        helper::WriteAccessor<DataVecCoord> res = xData;
        const vector<unsigned> & indices = f_index.getValue();
        oldPos.resize(indices.size());
        for(unsigned i=0; i<indices.size(); i++)      { oldPos[i]=res[indices[i]];  res[indices[i]].setRigid(); }
    }

    virtual void projectJacobianMatrix(const core::MechanicalParams* /*mparams*/, DataMatrixDeriv& cData)
    {
        helper::WriteAccessor<DataMatrixDeriv> c = cData;

        MatrixDerivRowIterator rowIt = c->begin();
        MatrixDerivRowIterator rowItEnd = c->end();

        while (rowIt != rowItEnd)
        {
            projectResponseT<MatrixDerivRowType>(rowIt.row());
            ++rowIt;
        }
    }

    virtual void applyConstraint(defaulttype::BaseMatrix *, unsigned int /*offset*/) {}
    virtual void applyConstraint(defaulttype::BaseVector *, unsigned int /*offset*/) {}

    Block getBlock(unsigned int ind_index)
    {
        Block J;
        helper::ReadAccessor< Data< VecDeriv > > vel(*this->getMState()->read(core::ConstVecDerivId::velocity()));
        unsigned int i=f_index.getValue()[ind_index];
        vel[i].getJRigid(oldPos[ind_index],J);
        return J;
    }

    void projectMatrix( sofa::defaulttype::BaseMatrix* M, unsigned offset )
    {
        // resize the jacobian
        SparseMatrix jacobian; ///< projection matrix
        unsigned numBlocks = this->mstate->getSize();
        unsigned blockSize = DataTypes::deriv_total_size;
        jacobian.resize( numBlocks*blockSize,numBlocks*blockSize );

        // fill jacobian
        const vector<unsigned> & indices = f_index.getValue();
        unsigned i = 0, j = 0;
        while( i < numBlocks )
        {
            jacobian.beginBlockRow(i);
            if( j!=indices.size() && i==indices[j] )  { jacobian.createBlock(i,getBlock(j)); j++; }  // constrained particle
            else jacobian.createBlock(i,bIdentity);         // unconstrained particle: set diagonal to identity block
            jacobian.endBlockRow();   // only one block to create
            i++;
        }
        jacobian.compress();

        SparseMatrix J;
        J.copy(jacobian, M->colSize(), offset); // projection matrix for an assembled state
        BaseSparseMatrix* E = dynamic_cast<BaseSparseMatrix*>(M);
        assert(E);
        E->compressedMatrix = J.compressedMatrix * E->compressedMatrix * J.compressedMatrix;
    }


protected:
    RigidConstraint()
        : core::behavior::ProjectiveConstraintSet<DataTypes>(NULL)
        , f_index( initData(&f_index,"indices","Indices of the constrained frames") )
        , _drawSize( initData(&_drawSize,0.0,"drawSize","0 -> point based rendering, >0 -> radius of spheres") )
    {

    }

    virtual ~RigidConstraint()  {    }

    virtual void draw(const core::visual::VisualParams* vparams)
    {
        if (!vparams->displayFlags().getShowBehaviorModels()) return;
        if (!this->isActive()) return;
        const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
        const vector<unsigned> & indices = f_index.getValue();
        std::vector< Vector3 > points;
        for (vector<unsigned>::const_iterator it = indices.begin(); it != indices.end(); ++it) points.push_back(DataTypes::getCPos(x[*it]));
        if( _drawSize.getValue() == 0)  vparams->drawTool()->drawPoints(points, 10, Vec<4,float>(1,0.0,0.5,1)); // old classical drawing by points
        else  vparams->drawTool()->drawSpheres(points, (float)_drawSize.getValue(), Vec<4,float>(1.0f,0.0f,0.35f,1.0f)); // new drawing by spheres
    }


};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(FLEXIBLE_RigidConstraint_CPP)
extern template class SOFA_Flexible_API RigidConstraint<Affine3dTypes>;
extern template class SOFA_Flexible_API RigidConstraint<Quadratic3dTypes>;
extern template class SOFA_Flexible_API RigidConstraint<Affine3fTypes>;
extern template class SOFA_Flexible_API RigidConstraint<Quadratic3fTypes>;
#endif

}

}

}

#endif
