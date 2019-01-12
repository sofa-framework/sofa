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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PointConstraint_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_PointConstraint_INL

#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBoundaryCondition/PointConstraint.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <iostream>
#include <SofaBaseTopology/TopologySubsetData.inl>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{



template <class DataTypes>
PointConstraint<DataTypes>::PointConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(NULL)
    , f_indices( initData(&f_indices,"indices","Indices of the fixed points") )
    , _drawSize( initData(&_drawSize,(SReal)0.0,"drawSize","0 -> point based rendering, >0 -> radius of spheres") )
{
}


template <class DataTypes>
PointConstraint<DataTypes>::~PointConstraint()
{
}


// -- Constraint interface


template <class DataTypes>
void PointConstraint<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();
}

/// Update and return the jacobian. @todo update it when needed using topological engines instead of recomputing it at each call.
template <class DataTypes>
const sofa::defaulttype::BaseMatrix*  PointConstraint<DataTypes>::getJ(const core::MechanicalParams* )
{
    unsigned numBlocks = this->mstate->getSize();
    unsigned blockSize = DataTypes::deriv_total_size;
    jacobian.resize( numBlocks*blockSize,numBlocks*blockSize );

    for(unsigned i=0; i<numBlocks*blockSize; i++ )
    {
        jacobian.set(i,i,1);
    }
    for(unsigned i=0; i<f_indices.getValue().size(); i++ )
    {
        for(unsigned j=0; j<blockSize; j++)
            jacobian.set( f_indices.getValue()[i]*blockSize+j, f_indices.getValue()[i]*blockSize+j, 0);
        msg_info("PointConstraint<DataTypes>")<<"getJ, , set null block in " << f_indices.getValue()[i] << ", matrix after = " << jacobian ;
    }

    return &jacobian;
}


template <class DataTypes>
void PointConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    helper::WriteAccessor<DataVecDeriv> res ( mparams, resData );
    const SetIndexArray & indices = f_indices.getValue(mparams);
    for (unsigned int indice : indices)
    {
        res[indice] = Deriv();
    }
}

template <class DataTypes>
void PointConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData)
{
    helper::WriteAccessor<DataMatrixDeriv> c ( mparams, cData );
    const SetIndexArray & indices = f_indices.getValue(mparams);

    MatrixDerivRowIterator rowIt = c->begin();
    MatrixDerivRowIterator rowItEnd = c->end();

    while (rowIt != rowItEnd)
    {
        for (unsigned int indice : indices)
        {
            rowIt.row().erase(indice);
        }
        ++rowIt;
    }
}

template <class DataTypes>
void PointConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* /*mparams*/, DataVecDeriv& /*vData*/)
{
}

template <class DataTypes>
void PointConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& /*xData*/)
{
}

template <class DataTypes>
void PointConstraint<DataTypes>::applyConstraint(defaulttype::BaseMatrix *mat, unsigned int offset)
{
    const unsigned int N = Deriv::size();
    const SetIndexArray & indices = f_indices.getValue();

    for (unsigned int indice : indices)
    {
        // Reset Fixed Row and Col
        for (unsigned int c=0; c<N; ++c)
            mat->clearRowCol(offset + N * indice + c);
        // Set Fixed Vertex
        for (unsigned int c=0; c<N; ++c)
            mat->set(offset + N * indice + c, offset + N * indice + c, 1.0);
    }
}

template <class DataTypes>
void PointConstraint<DataTypes>::applyConstraint(defaulttype::BaseVector *vect, unsigned int offset)
{
    const unsigned int N = Deriv::size();

    const SetIndexArray & indices = f_indices.getValue();
    for (unsigned int indice : indices)
    {
        for (unsigned int c=0; c<N; ++c)
            vect->clear(offset + N * indice + c);
    }
}




template <class DataTypes>
void PointConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    if (!this->isActive()) return;

    vparams->drawTool()->saveLastState();

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    const SetIndexArray & indices = f_indices.getValue();

    if( _drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector< sofa::defaulttype::Vector3 > points;
        sofa::defaulttype::Vector3 point;
        for (unsigned int indice : indices)
        {
            point = DataTypes::getCPos(x[indice]);
            points.push_back(point);
        }
        vparams->drawTool()->drawPoints(points, 10, sofa::defaulttype::Vec<4,float>(1,0.5,0.5,1));
    }
    else // new drawing by spheres
    {
        std::vector< sofa::defaulttype::Vector3 > points;
        sofa::defaulttype::Vector3 point;
        for (unsigned int indice : indices)
        {
            point = DataTypes::getCPos(x[indice]);
            points.push_back(point);
        }
        vparams->drawTool()->drawSpheres(points, (float)_drawSize.getValue(), sofa::defaulttype::Vec<4,float>(1.0f,0.35f,0.35f,1.0f));
    }
    vparams->drawTool()->restoreLastState();

}


} // namespace constraint

} // namespace component

} // namespace sofa

#endif


