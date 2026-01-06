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

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/component/constraint/projective/DirectionProjectiveConstraint.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Simulation.h>
#include <iostream>
#include <sofa/type/vector_algorithm.h>


namespace sofa::component::constraint::projective
{

template <class DataTypes>
DirectionProjectiveConstraint<DataTypes>::DirectionProjectiveConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(nullptr)
    , d_indices( initData(&d_indices,"indices","Indices the particles to project") )
    , d_drawSize( initData(&d_drawSize,(SReal)0.0,"drawSize","Size of the rendered particles (0 -> point based rendering, >0 -> radius of spheres)") )
    , d_direction( initData(&d_direction,CPos(),"direction","Direction of the line"))
    , l_topology(initLink("topology", "link to the topology container"))
    , data(std::make_unique<DirectionProjectiveConstraintInternalData<DataTypes>>())
{
    d_indices.beginEdit()->push_back(0);
    d_indices.endEdit();
}


template <class DataTypes>
DirectionProjectiveConstraint<DataTypes>::~DirectionProjectiveConstraint()
{
}

template <class DataTypes>
void DirectionProjectiveConstraint<DataTypes>::clearConstraints()
{
    d_indices.beginEdit()->clear();
    d_indices.endEdit();
}

template <class DataTypes>
void DirectionProjectiveConstraint<DataTypes>::addConstraint(Index index)
{
    d_indices.beginEdit()->push_back(index);
    d_indices.endEdit();
}

template <class DataTypes>
void DirectionProjectiveConstraint<DataTypes>::removeConstraint(Index index)
{
    sofa::type::removeValue(*d_indices.beginEdit(), index);
    d_indices.endEdit();
}

// -- Constraint interface


template <class DataTypes>
void DirectionProjectiveConstraint<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }


    if (sofa::core::topology::BaseMeshTopology* _topology = l_topology.get())
    {
        msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";
        
        // Initialize topological changes support
        d_indices.createTopologyHandler(_topology);
    }
    else
    {
        msg_info() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
    }

    const Indices & indices = d_indices.getValue();

    const Index maxIndex=this->mstate->getSize();
    for (unsigned int i=0; i<indices.size(); ++i)
    {
        const Index index=indices[i];
        if (index >= maxIndex)
        {
            msg_error() << "Index " << index << " not valid!";
            removeConstraint(index);
        }
    }

    reinit();
}

template <class DataTypes>
void  DirectionProjectiveConstraint<DataTypes>::reinit()
{
    // normalize the normal vector
    CPos n = d_direction.getValue();
    if( n.norm()==0 )
        n[1]=0;
    else n *= 1/n.norm();
    d_direction.setValue(n);

    // create the matrix blocks corresponding to the projection to the line: nn^t or to the identity
    Block bProjection;
    for(unsigned i=0; i<bsize; i++)
        for(unsigned j=0; j<bsize; j++)
        {
            bProjection(i,j) = n[i]*n[j];
        }

    // get the indices sorted
    Indices tmp = d_indices.getValue();
    std::sort(tmp.begin(),tmp.end());

    // resize the jacobian
    const unsigned numBlocks = this->mstate->getSize();
    const unsigned blockSize = DataTypes::deriv_total_size;
    jacobian.resize( numBlocks*blockSize,numBlocks*blockSize );

    // fill the jacobian in ascending order
    Indices::const_iterator it = tmp.begin();
    unsigned i = 0;
    while( i < numBlocks )
    {
        if( it != tmp.end() && i==*it )  // constrained particle: set diagonal to projection block, and  the cursor to the next constraint
        {
            jacobian.insertBackBlock(i,i,bProjection);
            ++it;
        }
        else           // unconstrained particle: set diagonal to identity block
        {
            jacobian.insertBackBlock(i,i,Block::Identity());
        }
        i++;
    }
    jacobian.compress();

    const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();
    const Indices &indices = d_indices.getValue();
    for (const auto id : indices)
    {
        m_origin.push_back(DataTypes::getCPos(x[id]));
    }

}

template <class DataTypes>
void DirectionProjectiveConstraint<DataTypes>::projectMatrix( sofa::linearalgebra::BaseMatrix* M, unsigned offset )
{
    J.copy(jacobian, M->colSize(), offset); // projection matrix for an assembled state
    BaseSparseMatrix* E = dynamic_cast<BaseSparseMatrix*>(M);
    assert(E);
    E->compressedMatrix = J.compressedMatrix * E->compressedMatrix * J.compressedMatrix;
}



template <class DataTypes>
void DirectionProjectiveConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    SOFA_UNUSED(mparams);

    helper::WriteAccessor<DataVecDeriv> res ( resData );
    jacobian.mult(res.wref(),res.ref());
}

template <class DataTypes>
void DirectionProjectiveConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* /*mparams*/ , DataMatrixDeriv& /*cData*/)
{
    msg_error() << "projectJacobianMatrix(const core::MechanicalParams*, DataMatrixDeriv& ) is not implemented";
}

template <class DataTypes>
void DirectionProjectiveConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData)
{
    projectResponse(mparams,vData);
}

template <class DataTypes>
void DirectionProjectiveConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/ , DataVecCoord& xData)
{
    VecCoord& x = *xData.beginEdit();

    const CPos& n = d_direction.getValue();

    const Indices& indices = d_indices.getValue();
    for(unsigned i=0; i<indices.size(); i++ )
    {
        // replace the point with its projection to the line

        const CPos xi = DataTypes::getCPos( x[indices[i]] );
        DataTypes::setCPos( x[indices[i]], m_origin[i] + n * ((xi-m_origin[i])*n) );
    }

    xData.endEdit();
}

template <class DataTypes>
void DirectionProjectiveConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* /*mparams*/, const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/)
{
    msg_error() << "applyConstraint is not implemented";
}

template <class DataTypes>
void DirectionProjectiveConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* /*mparams*/, linearalgebra::BaseVector* /*vector*/, const sofa::core::behavior::MultiMatrixAccessor* /*matrix*/)
{
    dmsg_error() << "DirectionProjectiveConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* mparams, linearalgebra::BaseVector* vector, const sofa::core::behavior::MultiMatrixAccessor* matrix) is not implemented";
}




template <class DataTypes>
void DirectionProjectiveConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    if (!this->isActive()) return;
    const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    const Indices & indices = d_indices.getValue();

    if(d_drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector< sofa::type::Vec3 > points;
        for (unsigned int index : indices)
        {
            const type::Vec3 point = type::toVec3(DataTypes::getCPos(x[index]));
            points.push_back(point);
        }
        vparams->drawTool()->drawPoints(points, 10, sofa::type::RGBAColor(1,0.5,0.5,1));
    }
    else // new drawing by spheres
    {
        std::vector< sofa::type::Vec3 > points;
        sofa::type::Vec3 point;
        for (unsigned int index : indices)
        {
            const type::Vec3 point = type::toVec3(DataTypes::getCPos(x[index]));
            points.push_back(point);
        }
        vparams->drawTool()->drawSpheres(points, (float)d_drawSize.getValue(), sofa::type::RGBAColor(1.0f, 0.35f, 0.35f, 1.0f));
    }


}

} // namespace sofa::component::constraint::projective
