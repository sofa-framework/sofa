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

#include <sofa/component/constraint/projective/FixedProjectiveConstraint.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/vector_algorithm.h>

using sofa::core::objectmodel::ComponentState;


namespace sofa::component::constraint::projective
{

template <class DataTypes>
FixedProjectiveConstraint<DataTypes>::FixedProjectiveConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(nullptr)
    , d_indices( initData(&d_indices,"indices","Indices of the fixed points") )
    , d_fixAll( initData(&d_fixAll,false,"fixAll","filter all the DOF to implement a fixed object") )
    , d_showObject(initData(&d_showObject,true,"showObject","draw or not the fixed constraints"))
    , d_drawSize( initData(&d_drawSize,(SReal)0.0,"drawSize","Size of the rendered particles (0 -> point based rendering, >0 -> radius of spheres)") )
    , d_projectVelocity( initData(&d_projectVelocity,false,"activate_projectVelocity","if true, projects not only a constant but a zero velocity") )
    , l_topology(initLink("topology", "link to the topology container"))
    , data(std::make_unique<FixedProjectiveConstraintInternalData<DataTypes>>())
{
    // default to indice 0
    d_indices.beginEdit()->push_back(0);
    d_indices.endEdit();

    this->addUpdateCallback("updateIndices", { &d_indices}, [this](const core::DataTracker& t)
    {
        SOFA_UNUSED(t);
        checkIndices();
        return sofa::core::objectmodel::ComponentState::Valid;
    }, {});
}


template <class DataTypes>
FixedProjectiveConstraint<DataTypes>::~FixedProjectiveConstraint()
{
}

template <class DataTypes>
void FixedProjectiveConstraint<DataTypes>::clearConstraints()
{
    d_indices.beginEdit()->clear();
    d_indices.endEdit();
}

template <class DataTypes>
void FixedProjectiveConstraint<DataTypes>::addConstraint(Index index)
{
    d_indices.beginEdit()->push_back(index);
    d_indices.endEdit();
}

template <class DataTypes>
void FixedProjectiveConstraint<DataTypes>::removeConstraint(Index index)
{
    sofa::type::removeValue(*d_indices.beginEdit(),index);
    d_indices.endEdit();
}

// -- Constraint interface


template <class DataTypes>
void FixedProjectiveConstraint<DataTypes>::init()
{
    this->d_componentState.setValue(ComponentState::Invalid);
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();

    if (!this->mstate.get())
    {
        msg_warning() << "Missing mstate, cannot initialize the component.";
        return;
    }

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
        msg_info() << "Can not find the topology, won't be able to handle topological changes";
    }

    this->checkIndices();
    this->d_componentState.setValue(ComponentState::Valid);
}

template <class DataTypes>
void  FixedProjectiveConstraint<DataTypes>::reinit()
{
    this->checkIndices();
}

template <class DataTypes>
void  FixedProjectiveConstraint<DataTypes>::checkIndices()
{
    // Check value of given indices
    Index maxIndex=this->mstate->getSize();

    const SetIndexArray & indices = d_indices.getValue();
    SetIndexArray invalidIndices;
    for (unsigned int i=0; i<indices.size(); ++i)
    {
        const Index index=indices[i];
        if (index >= maxIndex)
        {
            msg_warning() << "Index " << index << " not valid, should be [0,"<< maxIndex <<"]. Constraint will be removed.";
            invalidIndices.push_back(index);
        }
    }

    // if invalid indices, sort them and remove in decreasing order as removeConstraint perform a swap and pop_back.
    if (!invalidIndices.empty())
    {
        std::sort( invalidIndices.begin(), invalidIndices.end(), std::greater<Index>() );
        const int max = invalidIndices.size()-1;
        for (int i=max; i>= 0; i--)
        {
            removeConstraint(invalidIndices[i]);
        }
    }
}

template <class DataTypes>
void FixedProjectiveConstraint<DataTypes>::projectMatrix( sofa::linearalgebra::BaseMatrix* M, unsigned offset )
{
    static const unsigned blockSize = DataTypes::deriv_total_size;

    if( d_fixAll.getValue() )
    {
        const unsigned size = this->mstate->getSize();
        for( unsigned i=0; i<size; i++ )
        {
            M->clearRowsCols( offset + i * blockSize, offset + (i+1) * (blockSize) );
        }
    }
    else
    {
        // clears the rows and columns associated with fixed particles
        for (const auto id : d_indices.getValue())
        {
            M->clearRowsCols( offset + id * blockSize, offset + (id+1) * blockSize );
        }
    }
}


template <class DataTypes>
void FixedProjectiveConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    SOFA_UNUSED(mparams);

    helper::WriteAccessor<DataVecDeriv> res (resData );
    const SetIndexArray & indices = d_indices.getValue();

    if( d_fixAll.getValue() )
    {
        // fix everything
        typename VecDeriv::iterator it;
        for( it = res.begin(); it != res.end(); ++it )
        {
            *it = Deriv();
        }
    }
    else
    {
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            res[*it] = Deriv();
        }
    }
}

template <class DataTypes>
void FixedProjectiveConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData)
{
    SOFA_UNUSED(mparams);

    helper::WriteAccessor<DataMatrixDeriv> c (cData );

    if( d_fixAll.getValue() )
    {
        // fix everything
        c->clear();
    }
    else
    {
        const SetIndexArray& indices = d_indices.getValue();
        for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
        {
            c->clearColBlock(*it);
        }
    }
}

// projectVelocity applies the same changes on velocity vector as projectResponse on position vector :
// Each fixed point received a null velocity vector.
// When a new fixed point is added while its velocity vector is already null, projectVelocity is not useful.
// But when a new fixed point is added while its velocity vector is not null, it's necessary to fix it to null or 
// to set the projectVelocity option to True. If not, the fixed point is going to drift.
template <class DataTypes>
void FixedProjectiveConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vData)
{
    SOFA_UNUSED(mparams);

    if(!d_projectVelocity.getValue()) return;

    helper::WriteAccessor<DataVecDeriv> res (vData );

    if ( d_fixAll.getValue() )    // fix everything
    {
        for(Size i=0; i<res.size(); i++)
            res[i] = Deriv();
    }
    else
    {
        const SetIndexArray & indices = this->d_indices.getValue();
        for(Index ind : indices)
        {
            res[ind] = Deriv();
        }
    }
}


template <class DataTypes>
void FixedProjectiveConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/, DataVecCoord& /*xData*/)
{

}

// Matrix Integration interface
template <class DataTypes>
void FixedProjectiveConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    SOFA_UNUSED(mparams);
    if(const core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate.get()))
    {
        const unsigned int N = Deriv::size();

        if( d_fixAll.getValue() )
        {
            const unsigned size = this->mstate->getSize();
            for(unsigned int i=0; i<size; i++)
            {
                // Reset Fixed Row and Col
                for (unsigned int c=0; c<N; ++c)
                    r.matrix->clearRowCol(r.offset + N * i + c);
                // Set Fixed Vertex
                for (unsigned int c=0; c<N; ++c)
                    r.matrix->set(r.offset + N * i + c, r.offset + N * i + c, 1.0);
            }
        }
        else
        {
            const SetIndexArray & indices = d_indices.getValue();

            for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
            {
                // Reset Fixed Row and Col
                for (unsigned int c=0; c<N; ++c)
                    r.matrix->clearRowCol(r.offset + N * (*it) + c);
                // Set Fixed Vertex
                for (unsigned int c=0; c<N; ++c)
                    r.matrix->set(r.offset + N * (*it) + c, r.offset + N * (*it) + c, 1.0);
            }
        }
    }
}

template <class DataTypes>
void FixedProjectiveConstraint<DataTypes>::applyConstraint(const core::MechanicalParams* mparams, linearalgebra::BaseVector* vect, const sofa::core::behavior::MultiMatrixAccessor* matrix)
{
    SOFA_UNUSED(mparams);
    const int o = matrix->getGlobalOffset(this->mstate.get());
    if (o >= 0)
    {
        const unsigned int offset = (unsigned int)o;
        const unsigned int N = Deriv::size();

        if( d_fixAll.getValue() )
        {
            for(sofa::Size i=0; i < (sofa::Size) vect->size(); i++ )
            {
                for (unsigned int c=0; c<N; ++c)
                    vect->clear(offset + N * i + c);
            }
        }
        else
        {
            const SetIndexArray & indices = d_indices.getValue();
            for (const auto & index : indices)
            {
                for (unsigned int c=0; c<N; ++c)
                    vect->clear(offset + N * index + c);
            }
        }
    }
}

template <class DataTypes>
void FixedProjectiveConstraint<DataTypes>::applyConstraint(sofa::core::behavior::ZeroDirichletCondition* matrix)
{
    static constexpr unsigned int N = Deriv::size();

    if( d_fixAll.getValue() )
    {
        const sofa::Size size = this->mstate->getMatrixSize();
        for(sofa::Index i = 0; i < size; ++i)
        {
            matrix->discardRowCol(i, i);
        }
    }
    else
    {
        const SetIndexArray & indices = d_indices.getValue();

        for (const auto index : indices)
        {
            for (unsigned int c = 0; c < N; ++c)
            {
                matrix->discardRowCol(N * index + c, N * index + c);
            }
        }
    }
}

template <class DataTypes>
void FixedProjectiveConstraint<DataTypes>::computeBBoxForIndices(const type::vector<Index>& indices)
{
    using Real = typename DataTypes::Real;

    const auto drawSize = static_cast<Real>(d_drawSize.getValue());

    const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();

    type::BoundingBox bbox;
    for (const auto index : indices )
    {
        const auto x3d = DataTypes::getCPos(x[index]);

        for (unsigned int i = 0; i < DataTypes::Coord::spatial_dimensions && i<3; ++i)
        {
            bbox.maxBBox()[i] = std::max(static_cast<SReal>(x3d[i] + drawSize), bbox.maxBBox()[i]);
            bbox.minBBox()[i] = std::min(static_cast<SReal>(x3d[i] - drawSize), bbox.minBBox()[i]);
        }
    }

    this->f_bbox.setValue(bbox);
}

template <class DataTypes>
void FixedProjectiveConstraint<DataTypes>::computeBBox(
    const core::ExecParams* params, bool onlyVisible)
{
    SOFA_UNUSED(params);

    if( onlyVisible && !d_showObject.getValue() )
    {
        return;
    }

    if(this->d_componentState.getValue() == ComponentState::Invalid)
    {
        return;
    }

    const auto& indices = d_indices.getValue();

    if (d_fixAll.getValue())
    {
        const auto bbox = this->mstate->computeBBox(); //this may compute twice the mstate bbox, but there is no way to determine if the bbox has already been computed
        this->f_bbox.setValue(std::move(bbox));
    }
    else if (!indices.empty())
    {
        computeBBoxForIndices(indices);
    }
}

template <class DataTypes>
void FixedProjectiveConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (this->d_componentState.getValue() != ComponentState::Valid) return;
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    if (!d_showObject.getValue()) return;
    if (!this->isActive()) return;

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();
    const SetIndexArray & indices = d_indices.getValue();

    if( d_drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector< sofa::type::Vec3 > points;

        if (d_fixAll.getValue())
        {
            for (unsigned i = 0; i < x.size(); i++)
            {
                const type::Vec3 point = type::toVec3(DataTypes::getCPos(x[i]));
                points.push_back(point);
            }
        }
        else
        {
            for (const auto index : indices)
            {
                const type::Vec3 point = type::toVec3(DataTypes::getCPos(x[index]));
                points.push_back(point);
            }
        }
        vparams->drawTool()->drawPoints(points, 10, sofa::type::RGBAColor(1,0.5,0.5,1));
    }
    else // new drawing by spheres
    {
        vparams->drawTool()->setLightingEnabled(true);

        std::vector< sofa::type::Vec3 > points;
        sofa::type::Vec3 point;
        if( d_fixAll.getValue()==true )
        {
            for (unsigned i=0; i<x.size(); i++ )
            {
                const type::Vec3 point = type::toVec3(DataTypes::getCPos(x[i]));
                points.push_back(point);
            }
        }
        else
        {
            for (const auto index : indices)
            {
                const type::Vec3 point = type::toVec3(DataTypes::getCPos(x[index]));
                points.push_back(point);
            }
        }
        vparams->drawTool()->drawSpheres(points, (float)d_drawSize.getValue(), sofa::type::RGBAColor(1.0f,0.35f,0.35f,1.0f));
    }


}

} // namespace sofa::component::constraint::projective
