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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectToPointConstraint_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectToPointConstraint_INL

#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBoundaryCondition/ProjectToPointConstraint.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Simulation.h>
#include <iostream>
#include <SofaBaseTopology/TopologySubsetData.inl>


namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

template< class DataTypes>
bool ProjectToPointConstraint<DataTypes>::FCPointHandler::applyTestCreateFunction(unsigned int, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
{
    if (fc)
    {
        return false;
    }
    else
    {
        return false;
    }
}

template< class DataTypes>
void ProjectToPointConstraint<DataTypes>::FCPointHandler::applyDestroyFunction(unsigned int pointIndex, core::objectmodel::Data<value_type> &)
{
    if (fc)
    {
        fc->removeConstraint((unsigned int) pointIndex);
    }
}

template <class DataTypes>
ProjectToPointConstraint<DataTypes>::ProjectToPointConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(nullptr)
    , f_indices( initData(&f_indices,"indices","Indices of the points to project") )
    , f_point( initData(&f_point,"point","Target of the projection") )
    , f_fixAll( initData(&f_fixAll,false,"fixAll","filter all the DOF to implement a fixed object") )
    , f_drawSize( initData(&f_drawSize,(SReal)0.0,"drawSize","0 -> point based rendering, >0 -> radius of spheres") )
    , l_topology(initLink("topology", "link to the topology container"))
    , data(new ProjectToPointConstraintInternalData<DataTypes>())    
    , m_pointHandler(nullptr)
{
    f_indices.beginEdit()->push_back(0);
    f_indices.endEdit();
}


template <class DataTypes>
ProjectToPointConstraint<DataTypes>::~ProjectToPointConstraint()
{
    if (m_pointHandler)
        delete m_pointHandler;

    delete data;
}

template <class DataTypes>
void ProjectToPointConstraint<DataTypes>::clearConstraints()
{
    f_indices.beginEdit()->clear();
    f_indices.endEdit();
}

template <class DataTypes>
void ProjectToPointConstraint<DataTypes>::addConstraint(unsigned int index)
{
    f_indices.beginEdit()->push_back(index);
    f_indices.endEdit();
}

template <class DataTypes>
void ProjectToPointConstraint<DataTypes>::removeConstraint(unsigned int index)
{
    removeValue(*f_indices.beginEdit(),index);
    f_indices.endEdit();
}

// -- Constraint interface


template <class DataTypes>
void ProjectToPointConstraint<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();

    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }

    sofa::core::topology::BaseMeshTopology* _topology = l_topology.get();

    if (_topology)
    {
        msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

        // Initialize functions and parameters
        m_pointHandler = new FCPointHandler(this, &f_indices);
        f_indices.createTopologicalEngine(_topology, m_pointHandler);
        f_indices.registerTopologicalData();
    }
    else
    {
        msg_info() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
    }

    const SetIndexArray & indices = f_indices.getValue();

    std::stringstream sstream;
    unsigned int maxIndex=this->mstate->getSize();
    for (unsigned int i=0; i<indices.size(); ++i)
    {
        const unsigned int index=indices[i];
        if (index >= maxIndex)
        {
            sstream << "Index " << index << " not valid!\n";
            removeConstraint(index);
        }
    }
    msg_error_when(!sstream.str().empty()) << sstream.str();

    reinit();
}

template <class DataTypes>
void  ProjectToPointConstraint<DataTypes>::reinit()
{

    // get the indices sorted
    SetIndexArray tmp = f_indices.getValue();
    std::sort(tmp.begin(),tmp.end());
}

template <class DataTypes>
void ProjectToPointConstraint<DataTypes>::projectMatrix( sofa::defaulttype::BaseMatrix* M, unsigned offset )
{
    unsigned blockSize = DataTypes::deriv_total_size;

    // clears the rows and columns associated with fixed particles
    for(SetIndexArray::const_iterator it= f_indices.getValue().begin(), iend=f_indices.getValue().end(); it!=iend; it++ )
    {
        M->clearRowsCols( offset + (*it) * blockSize, offset + (*it+1) * (blockSize) );
    }
}

template <class DataTypes>
void ProjectToPointConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    SOFA_UNUSED(mparams);

    helper::WriteAccessor<DataVecDeriv> res ( resData );
    const SetIndexArray & indices = f_indices.getValue();
    if( f_fixAll.getValue() )
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
        for (SetIndexArray::const_iterator it = indices.begin();
                it != indices.end();
                ++it)
        {
            res[*it] = Deriv();
        }
    }
}

template <class DataTypes>
void ProjectToPointConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* mparams, DataMatrixDeriv& cData)
{
    SOFA_UNUSED(mparams);

    helper::WriteAccessor<DataMatrixDeriv> c ( cData );
    const SetIndexArray & indices = f_indices.getValue();

    MatrixDerivRowIterator rowIt = c->begin();
    MatrixDerivRowIterator rowItEnd = c->end();

    if( f_fixAll.getValue() )
    {
        // fix everything
        while (rowIt != rowItEnd)
        {
            rowIt.row().clear();
            ++rowIt;
        }
    }
    else
    {
        while (rowIt != rowItEnd)
        {
            for (SetIndexArray::const_iterator it = indices.begin();
                    it != indices.end();
                    ++it)
            {
                rowIt.row().erase(*it);
            }
            ++rowIt;
        }
    }
}

template <class DataTypes>
void ProjectToPointConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* mparams , DataVecDeriv& vData)
{
    projectResponse(mparams, vData);
}

template <class DataTypes>
void ProjectToPointConstraint<DataTypes>::projectPosition(const core::MechanicalParams* mparams, DataVecCoord& xData)
{
    SOFA_UNUSED(mparams);

    helper::WriteAccessor<DataVecCoord> res ( xData );
    const SetIndexArray & indices = f_indices.getValue();
    if( f_fixAll.getValue() )
    {
        // fix everything
        typename VecCoord::iterator it;
        for( it = res.begin(); it != res.end(); ++it )
        {
            *it = f_point.getValue();
        }
    }
    else
    {
        for (SetIndexArray::const_iterator it = indices.begin();
                it != indices.end();
                ++it)
        {
            res[*it] = f_point.getValue();
        }
    }
}

template <class DataTypes>
void ProjectToPointConstraint<DataTypes>::applyConstraint(defaulttype::BaseMatrix *mat, unsigned int offset)
{
    const unsigned int N = Deriv::size();
    const SetIndexArray & indices = f_indices.getValue();

    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        // Reset Fixed Row and Col
        for (unsigned int c=0; c<N; ++c)
            mat->clearRowCol(offset + N * (*it) + c);
        // Set Fixed Vertex
        for (unsigned int c=0; c<N; ++c)
            mat->set(offset + N * (*it) + c, offset + N * (*it) + c, 1.0);
    }
}

template <class DataTypes>
void ProjectToPointConstraint<DataTypes>::applyConstraint(defaulttype::BaseVector *vect, unsigned int offset)
{
    const unsigned int N = Deriv::size();

    const SetIndexArray & indices = f_indices.getValue();
    for (SetIndexArray::const_iterator it = indices.begin(); it != indices.end(); ++it)
    {
        for (unsigned int c=0; c<N; ++c)
            vect->clear(offset + N * (*it) + c);
    }
}




template <class DataTypes>
void ProjectToPointConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    if (!this->isActive()) return;
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const SetIndexArray & indices = f_indices.getValue();

    vparams->drawTool()->saveLastState();

    if( f_drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector< sofa::defaulttype::Vector3 > points;
        sofa::defaulttype::Vector3 point;
        if( f_fixAll.getValue() )
            for (unsigned i=0; i<x.size(); i++ )
            {
                point = DataTypes::getCPos(x[i]);
                points.push_back(point);
            }
        else
            for (SetIndexArray::const_iterator it = indices.begin();
                    it != indices.end();
                    ++it)
            {
                point = DataTypes::getCPos(x[*it]);
                points.push_back(point);
            }
        vparams->drawTool()->drawPoints(points, 10, sofa::defaulttype::Vec<4,float>(1,0.5,0.5,1));
    }
    else // new drawing by spheres
    {
        std::vector< sofa::defaulttype::Vector3 > points;
        sofa::defaulttype::Vector3 point;
        if( f_fixAll.getValue()==true )
            for (unsigned i=0; i<x.size(); i++ )
            {
                point = DataTypes::getCPos(x[i]);
                points.push_back(point);
            }
        else
            for (SetIndexArray::const_iterator it = indices.begin();
                    it != indices.end();
                    ++it)
            {
                point = DataTypes::getCPos(x[*it]);
                points.push_back(point);
            }
        vparams->drawTool()->drawSpheres(points, (float)f_drawSize.getValue(), sofa::defaulttype::Vec<4,float>(1.0f,0.35f,0.35f,1.0f));
    }

    vparams->drawTool()->restoreLastState();
}

} // namespace constraint

} // namespace component

} // namespace sofa

#endif


