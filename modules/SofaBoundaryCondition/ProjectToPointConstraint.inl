/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectToPointConstraint_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectToPointConstraint_INL

#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBoundaryCondition/ProjectToPointConstraint.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/helper/gl/template.h>
//#include <sofa/defaulttype/RigidTypes.h>
#include <iostream>
#include <SofaBaseTopology/TopologySubsetData.inl>


#include <sofa/helper/gl/BasicShapes.h>

namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

// Define TestNewPointFunction
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

// Define RemovalFunction
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
    : core::behavior::ProjectiveConstraintSet<DataTypes>(NULL)
    , f_indices( initData(&f_indices,"indices","Indices of the points to project") )
    , f_point( initData(&f_point,"point","Target of the projection") )
    , f_fixAll( initData(&f_fixAll,false,"fixAll","filter all the DOF to implement a fixed object") )
    , f_drawSize( initData(&f_drawSize,(SReal)0.0,"drawSize","0 -> point based rendering, >0 -> radius of spheres") )
    , data(new ProjectToPointConstraintInternalData<DataTypes>())
{
    // default to indice 0
    f_indices.beginEdit()->push_back(0);
    f_indices.endEdit();

    pointHandler = new FCPointHandler(this, &f_indices);
}


template <class DataTypes>
ProjectToPointConstraint<DataTypes>::~ProjectToPointConstraint()
{
    if (pointHandler)
        delete pointHandler;

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

    topology = this->getContext()->getMeshTopology();

    //  if (!topology)
    //    serr << "Can not find the topology." << sendl;

    // Initialize functions and parameters
    f_indices.createTopologicalEngine(topology, pointHandler);
    f_indices.registerTopologicalData();

    const SetIndexArray & indices = f_indices.getValue();

    unsigned int maxIndex=this->mstate->getSize();
    for (unsigned int i=0; i<indices.size(); ++i)
    {
        const unsigned int index=indices[i];
        if (index >= maxIndex)
        {
            serr << "Index " << index << " not valid!" << sendl;
            removeConstraint(index);
        }
    }

    reinit();
}

template <class DataTypes>
void  ProjectToPointConstraint<DataTypes>::reinit()
{

    // get the indices sorted
    SetIndexArray tmp = f_indices.getValue();
    std::sort(tmp.begin(),tmp.end());

//    // resize the jacobian
//    unsigned numBlocks = this->mstate->getSize();
//    unsigned blockSize = DataTypes::deriv_total_size;
//    jacobian.resize( numBlocks*blockSize,numBlocks*blockSize );

//    // fill the jacobian is ascending order
//    SetIndexArray::const_iterator it= tmp.begin();
//    unsigned i=0;
//    for(SetIndexArray::const_iterator it= tmp.begin(); i<numBlocks && it!=tmp.end(); i++ )
//    {
//        if( i==*it )  // constrained particle: set diagonal to 0, and move the cursor to the next constraint
//        {
//            it++;
//            for( unsigned j=0; j<blockSize; j++ )
//            {
//                jacobian.beginRow(blockSize*i+j );
//                jacobian.set( blockSize*i+j, blockSize*i+j, 0); // constrained particle: set the diagonal to
//            }
//        }
//        else
//            for( unsigned j=0; j<blockSize; j++ )
//            {
//                jacobian.beginRow(blockSize*i+j );
//                jacobian.set( blockSize*i+j, blockSize*i+j, 1); // unconstrained particle: set the diagonal to identity
//            }
//    }
//    // Set the matrix to identity beyond the last constrained particle
//    for(; i<numBlocks && it!=tmp.end(); i++ )
//    {
//        for( unsigned j=0; j<blockSize; j++ )
//        {
//            jacobian.beginRow( blockSize*i+j );
//            jacobian.set( blockSize*i+j, blockSize*i+j, 1);
//        }
//    }
//    jacobian.compress();

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



///// Update and return the jacobian. @todo update it when needed using topological engines instead of recomputing it at each call.
//template <class DataTypes>
//const sofa::defaulttype::BaseMatrix*  ProjectToPointConstraint<DataTypes>::getJ(const core::MechanicalParams* )
//{
//    return &jacobian;
//}


template <class DataTypes>
void ProjectToPointConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    helper::WriteAccessor<DataVecDeriv> res ( mparams, resData );
    const SetIndexArray & indices = f_indices.getValue(mparams);
    if( f_fixAll.getValue(mparams) )
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
    helper::WriteAccessor<DataMatrixDeriv> c ( mparams, cData );
    const SetIndexArray & indices = f_indices.getValue(mparams);

    MatrixDerivRowIterator rowIt = c->begin();
    MatrixDerivRowIterator rowItEnd = c->end();

    if( f_fixAll.getValue(mparams) )
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
    helper::WriteAccessor<DataVecCoord> res ( mparams, xData );
    const SetIndexArray & indices = f_indices.getValue(mparams);
    if( f_fixAll.getValue(mparams) )
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

// Matrix Integration interface
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
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    if (!this->isActive()) return;
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const SetIndexArray & indices = f_indices.getValue();

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
        glColor4f (1.0f,0.35f,0.35f,1.0f);
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
#endif /* SOFA_NO_OPENGL */
}

//// Specialization for rigids
//#ifndef SOFA_FLOAT
//template <>
//void ProjectToPointConstraint<Rigid3dTypes >::draw(const core::visual::VisualParams* vparams);
//template <>
//void ProjectToPointConstraint<Rigid2dTypes >::draw(const core::visual::VisualParams* vparams);
//#endif
//#ifndef SOFA_DOUBLE
//template <>
//void ProjectToPointConstraint<Rigid3fTypes >::draw(const core::visual::VisualParams* vparams);
//template <>
//void ProjectToPointConstraint<Rigid2fTypes >::draw(const core::visual::VisualParams* vparams);
//#endif



} // namespace constraint

} // namespace component

} // namespace sofa

#endif


