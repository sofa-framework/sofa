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
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectToLineConstraint_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectToLineConstraint_INL

#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBoundaryCondition/ProjectToLineConstraint.h>
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
bool ProjectToLineConstraint<DataTypes>::FCPointHandler::applyTestCreateFunction(unsigned int, const sofa::helper::vector<unsigned int> &, const sofa::helper::vector<double> &)
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
void ProjectToLineConstraint<DataTypes>::FCPointHandler::applyDestroyFunction(unsigned int pointIndex, core::objectmodel::Data<value_type> &)
{
    if (fc)
    {
        fc->removeConstraint((unsigned int) pointIndex);
    }
}

template <class DataTypes>
ProjectToLineConstraint<DataTypes>::ProjectToLineConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(NULL)
    , f_indices( initData(&f_indices,"indices","Indices of the fixed points") )
    , f_drawSize( initData(&f_drawSize,(SReal)0.0,"drawSize","0 -> point based rendering, >0 -> radius of spheres") )
    , f_origin( initData(&f_origin,CPos(),"origin","A point in the line"))
    , f_direction( initData(&f_direction,CPos(),"direction","Direction of the line"))
    , data(new ProjectToLineConstraintInternalData<DataTypes>())
{
    // default to index 0
    f_indices.beginEdit()->push_back(0);
    f_indices.endEdit();

    pointHandler = new FCPointHandler(this, &f_indices);
}


template <class DataTypes>
ProjectToLineConstraint<DataTypes>::~ProjectToLineConstraint()
{
    if (pointHandler)
        delete pointHandler;

    delete data;
}

template <class DataTypes>
void ProjectToLineConstraint<DataTypes>::clearConstraints()
{
    f_indices.beginEdit()->clear();
    f_indices.endEdit();
}

template <class DataTypes>
void ProjectToLineConstraint<DataTypes>::addConstraint(unsigned int index)
{
    f_indices.beginEdit()->push_back(index);
    f_indices.endEdit();
}

template <class DataTypes>
void ProjectToLineConstraint<DataTypes>::removeConstraint(unsigned int index)
{
    removeValue(*f_indices.beginEdit(),index);
    f_indices.endEdit();
}

// -- Constraint interface


template <class DataTypes>
void ProjectToLineConstraint<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();

    topology = this->getContext()->getMeshTopology();

    //  if (!topology)
    //    serr << "Can not find the topology." << sendl;

    // Initialize functions and parameters
    f_indices.createTopologicalEngine(topology, pointHandler);
    f_indices.registerTopologicalData();

    const Indices & indices = f_indices.getValue();

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

//  cerr<<"ProjectToLineConstraint<DataTypes>::init(), getJ = " << *getJ(0) << endl;

}

template <class DataTypes>
void  ProjectToLineConstraint<DataTypes>::reinit()
{
//    cerr<<"ProjectToLineConstraint<DataTypes>::getJ, numblocs = "<< numBlocks << ", block size = " << blockSize << endl;

    // normalize the normal vector
    CPos n = f_direction.getValue();
    if( n.norm()==0 )
        n[0]=1; // arbritary normal vector
    else n *= 1/n.norm();
    f_direction.setValue(n);

    // create the matrix blocks corresponding to the projection to the line: nn^t or to the identity
    Block bProjection;
    for(unsigned i=0; i<bsize; i++)
        for(unsigned j=0; j<bsize; j++)
        {
            bProjection[i][j] = n[i]*n[j];
        }
//    cerr<<"ProjectToLineConstraint<DataTypes>::reinit() bProjection[0] = " << endl << bProjection[0] << endl;

    // get the indices sorted
    Indices tmp = f_indices.getValue();
    std::sort(tmp.begin(),tmp.end());

    // resize the jacobian
    unsigned numBlocks = this->mstate->getSize();
    unsigned blockSize = DataTypes::deriv_total_size;
    jacobian.resize( numBlocks*blockSize,numBlocks*blockSize );

    // fill the jacobian in ascending order
    Indices::const_iterator it = tmp.begin();
    unsigned i = 0;
    while( i < numBlocks ) // (FF) do not stop after the last constrained particle, for the remainder of the diagonal would be null, while it must be identity.
    {
        if( it!=tmp.end() && i==*it )  // constrained particle: set diagonal to projection block, and  the cursor to the next constraint
        {
            jacobian.insertBackBlock(i,i,bProjection);
            it++;
        }
        else           // unconstrained particle: set diagonal to identity block
        {
            jacobian.insertBackBlock(i,i,Block::s_identity);
        }
        i++;
    }
    jacobian.compress();
//    cerr<<"ProjectToLineConstraint<DataTypes>::reinit(), jacobian = " << jacobian << endl;

}

template <class DataTypes>
void ProjectToLineConstraint<DataTypes>::projectMatrix( sofa::defaulttype::BaseMatrix* M, unsigned offset )
{
    J.copy(jacobian, M->colSize(), offset); // projection matrix for an assembled state
    BaseSparseMatrix* E = dynamic_cast<BaseSparseMatrix*>(M);
    assert(E);
    E->compressedMatrix = J.compressedMatrix * E->compressedMatrix * J.compressedMatrix;
}



template <class DataTypes>
void ProjectToLineConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams, DataVecDeriv& resData)
{
    helper::WriteAccessor<DataVecDeriv> res ( mparams, resData );
    jacobian.mult(res.wref(),res.ref());
}

template <class DataTypes>
void ProjectToLineConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* /*mparams*/ , DataMatrixDeriv& /*cData*/)
{
    serr<<"projectJacobianMatrix(const core::MechanicalParams*, DataMatrixDeriv& ) is not implemented" << sendl;
}

template <class DataTypes>
void ProjectToLineConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vdata)
{
    projectResponse(mparams,vdata);
}

template <class DataTypes>
void ProjectToLineConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/ , DataVecCoord& xData)
{
    VecCoord& x = *xData.beginEdit();

    const CPos& n = f_direction.getValue();
    const CPos& o = f_origin.getValue();

    const Indices& indices = f_indices.getValue();
    for(unsigned i=0; i<indices.size(); i++ )
    {
        // replace the point with its projection to the line

        const CPos xi = DataTypes::getCPos( x[indices[i]] );
        DataTypes::setCPos( x[indices[i]], o + n * ((xi-o)*n) );

//        x[indices[i]] = o + n * ((x[indices[i]]-o)*n) );
    }

    xData.endEdit();
}

// Matrix Integration interface
template <class DataTypes>
void ProjectToLineConstraint<DataTypes>::applyConstraint(defaulttype::BaseMatrix * /*mat*/, unsigned int /*offset*/)
{
    serr << "applyConstraint is not implemented " << sendl;
}

template <class DataTypes>
void ProjectToLineConstraint<DataTypes>::applyConstraint(defaulttype::BaseVector * /*vect*/, unsigned int /*offset*/)
{
    serr<<"ProjectToLineConstraint<DataTypes>::applyConstraint(defaulttype::BaseVector *vect, unsigned int offset) is not implemented "<< sendl;
}




template <class DataTypes>
void ProjectToLineConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    if (!this->isActive()) return;
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    const Indices & indices = f_indices.getValue();

    if( f_drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector< sofa::defaulttype::Vector3 > points;
        sofa::defaulttype::Vector3 point;
        //serr<<"ProjectToLineConstraint<DataTypes>::draw(), indices = "<<indices<<sendl;
        for (Indices::const_iterator it = indices.begin();
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
        for (Indices::const_iterator it = indices.begin();
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
//    void ProjectToLineConstraint<Rigid3dTypes >::draw(const core::visual::VisualParams* vparams);
//template <>
//    void ProjectToLineConstraint<Rigid2dTypes >::draw(const core::visual::VisualParams* vparams);
//#endif
//#ifndef SOFA_DOUBLE
//template <>
//    void ProjectToLineConstraint<Rigid3fTypes >::draw(const core::visual::VisualParams* vparams);
//template <>
//    void ProjectToLineConstraint<Rigid2fTypes >::draw(const core::visual::VisualParams* vparams);
//#endif



} // namespace constraint

} // namespace component

} // namespace sofa

#endif


