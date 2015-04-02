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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectToPlaneUnilateralConstraint_INL
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_ProjectToPlaneUnilateralConstraint_INL

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/ProjectiveConstraintSet.inl>
#include <sofa/component/projectiveconstraintset/ProjectToPlaneUnilateralConstraint.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/helper/gl/template.h>
//#include <sofa/defaulttype/RigidTypes.h>
#include <iostream>
using std::cerr;
using std::endl;
#include <SofaBaseTopology/TopologySubsetData.inl>


#include <sofa/helper/gl/BasicShapes.h>




namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

using namespace core::topology;

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace sofa::core::behavior;



template <class DataTypes>
ProjectToPlaneUnilateralConstraint<DataTypes>::ProjectToPlaneUnilateralConstraint()
    : core::behavior::ProjectiveConstraintSet<DataTypes>(NULL)
    , f_origin( initData(&f_origin,CPos(),"origin","A point in the plane"))
    , f_normal( initData(&f_normal,CPos(),"normal","Normal vector to the plane"))
    , f_drawSize( initData(&f_drawSize,(Real)0.0,"drawSize","0 -> one pixel, >0 -> radius of spheres") )
    , f_drawPlane( initData(&f_drawPlane,(Real)0.0,"drawPlane","size of the plane to draw") )
    , data(new ProjectToPlaneUnilateralConstraintInternalData<DataTypes>())
{
}


template <class DataTypes>
ProjectToPlaneUnilateralConstraint<DataTypes>::~ProjectToPlaneUnilateralConstraint()
{
    delete data;
}

// -- Constraint interface


template <class DataTypes>
void ProjectToPlaneUnilateralConstraint<DataTypes>::init()
{
    this->core::behavior::ProjectiveConstraintSet<DataTypes>::init();

    reinit();

}

template <class DataTypes>
void  ProjectToPlaneUnilateralConstraint<DataTypes>::reinit()
{
    // normalize the normal vector
    CPos n = f_normal.getValue();
    if( n.norm()==0 )
        n[1]=0;
    else n *= 1/n.norm();
    f_normal.setValue(n);

}

template <class DataTypes>
void ProjectToPlaneUnilateralConstraint<DataTypes>::projectMatrix( sofa::defaulttype::BaseMatrix* M, unsigned offset )
{
    J.copy(jacobian, M->colSize(), offset); // projection matrix for an assembled state
    BaseSparseMatrix* E = dynamic_cast<BaseSparseMatrix*>(M);
    assert(E);
    E->compressedMatrix = J.compressedMatrix * E->compressedMatrix * J.compressedMatrix;
}



template <class DataTypes>
void ProjectToPlaneUnilateralConstraint<DataTypes>::projectResponse(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& resData)
{
    helper::WriteAccessor<DataVecDeriv> res ( mparams, resData );
    //    cerr<< "ProjectToPlaneUnilateralConstraint<DataTypes>::projectResponse input  = "<< endl << res.ref() << endl;
    jacobian.mult(res.wref(),res.ref());
    //    cerr<< "ProjectToPlaneUnilateralConstraint<DataTypes>::projectResponse output = "<< endl << res.wref() << endl;
}

template <class DataTypes>
void ProjectToPlaneUnilateralConstraint<DataTypes>::projectJacobianMatrix(const core::MechanicalParams* /*mparams*/ , DataMatrixDeriv& /*cData*/)
{
    serr<<"projectJacobianMatrix(const core::MechanicalParams*, DataMatrixDeriv& ) is not implemented" << sendl;
}

template <class DataTypes>
void ProjectToPlaneUnilateralConstraint<DataTypes>::projectVelocity(const core::MechanicalParams* mparams, DataVecDeriv& vdata)
{
    projectResponse(mparams,vdata);
}

template <class DataTypes>
void ProjectToPlaneUnilateralConstraint<DataTypes>::projectPosition(const core::MechanicalParams* /*mparams*/ , DataVecCoord& xData)
{
    VecCoord& x = *xData.beginEdit();
    isActive.resize(x.size());

    const CPos& n = f_normal.getValue();
    const CPos& o = f_origin.getValue();
    //    cerr<<"ProjectToPlaneUnilateralConstraint<DataTypes>::projectPosition, origin = "<<o <<", normal = " << n << endl;

    for(unsigned i=0; i<x.size(); i++ )
    {
        // replace the point with its projection to the plane
        Real distance = (x[i]-o)*n ;
        if( distance < 0 ){ // negative side of the plane
            //            cerr<<"ProjectToPlaneUnilateralConstraint<DataTypes>::projectPosition particle  "<<i<<", pos = "<<x[i]<<", distance = "<< distance << endl;
            DataTypes::setCPos( x[i], x[i] - n * distance );
            isActive[i] = true;
        }
        else {
            isActive[i]=false;
        }
    }
    xData.endEdit();

    // resize the jacobian
    unsigned numBlocks = this->mstate->getSize();
    unsigned blockSize = DataTypes::deriv_total_size;
    jacobian.resize( numBlocks*blockSize,numBlocks*blockSize );

    // create the matrix blocks corresponding to the projection to the plane: I-nn^t or to the identity
    Block bProjection, bIdentity;
    for(unsigned i=0; i<bsize; i++)
        for(unsigned j=0; j<bsize; j++)
        {
            if(i==j)
            {
                bIdentity[i][j]   = 1;
                bProjection[i][j] = 1 - n[i]*n[j];
            }
            else
            {
                bIdentity[i][j]   = 0;
                bProjection[i][j] =    - n[i]*n[j];
            }
        }
    //    cerr<<"ProjectToPlaneUnilateralConstraint<DataTypes>::reinit() bIdentity[0] = " << endl << bIdentity[0] << endl;
    //    cerr<<"ProjectToPlaneUnilateralConstraint<DataTypes>::reinit() bProjection[0] = " << endl << bProjection[0] << endl;

    // fill the jacobian in ascending order
    for( unsigned i=0; i<numBlocks; i++ )
    {
        jacobian.beginBlockRow(i);
        if(  isActive[i] )  // constrained particle: set diagonal to projection block, and  the cursor to the next constraint
        {
            jacobian.createBlock(i,bProjection); // only one block to create
            //            cerr<<"ProjectToPlaneUnilateralConstraint<DataTypes>::reinit(), constrain index " << i << endl;
        }
        else           // unconstrained particle: set diagonal to identity block
        {
            jacobian.createBlock(i,bIdentity); // only one block to create
        }
        jacobian.endBlockRow();
    }
    jacobian.compress();
    //    cerr<<"ProjectToPlaneUnilateralConstraint<DataTypes>::projectPositions, jacobian = " << jacobian << endl;


}

// Matrix Integration interface
template <class DataTypes>
void ProjectToPlaneUnilateralConstraint<DataTypes>::applyConstraint(defaulttype::BaseMatrix * /*mat*/, unsigned int /*offset*/)
{
    serr << "applyConstraint is not implemented " << sendl;
}

template <class DataTypes>
void ProjectToPlaneUnilateralConstraint<DataTypes>::applyConstraint(defaulttype::BaseVector * /*vect*/, unsigned int /*offset*/)
{
    serr<<"ProjectToPlaneUnilateralConstraint<DataTypes>::applyConstraint(defaulttype::BaseVector *vect, unsigned int offset) is not implemented "<< sendl;
}




template <class DataTypes>
void ProjectToPlaneUnilateralConstraint<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if (!vparams->displayFlags().getShowBehaviorModels()) return;
    const VecCoord& x = *this->mstate->getX();

    if( f_drawSize.getValue() == 0) // old classical drawing by points
    {
        std::vector< Vector3 > points;
        Vector3 point;
        //serr<<"ProjectToPlaneUnilateralConstraint<DataTypes>::draw(), indices = "<<indices<<sendl;
        for (unsigned i=0; i<x.size(); i++) if(isActive[i])
        {
            point = DataTypes::getCPos(x[i]);
            points.push_back(point);
        }
        vparams->drawTool()->drawPoints(points, 10, Vec<4,float>(1,0.5,0.5,1));
    }
    else // new drawing by spheres
    {
        std::vector< Vector3 > points;
        Vector3 point;
        glColor4f (1.0f,0.35f,0.35f,1.0f);
        for (unsigned i=0; i<x.size(); i++) if(isActive[i])
        {
            point = DataTypes::getCPos(x[i]);
            points.push_back(point);
        }
        vparams->drawTool()->drawSpheres(points, (float)f_drawSize.getValue(), Vec<4,float>(1.0f,0.35f,0.35f,1.0f));
    }
    if( f_drawPlane.getValue()>0.0 )
    {
        std::vector< Vector3 > points;
        std::vector< Vector3 > normals;
        Real x,y,z,ox,oy,oz;
        DataTypes::get(x,y,z,f_normal.getValue() );
        DataTypes::get(ox,oy,oz,f_origin.getValue() );
        Vector3 n(x,y,z), t1, t2, o(ox,oy,oz);

        // find two vectors orthogonal to n
        if(n[0]<=n[1] && n[0]<=n[2])
        {
            t1 = Vector3(1,0,0);
        }
        else if(n[1]<=n[0] && n[1]<=n[2])
            t1 = Vector3(0,1,0);
        else
            t1 = Vector3(0,0,1);
        t2= cross(n,t1);
        t2.normalize();
        t1=cross(t2,n);

        // create a four-point fan
        points.push_back( o ); normals.push_back(n);
        points.push_back( o+( t1-t2)* f_drawPlane.getValue() );
        points.push_back( o+( t1+t2)* f_drawPlane.getValue() );
        points.push_back( o+(-t1+t2)* f_drawPlane.getValue() );
        points.push_back( o+(-t1-t2)* f_drawPlane.getValue() );
        points.push_back( o+( t1-t2)* f_drawPlane.getValue() );
        vparams->drawTool()->drawTriangleFan(points,normals, Vec<4,float>(0.4f,0.0f,0.0f,1.0f));
    }
#endif /* SOFA_NO_OPENGL */
}


} // namespace constraint

} // namespace component

} // namespace sofa

#endif


