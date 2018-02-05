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
#include "CudaPointModel.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/CubeModel.h>
#include <fstream>
#include <sofa/helper/system/gl.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaPointModel)

int CudaPointModelClass = core::RegisterObject("GPU-based point collision model using CUDA")
        .add< CudaPointModel >()
        .addAlias("CudaPoint")
        ;

using namespace defaulttype;

CudaPointModel::CudaPointModel()
    : groupSize( initData( &groupSize, (int)BSIZE, "groupSize", "number of point per collision element" ) )
    , mstate(NULL)
{
}

void CudaPointModel::resize(int size)
{
    this->core::CollisionModel::resize(size);
}

void CudaPointModel::init()
{
    this->CollisionModel::init();
    mstate = dynamic_cast< core::behavior::MechanicalState<InDataTypes>* > (getContext()->getMechanicalState());

    if (mstate==NULL)
    {
        serr << "ERROR: CudaPointModel requires a CudaVec3f Mechanical Model.\n";
        return;
    }

    const int npoints = mstate->getSize();
    int gsize = groupSize.getValue();
    int nelems = (npoints + gsize-1)/gsize;
    resize(nelems);
}

void CudaPointModel::draw(const core::visual::VisualParams* ,int index)
{
    const int gsize = groupSize.getValue();
    CudaPoint t(this,index);
    glBegin(GL_POINTS);
    const VecCoord& x = mstate->read(core::ConstVecCoordId::position())->getValue();
    int i0 = index*gsize;
    int n = (index==size-1) ? x.size()-i0 : gsize;
    for (int p=0; p<n; p++)
    {
        glVertex3fv(x[i0+p].ptr());
    }
    glEnd();
}

void CudaPointModel::draw(const core::visual::VisualParams* vparams)
{
    if (isActive() && vparams->displayFlags().getShowCollisionModels())
    {
        if (vparams->displayFlags().getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        glDisable(GL_LIGHTING);
        glPointSize(3);
        glColor4fv(getColor4f());

        for (int i=0; i<size; i++)
        {
            draw(vparams,i);
        }

        glColor3f(1.0f, 1.0f, 1.0f);
        glDisable(GL_LIGHTING);
        glPointSize(1);
        if (vparams->displayFlags().getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    if (isActive() && getPrevious()!=NULL && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);
}

using sofa::component::collision::CubeModel;

void CudaPointModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int npoints = mstate->getSize();
    const int gsize = groupSize.getValue();
    const int nelems = (npoints + gsize-1)/gsize;
    bool updated = false;
    if (nelems != size)
    {
        resize(nelems);
        updated = true;
    }
    if (updated) cubeModel->resize(0);
    if (!isMoving() && !cubeModel->empty() && !updated) return; // No need to recompute BBox if immobile

    cubeModel->resize(size);
    if (!empty())
    {
        const VecCoord& x = mstate->read(core::ConstVecCoordId::position())->getValue();
        for (int i=0; i<size; i++)
        {
            int i0 = i*gsize;
            int n = (i==size-1) ? npoints-i0 : gsize;
            Vec3f bmin = x[i0];
            Vec3f bmax = bmin;
            for (int p=1; p<n; p++)
            {
                const Vec3f& xi = x[i0+p];
                for (int c=0; c<3; c++)
                {
                    if (xi[c] < bmin[c]) bmin[c] = xi[c];
                    else if (xi[c] > bmax[c]) bmax[c] = xi[c];
                }
            }
            cubeModel->setParentOf(i, bmin, bmax);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

} // namespace cuda

} // namespace gpu

} // namespace sofa
