#include "CudaPointModel.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/collision/CubeModel.h>
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
    mstate = dynamic_cast< core::componentmodel::behavior::MechanicalState<InDataTypes>* > (getContext()->getMechanicalState());

    if (mstate==NULL)
    {
        std::cerr << "ERROR: CudaPointModel requires a CudaVec3f Mechanical Model.\n";
        return;
    }

    const int npoints = mstate->getX()->size();
    int gsize = groupSize.getValue();
    int nelems = (npoints + gsize-1)/gsize;
    resize(nelems);
}

void CudaPointModel::draw(int index)
{
    const int gsize = groupSize.getValue();
    CudaPoint t(this,index);
    glBegin(GL_POINTS);
    const VecCoord& x = *mstate->getX();
    int i0 = index*gsize;
    int n = (index==size-1) ? x.size()-i0 : gsize;
    for (int p=0; p<n; p++)
    {
        glVertex3fv(x[i0+p].ptr());
    }
    glEnd();
}

void CudaPointModel::draw()
{
    if (isActive() && getContext()->getShowCollisionModels())
    {
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        glDisable(GL_LIGHTING);
        glPointSize(3);
        if (isStatic())
            glColor3f(0.5, 0.5, 0.5);
        else
            glColor3f(1.0, 0.0, 0.0);

        for (int i=0; i<size; i++)
        {
            draw(i);
        }

        glColor3f(1.0f, 1.0f, 1.0f);
        glDisable(GL_LIGHTING);
        glPointSize(1);
        if (getContext()->getShowWireFrame())
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }
    if (isActive() && getPrevious()!=NULL && getContext()->getShowBoundingCollisionModels() && dynamic_cast<core::VisualModel*>(getPrevious())!=NULL)
        dynamic_cast<core::VisualModel*>(getPrevious())->draw();
}

using sofa::component::collision::CubeModel;

void CudaPointModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int npoints = mstate->getX()->size();
    const int gsize = groupSize.getValue();
    const int nelems = (npoints + gsize-1)/gsize;
    bool updated = false;
    if (nelems != size)
    {
        resize(nelems);
        updated = true;
    }
    if (updated) cubeModel->resize(0);
    if (isStatic() && !cubeModel->empty() && !updated) return; // No need to recompute BBox if immobile

    cubeModel->resize(size);
    if (!empty())
    {
        const VecCoord& x = *mstate->getX();
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
