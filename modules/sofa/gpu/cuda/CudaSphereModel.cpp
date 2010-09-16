/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include "CudaSphereModel.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/collision/CubeModel.h>
#include <fstream>

#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glut.h>

#include <sofa/helper/io/SphereLoader.h>
#include <sofa/helper/system/FileRepository.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaSphereModel)

int CudaSphereModelClass = core::RegisterObject("GPU-based sphere collision model using CUDA")
        .add< CudaSphereModel >()
        .addAlias("CudaSphere")
        ;

using namespace defaulttype;

CudaSphereModel::CudaSphereModel()
    : mstate(NULL)
    , radius( initData(&radius, "listRadius", "Radius of each sphere"))
    , defaultRadius( initData(&defaultRadius,(SReal)(1.0), "radius", "Default Radius"))
    , filename(initData(&filename, "fileSphere", "File .sph describing the spheres"))
{

}

void CudaSphereModel::init()
{
    this->CollisionModel::init();
    mstate = dynamic_cast< core::behavior::MechanicalState<InDataTypes>*> (getContext()->getMechanicalState());

    if (mstate==NULL)
    {
        serr << "ERROR: CudaSphereModel requires a CudaVec3f Mechanical Model.\n";
        return;
    }

    const int npoints = mstate->getX()->size();
    resize(npoints);
}

CudaSphereModel::Real CudaSphereModel::getRadius(const int i) const
{
    if (i < (int) this->radius.getValue().size())
    {
        return radius.getValue()[i];
    }
    else
    {
        return defaultRadius.getValue();
    }
}

void CudaSphereModel::draw(int index)
{
    CudaSphere t(this,index);

    Coord p = t.p();
    glPushMatrix();
    glTranslated(p[0], p[1], p[2]);
    glutSolidSphere(t.r(), 32, 16);
    glPopMatrix();
}

void CudaSphereModel::drawColourPicking(const ColourCode method)
{
    using namespace sofa::core::objectmodel;

    if( method == ENCODE_RELATIVEPOSITION ) return; // we pick the center of the sphere.

    helper::vector<core::CollisionModel*> listCollisionModel;
    this->getContext()->get<core::CollisionModel>(&listCollisionModel,BaseContext::SearchRoot);
    const int totalCollisionModel = listCollisionModel.size();
    helper::vector<core::CollisionModel*>::iterator iter = std::find(listCollisionModel.begin(), listCollisionModel.end(), this);
    const int indexCollisionModel = std::distance(listCollisionModel.begin(),iter ) + 1 ;
    float red = (float)indexCollisionModel / (float)totalCollisionModel;
    // Check topological modifications
    const int npoints = mstate->getX()->size();
    std::vector<Vector3> points;
    std::vector<float> radius;
    for (int i=0; i<npoints; i++)
    {
        CudaSphere t(this,i);
        Vector3 p = t.p();
        points.push_back(p);
        radius.push_back(t.r());
    }
    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_DITHER);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    float ratio;
    for( int i=0; i<npoints; i++)
    {
        Vector3 p = points[i];

        glPushMatrix();
        glTranslated(p[0], p[1], p[2]);
        ratio = (float)i / (float)npoints;
        glColor4f(red,ratio,0,1);
        glutSolidSphere(radius[i], 32, 16);

        glPopMatrix();
    }
}

sofa::defaulttype::Vector3 CudaSphereModel::getPositionFromWeights(int index, Real /*a*/ ,Real /*b*/, Real /*c*/)
{
    Element sphere(this,index);

    return sphere.center();

}

void CudaSphereModel::draw()
{
    /*	serr<<"CudaSphereModel::draw()"<<endl; */
    if(getContext()->getShowCollisionModels())
    {
        /*		serr<<"CudaSphereModel::draw : yes"<<endl; */
        glEnable(GL_LIGHTING);
        glEnable(GL_COLOR_MATERIAL);
        glColor4fv(getColor4f());

        const int npoints = mstate->getX()->size();

        for (int i=0; i<npoints; i++)
        {
            draw(i);
        }

        glDisable(GL_LIGHTING);
        glDisable(GL_COLOR_MATERIAL);
    }
    if (getPrevious()!=NULL && getContext()->getShowBoundingCollisionModels())
    {
        getPrevious()->draw();
    }
}

using sofa::component::collision::CubeModel;

void CudaSphereModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int npoints = mstate->getX()->size();
    bool updated = false;
    if (npoints != size)
    {
        resize(npoints);
        updated = true;
        cubeModel->resize(0);
    }

    if (!isMoving() && !cubeModel->empty() && !updated)
    {
        return; // No need to recompute BBox if immobile
    }

    cubeModel->resize(size);
    if (!empty())
    {
        for (int i=0; i<size; i++)
        {
            CudaSphere p(this,i);
            const CudaSphere::Real r = p.r();
            const Vector3 minElem = p.center() - Vector3(r, r, r);
            const Vector3 maxElem = p.center() + Vector3(r, r, r);

            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

void CudaSphereModel::resize(int size)
{
    this->core::CollisionModel::resize(size);

    if((int) radius.getValue().size() < size)
    {
        radius.beginEdit()->reserve(size);
        while((int)radius.getValue().size() < size)
            radius.beginEdit()->push_back(defaultRadius.getValue());
    }
    else
    {
        radius.beginEdit()->resize(size);
    }
}

void CudaSphereModel::setRadius(const int i, const CudaSphereModel::Real r)
{
    if((int) radius.getValue().size() <= i)
    {
        radius.beginEdit()->reserve(i+1);
        while((int)radius.getValue().size() <= i)
            radius.beginEdit()->push_back(defaultRadius.getValue());
    }

    (*radius.beginEdit())[i] = r;
}

void CudaSphereModel::setRadius(const CudaSphereModel::Real r)
{
    *defaultRadius.beginEdit() = r;
    radius.beginEdit()->clear();
}

class CudaSphereModel::Loader : public helper::io::SphereLoader
{
public:
    CudaSphereModel* dest;
    Loader(CudaSphereModel* dest)
        : dest(dest) {}

    void addSphere(SReal x, SReal y, SReal z, SReal r)
    {
        dest->addSphere(Vector3(x,y,z),r);
    }
};

bool CudaSphereModel::load(const char* filename)
{
    this->resize(0);
    std::string sphereFilename(filename);
    if (!sofa::helper::system::DataRepository.findFile (sphereFilename))
        serr<<"Sphere File \""<< filename <<"\" not found"<< sendl;

    Loader loader(this);
    return loader.load(filename);
}

int CudaSphereModel::addSphere(const Vector3& pos, Real r)
{
    int i = size;
    resize(i+1);
    if((int) mstate->getX()->size() != i+1)
        mstate->resize(i+1);

    setSphere(i, pos, r);
    return i;
}

void CudaSphereModel::setSphere(int i, const Vector3& pos, Real r)
{
    (*mstate->getX())[i] = pos;
    setRadius(i,r);
}

} // namespace cuda

} // namespace gpu

} // namespace sofa


