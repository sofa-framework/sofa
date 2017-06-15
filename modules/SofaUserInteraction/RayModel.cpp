/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#include <SofaUserInteraction/RayModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/CubeModel.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/gl/template.h>


namespace sofa
{

namespace component
{

namespace collision
{

SOFA_DECL_CLASS(Ray)

int RayModelClass = core::RegisterObject("Collision model representing a ray in space, e.g. a mouse click")
        .add< RayModel >()
        .addAlias("Ray")
        ;


using namespace sofa::defaulttype;

RayModel::RayModel(SReal length)
    : defaultLength(initData(&defaultLength, length, "", "TODO"))
{
    this->contactResponse.setValue("ray"); // use RayContact response class
}

void RayModel::resize(int size)
{
    this->core::CollisionModel::resize(size);

    if ((int)length.size() < size)
    {
        length.reserve(size);
        while ((int)length.size() < size)
            length.push_back(defaultLength.getValue());
        direction.reserve(size);
        while ((int)direction.size() < size)
            direction.push_back(Vector3());

    }
    else
    {
        length.resize(size);
        direction.resize(size);
    }
}


void RayModel::init()
{
    this->CollisionModel::init();

    mstate = dynamic_cast< core::behavior::MechanicalState<Vec3Types>* > (getContext()->getMechanicalState());
    if (mstate==NULL)
    {
        serr<<"RayModel requires a Vec3 Mechanical Model" << sendl;
        return;
    }

    {
        const int npoints = mstate->getSize();
        resize(npoints);
    }
}


int RayModel::addRay(const Vector3& origin, const Vector3& direction, SReal length)
{
    int i = size;
    resize(i);
    Ray r = getRay(i);
    r.setOrigin(origin);
    r.setDirection(direction);
    r.setL(length);
    return i;
}

void RayModel::draw(const core::visual::VisualParams* vparams,int index)
{
#ifndef SOFA_NO_OPENGL
    if( !vparams->isSupported(core::visual::API_OpenGL) ) return;

    Ray r(this, index);
    const Vector3& p1 = r.origin();
    const Vector3 p2 = p1 + r.direction()*r.l();
    glBegin(GL_LINES);
    helper::gl::glVertexT(p1);
    helper::gl::glVertexT(p2);
    glEnd();
#endif /* SOFA_NO_OPENGL */
}

void RayModel::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    if( !vparams->isSupported(core::visual::API_OpenGL) ) return;

    if (vparams->displayFlags().getShowCollisionModels())
    {
        glDisable(GL_LIGHTING);
        glColor4fv(getColor4f());
        for (int i=0; i<size; i++)
        {
            draw(vparams,i);
        }
    }
    if (getPrevious()!=NULL && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);
#endif /* SOFA_NO_OPENGL */
}

void RayModel::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();

    if (!isMoving() && !cubeModel->empty()) return; // No need to recompute BBox if immobile

    Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        for (int i=0; i<size; i++)
        {
            Ray r(this, i);
            const Vector3& o = r.origin();
            const Vector3& d = r.direction();
            const SReal l = r.l();
            for (int c=0; c<3; c++)
            {
                if (d[c]<0)
                {
                    minElem[c] = o[c] + d[c]*l;
                    maxElem[c] = o[c];
                }
                else
                {
                    minElem[c] = o[c];
                    maxElem[c] = o[c] + d[c]*l;
                }
            }
            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }

}

void RayModel::applyTranslation(double dx, double dy, double dz)
{
    Vector3 d(dx,dy,dz);
    for (int i = 0; i < getNbRay(); i++)
    {
        Ray ray = getRay(i);
        ray.setOrigin(ray.origin() + d);
    }
}

} // namespace collision

} // namespace component

} // namespace sofa

