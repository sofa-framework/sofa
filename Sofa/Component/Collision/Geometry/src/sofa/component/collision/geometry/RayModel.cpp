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
#include <sofa/component/collision/geometry/RayModel.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/component/collision/geometry/CubeModel.h>

namespace sofa::component::collision::geometry
{

int RayCollisionModelClass = core::RegisterObject("Collision model representing a ray in space, e.g. a mouse click")
        .add< RayCollisionModel >()
        ;

using namespace sofa::type;
using namespace sofa::defaulttype;

RayCollisionModel::RayCollisionModel(SReal length)
    : defaultLength(initData(&defaultLength, length, "", "TODO"))
{
    this->contactResponse.setValue("RayContact"); // use RayContact response class
}

void RayCollisionModel::resize(sofa::Size size)
{
    this->core::CollisionModel::resize(size);

    if (length.size() < size)
    {
        length.reserve(size);
        while (length.size() < size)
            length.push_back(defaultLength.getValue());
        direction.reserve(size);
        while (direction.size() < size)
            direction.push_back(Vec3());

    }
    else
    {
        length.resize(size);
        direction.resize(size);
    }
}


void RayCollisionModel::init()
{
    this->CollisionModel::init();

    mstate = dynamic_cast< core::behavior::MechanicalState<Vec3Types>* > (getContext()->getMechanicalState());
    if (mstate==nullptr)
    {
        msg_error() << "RayCollisionModel requires a Vec3 Mechanical Model";
        return;
    }

    {
        const int npoints = mstate->getSize();
        resize(npoints);
    }
}


int RayCollisionModel::addRay(const Vec3& origin, const Vec3& direction, SReal length)
{
    const int i = size;
    resize(i);
    Ray r = getRay(i);
    r.setOrigin(origin);
    r.setDirection(direction);
    r.setL(length);
    return i;
}

void RayCollisionModel::draw(const core::visual::VisualParams* vparams, sofa::Index index)
{
    if( !vparams->isSupported(core::visual::API_OpenGL) ) return;

    const Ray r(this, index);
    const Vec3& p1 = r.origin();
    const Vec3 p2 = p1 + r.direction()*r.l();

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();
    vparams->drawTool()->disableLighting();
    constexpr sofa::type::RGBAColor color = sofa::type::RGBAColor::magenta();
    vparams->drawTool()->drawLine(p1,p2,color);

}

void RayCollisionModel::draw(const core::visual::VisualParams* vparams)
{
    if (vparams->displayFlags().getShowCollisionModels())
    {       
        for (sofa::Index i=0; i<size; i++)
        {
            draw(vparams,i);
        }
    }
    if (getPrevious()!=nullptr && vparams->displayFlags().getShowBoundingCollisionModels())
    {
        getPrevious()->draw(vparams);
    }
}

void RayCollisionModel::computeBoundingTree(int maxDepth)
{
    CubeCollisionModel* cubeModel = createPrevious<CubeCollisionModel>();

    if (!isMoving() && !cubeModel->empty()) return; // No need to recompute BBox if immobile

    Vec3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        for (sofa::Index i=0; i<size; i++)
        {
            Ray r(this, i);
            const Vec3& o = r.origin();
            const Vec3& d = r.direction();
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

void RayCollisionModel::applyTranslation(double dx, double dy, double dz)
{
    const Vec3 d(dx,dy,dz);
    for (int i = 0; i < getNbRay(); i++)
    {
        Ray ray = getRay(i);
        ray.setOrigin(ray.origin() + d);
    }
}

const type::Vec3& Ray::origin() const
{
    return model->getMechanicalState()->read(core::ConstVecCoordId::position())->getValue()[index];
}

const type::Vec3& Ray::direction() const
{
    return model->direction[index];
}

SReal Ray::l() const
{
    return model->length[index];
}

void Ray::setOrigin(const type::Vec3& newOrigin)
{
    auto xData = sofa::helper::getWriteAccessor(*model->getMechanicalState()->write(core::VecCoordId::position()));
    xData.wref()[index] = newOrigin;

    auto xDataFree = sofa::helper::getWriteAccessor(*model->getMechanicalState()->write(core::VecCoordId::freePosition()));
    auto& freePos = xDataFree.wref();
    freePos.resize(model->getMechanicalState()->getSize());
    freePos[index] = newOrigin;
}


} //namespace sofa::component::collision::geometry
