#ifndef RIGIDSPHEREMODEL_INL
#define RIGIDSPHEREMODEL_INL
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
#include <sofa/helper/system/config.h>
#include <sofa/helper/proximity.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/collision/Intersection.inl>
#include <iostream>
#include <algorithm>

#include <sofa/helper/io/SphereLoader.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/component/collision/RigidSphereModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glut.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/simulation/common/Simulation.h>


namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace helper;


template<class TReal>
TRigidSphereModel<TReal>::TRigidSphereModel()
    : radius(initData(&radius, "listRadius","Radius of each sphere"))
    , defaultRadius(initData(&defaultRadius,(SReal)(1.0), "radius","Default Radius"))
    , mstate(NULL)
{
}

template<class TReal>
TRigidSphereModel<TReal>::TRigidSphereModel(core::behavior::MechanicalState<DataTypes>* _mstate )
    : radius(initData(&radius, "listRadius","Radius of each sphere"))
    , defaultRadius(initData(&defaultRadius,(SReal)(1.0), "radius","Default Radius"))
    , mstate(_mstate)
{
}


template<class TReal>
void TRigidSphereModel<TReal>::resize(int size)
{
    this->core::CollisionModel::resize(size);

    VecReal &r = *radius.beginEdit();

    if ((int)r.size() < size)
    {
        r.reserve(size);
        while ((int)r.size() < size)
            r.push_back(defaultRadius.getValue());
    }
    else
    {
        r.resize(size);
    }

    radius.endEdit();
}


template<class TReal>
void TRigidSphereModel<TReal>::init()
{
    this->CollisionModel::init();
    mstate = dynamic_cast< core::behavior::MechanicalState<DataTypes >* > (getContext()->getMechanicalState());
    if (mstate==NULL)
    {
        serr<<"TRigidSphereModel requires a Vec3 Mechanical Model" << sendl;
        return;
    }

    const int npoints = mstate->getX()->size();
    resize(npoints);

}


template<class TReal>
void TRigidSphereModel<TReal>::draw(const core::visual::VisualParams* ,int index)
{
    TRigidSphere<TReal> t(this,index);

    Vector3 p = t.p();
    glPushMatrix();
    glTranslated(p[0], p[1], p[2]);
    glutSolidSphere(t.r(), 32, 16);
    glPopMatrix();
}


template<class TReal>
void TRigidSphereModel<TReal>::draw(const core::visual::VisualParams* vparams)
{
    //if (!vparams->isSupported(core::visual::API_OpenGL)) return;
    if (vparams->displayFlags().getShowCollisionModels())
    {
        vparams->drawTool()->setPolygonMode(0,vparams->displayFlags().getShowWireFrame());

        // Check topological modifications
        const int npoints = mstate->getX()->size();

        std::vector<Vector3> points;
        std::vector<float> radius;
        for (int i=0; i<npoints; i++)
        {
            TRigidSphere<TReal> t(this,i);
            Vector3 p = t.p();
            points.push_back(p);
            radius.push_back(t.r());
        }

        vparams->drawTool()->setLightingEnabled(true); //Enable lightning
        vparams->drawTool()->drawSpheres(points, radius, Vec<4,float>(getColor4f()));
        vparams->drawTool()->setLightingEnabled(false); //Disable lightning

    }

    if (getPrevious()!=NULL && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);

    vparams->drawTool()->setPolygonMode(0,false);
}

template <class TReal>
void TRigidSphereModel<TReal>::computeBoundingTree(int maxDepth)
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
        return; // No need to recompute BBox if immobile

    cubeModel->resize(size);
    if (!empty())
    {
        const TReal distance = (TReal)this->proximity.getValue();
        for (int i=0; i<size; i++)
        {
            TRigidSphere<TReal> p(this,i);
            const TReal r = p.r() + distance;
            const Vector3 minElem = p.center() - Pos(r,r,r);
            const Vector3 maxElem = p.center() + Pos(r,r,r);

            cubeModel->setParentOf(i, minElem, maxElem);

        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

template <class TReal>
TReal TRigidSphereModel<TReal>::getRadius(const int i) const
{
    if(i < (int) this->radius.getValue().size())
        return radius.getValue()[i];
    else
        return defaultRadius.getValue();
}


} // namespace collision

} // namespace component

} // namespace sofa


#endif // RIGIDSPHEREMODEL_INL
