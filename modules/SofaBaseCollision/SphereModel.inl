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
#include <SofaBaseCollision/SphereModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/CubeModel.h>
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

template<class DataTypes>
TSphereModel<DataTypes>::TSphereModel()
    : radius(initData(&radius, "listRadius","Radius of each sphere"))
    , defaultRadius(initData(&defaultRadius,(SReal)(1.0), "radius","Default Radius"))
    , mstate(NULL)
{
    enum_type = SPHERE_TYPE;
}

template<class DataTypes>
TSphereModel<DataTypes>::TSphereModel(core::behavior::MechanicalState<DataTypes>* _mstate )
    : radius(initData(&radius, "listRadius","Radius of each sphere"))
    , defaultRadius(initData(&defaultRadius,(SReal)(1.0), "radius","Default Radius"))
    , mstate(_mstate)
{
    enum_type = SPHERE_TYPE;
}


template<class DataTypes>
void TSphereModel<DataTypes>::resize(int size)
{
    this->core::CollisionModel::resize(size);

    VecReal &r = *radius.beginEdit();

    if ((int)r.size() < size)
    {
        r.reserve(size);
        while ((int)r.size() < size)
            r.push_back((Real)defaultRadius.getValue());
    }
    else
    {
        r.resize(size);
    }

    radius.endEdit();
}


template<class DataTypes>
void TSphereModel<DataTypes>::init()
{
    this->CollisionModel::init();
    mstate = dynamic_cast< core::behavior::MechanicalState<DataTypes>* > (getContext()->getMechanicalState());
    if (mstate==NULL)
    {
        serr<<"TSphereModel requires a Vec3 Mechanical Model" << sendl;
        return;
    }

    const int npoints = mstate->read(core::ConstVecCoordId::position())->getValue().size();
    resize(npoints);
}


template<class DataTypes>
void TSphereModel<DataTypes>::draw(const core::visual::VisualParams* ,int index)
{
#ifndef SOFA_NO_OPENGL
    TSphere<DataTypes> t(this,index);

    sofa::defaulttype::Vector3 p = t.p();
    glPushMatrix();
    glTranslated(p[0], p[1], p[2]);
    glutSolidSphere(t.r(), 32, 16);
    glPopMatrix();
#endif /* SOFA_NO_OPENGL */
}


template<class DataTypes>
void TSphereModel<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    using namespace sofa::defaulttype;

    if (!this->isActive()) return;
    //if (!vparams->isSupported(core::visual::API_OpenGL)) return;
    if (vparams->displayFlags().getShowCollisionModels())
    {
        vparams->drawTool()->setPolygonMode(0,vparams->displayFlags().getShowWireFrame());

        // Check topological modifications
        const int npoints = mstate->read(core::ConstVecCoordId::position())->getValue().size();

        std::vector<Vector3> points;
        std::vector<float> radius;
        for (int i=0; i<npoints; i++)
        {
            TSphere<DataTypes> t(this,i);
            Vector3 p = t.p();
            points.push_back(p);
            radius.push_back((float)t.r());
        }

        vparams->drawTool()->setLightingEnabled(true); //Enable lightning
        vparams->drawTool()->drawSpheres(points, radius, Vec<4,float>(getColor4f()));
        vparams->drawTool()->setLightingEnabled(false); //Disable lightning

    }

    if (getPrevious()!=NULL && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);

    vparams->drawTool()->setPolygonMode(0,false);
}

template <class DataTypes>
void TSphereModel<DataTypes>::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int npoints = mstate->read(core::ConstVecCoordId::position())->getValue().size();
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
        const typename TSphere<DataTypes>::Real distance = (typename TSphere<DataTypes>::Real)this->proximity.getValue();
        for (int i=0; i<size; i++)
        {
            TSphere<DataTypes> p(this,i);
            const typename TSphere<DataTypes>::Real r = p.r() + distance;
            const Coord minElem = p.center() - Coord(r,r,r);
            const Coord maxElem = p.center() + Coord(r,r,r);

            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}



template <class DataTypes>
void TSphereModel<DataTypes>::computeContinuousBoundingTree(double dt, int maxDepth)
{
    using namespace sofa::defaulttype;

    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int npoints = mstate->read(core::ConstVecCoordId::position())->getValue().size();
    bool updated = false;
    if (npoints != size)
    {
        resize(npoints);
        updated = true;
        cubeModel->resize(0);
    }

    if (!isMoving() && !cubeModel->empty() && !updated)
        return; // No need to recompute BBox if immobile

    Vector3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        const typename TSphere<DataTypes>::Real distance = (typename TSphere<DataTypes>::Real)this->proximity.getValue();
        for (int i=0; i<size; i++)
        {
            TSphere<DataTypes> p(this,i);
            const Vector3& pt = p.p();
            const Vector3 ptv = pt + p.v()*dt;

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt[c];
                maxElem[c] = pt[c];
                if (ptv[c] > maxElem[c]) maxElem[c] = ptv[c];
                else if (ptv[c] < minElem[c]) minElem[c] = ptv[c];
            }

            typename TSphere<DataTypes>::Real r = p.r() + distance;
            cubeModel->setParentOf(i, minElem - Vector3(r,r,r), maxElem + Vector3(r,r,r));
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

template <class DataTypes>
typename TSphereModel<DataTypes>::Real TSphereModel<DataTypes>::getRadius(const int i) const
{
    if(i < (int) this->radius.getValue().size())
        return radius.getValue()[i];
    else
        return (Real) defaultRadius.getValue();
}


} // namespace collision

} // namespace component

} // namespace sofa

