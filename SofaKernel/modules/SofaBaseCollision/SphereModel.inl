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
#include <sofa/helper/system/config.h>
#include <sofa/helper/proximity.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <iostream>
#include <algorithm>

#include <sofa/helper/io/SphereLoader.h>
#include <sofa/helper/system/FileRepository.h>
#include <SofaBaseCollision/SphereModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/CubeModel.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/simulation/Simulation.h>

#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::ComponentState ;

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
    , d_showImpostors(initData(&d_showImpostors, true, "showImpostors", "Draw spheres as impostors instead of \"real\" spheres"))
    , mstate(NULL)
{
    enum_type = SPHERE_TYPE;
}

template<class DataTypes>
TSphereModel<DataTypes>::TSphereModel(core::behavior::MechanicalState<DataTypes>* _mstate )
    : radius(initData(&radius, "listRadius","Radius of each sphere"))
    , defaultRadius(initData(&defaultRadius,(SReal)(1.0), "radius","Default Radius. (default=1.0)"))
    , d_showImpostors(initData(&d_showImpostors, true, "showImpostors", "Draw spheres as impostors instead of \"real\" spheres"))
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
    if(m_componentstate==ComponentState::Valid){
        msg_warning(this) << "Calling an already fully initialized component. You should use reinit instead." ;
    }

    this->CollisionModel::init();
    mstate = dynamic_cast< core::behavior::MechanicalState<DataTypes>* > (getContext()->getMechanicalState());
    if (mstate==nullptr)
    {
        //TODO(dmarchal): The previous message was saying this only work for a vec3 mechanicalstate but there
        // it seems that a mechanicalstate will work we well...where is the truth ?
        msg_error(this) << "Missing a MechanicalObject with template '" << DataTypes::Name() << ". "
                           "This MechnicalObject stores the position of the spheres. When this one is missing the collision model is deactivated. \n"
                           "To remove this error message you can add to your scene a line <MechanicalObject template='"<< DataTypes::Name() << "'/>. ";
        m_componentstate = ComponentState::Invalid ;

        return;
    }

    const int npoints = mstate->getSize();
    resize(npoints);

    m_componentstate = ComponentState::Valid ;
}


template<class DataTypes>
void TSphereModel<DataTypes>::draw(const core::visual::VisualParams* vparams,int index)
{
    if(m_componentstate!=ComponentState::Valid)
        return ;

    TSphere<DataTypes> t(this,index);

    vparams->drawTool()->drawSphere(t.p(), (float)t.r());
}


template<class DataTypes>
void TSphereModel<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if(m_componentstate!=ComponentState::Valid)
        return ;

    using namespace sofa::defaulttype;

    if (!this->isActive()) return;
    //if (!vparams->isSupported(core::visual::API_OpenGL)) return;
    if (vparams->displayFlags().getShowCollisionModels())
    {
        vparams->drawTool()->setPolygonMode(0,vparams->displayFlags().getShowWireFrame());

        // Check topological modifications
        const int npoints = mstate->getSize();

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
        if(d_showImpostors.getValue())
            vparams->drawTool()->drawFakeSpheres(points, radius, Vec<4,float>(getColor4f()));
        else
            vparams->drawTool()->drawSpheres(points, radius, Vec<4, float>(getColor4f()));
        vparams->drawTool()->setLightingEnabled(false); //Disable lightning

    }

    if (getPrevious()!=NULL && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);

    vparams->drawTool()->setPolygonMode(0,false);
}

template <class DataTypes>
void TSphereModel<DataTypes>::computeBoundingTree(int maxDepth)
{
    if(m_componentstate!=ComponentState::Valid)
        return ;

    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int npoints = mstate->getSize();
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
void TSphereModel<DataTypes>::computeContinuousBoundingTree(SReal dt, int maxDepth)
{
    using sofa::defaulttype::Vector3 ;

    if(m_componentstate!=ComponentState::Valid)
        return ;

    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int npoints = mstate->getSize();
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

template<class DataTypes>
void TSphereModel<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    if(m_componentstate!=ComponentState::Valid)
        return ;

    if( !onlyVisible )
        return;

    static const Real max_real = std::numeric_limits<Real>::max();
    Real maxBBox[3] = {-max_real,-max_real,-max_real}; //Warning: minimum of float/double is 0, not -inf
    Real minBBox[3] = {max_real,max_real,max_real};

    const int npoints = mstate->getSize();

    for(int i = 0 ; i < npoints ; ++i )
    {
        TSphere<DataTypes> t(this,i);
        const Coord& p = t.p();
        Real r = t.r();

        for (int c=0; c<3; c++)
        {
            if (p[c]+r > maxBBox[c]) maxBBox[c] = (Real)p[c]+r;
            if (p[c]-r < minBBox[c]) minBBox[c] = (Real)p[c]-r;
        }
    }

    this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<Real>(minBBox,maxBBox));
}


} // namespace collision

} // namespace component

} // namespace sofa

