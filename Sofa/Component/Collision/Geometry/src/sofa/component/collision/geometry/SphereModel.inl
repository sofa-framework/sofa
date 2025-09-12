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
#include <sofa/component/collision/geometry/SphereModel.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/geometry/CubeModel.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::ComponentState ;

namespace sofa::component::collision::geometry
{

template<class DataTypes>
SphereCollisionModel<DataTypes>::SphereCollisionModel()
    : d_radius(initData(&d_radius, "listRadius", "Radius of each sphere"))
    , d_defaultRadius(initData(&d_defaultRadius, (SReal)(1.0), "radius", "Default radius"))
    , d_showImpostors(initData(&d_showImpostors, true, "showImpostors", "Draw spheres as impostors instead of \"real\" spheres"))
    , mstate(nullptr)
{
    enum_type = SPHERE_TYPE;
}

template<class DataTypes>
SphereCollisionModel<DataTypes>::SphereCollisionModel(core::behavior::MechanicalState<DataTypes>* _mstate )
    : d_radius(initData(&d_radius, "listRadius", "Radius of each sphere"))
    , d_defaultRadius(initData(&d_defaultRadius, (SReal)(1.0), "radius", "Default radius"))
    , d_showImpostors(initData(&d_showImpostors, true, "showImpostors", "Draw spheres as impostors instead of \"real\" spheres"))
    , mstate(_mstate)
{
    enum_type = SPHERE_TYPE;
}


template<class DataTypes>
void SphereCollisionModel<DataTypes>::resize(sofa::Size size)
{
    this->core::CollisionModel::resize(size);

    helper::WriteAccessor< Data<VecReal> > r = d_radius;

    if (r.size() < size)
    {
        r.reserve(size);
        while (r.size() < size)
            r.push_back((Real)d_defaultRadius.getValue());
    }
    else
    {
        r.resize(size);
    }
}


template<class DataTypes>
void SphereCollisionModel<DataTypes>::init()
{
    if(this->isComponentStateValid())
    {
        msg_warning() << "Calling an already fully initialized component. You should use reinit instead." ;
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
        d_componentState.setValue(ComponentState::Invalid) ;

        return;
    }

    const auto npoints = mstate->getSize();
    resize(npoints);

    d_componentState.setValue(ComponentState::Valid) ;
}


template<class DataTypes>
void SphereCollisionModel<DataTypes>::draw(const core::visual::VisualParams* vparams, sofa::Index index)
{
    if(d_componentState.getValue() != ComponentState::Valid)
        return ;

    TSphere<DataTypes> t(this,index);

    vparams->drawTool()->drawSphere(t.p(), (float)t.r());
}


template<class DataTypes>
void SphereCollisionModel<DataTypes>::drawCollisionModel(const core::visual::VisualParams* vparams)
{
    using namespace sofa::type;
    using namespace sofa::defaulttype;

    //  Force no wireframe mode to draw sphere collision
    vparams->drawTool()->setPolygonMode(0, false);

    // Check topological modifications
    const auto npoints = mstate->getSize();

    std::vector<Vec3> points;
    std::vector<float> radius;
    for (sofa::Size i = 0; i < npoints; i++)
    {
        TSphere<DataTypes> t(this, i);
        if (t.isActive())
        {
            Vec3 p = t.p();
            points.push_back(p);
            radius.push_back((float)t.r());
        }
    }

    vparams->drawTool()->setLightingEnabled(true);  // Enable lightning
    if (d_showImpostors.getValue())
        vparams->drawTool()->drawFakeSpheres(
            points, radius,
            sofa::type::RGBAColor(getColor4f()[0], getColor4f()[1], getColor4f()[2],
                                  getColor4f()[3]));
    else
        vparams->drawTool()->drawSpheres(points, radius,
                                         sofa::type::RGBAColor(getColor4f()[0], getColor4f()[1],
                                                               getColor4f()[2], getColor4f()[3]));
    vparams->drawTool()->setLightingEnabled(false);  // Disable lightning

    // restore current polygon mode
    vparams->drawTool()->setPolygonMode(0,vparams->displayFlags().getShowWireFrame());
}

template <class DataTypes>
void SphereCollisionModel<DataTypes>::computeBoundingTree(int maxDepth)
{
    if(d_componentState.getValue() != ComponentState::Valid)
        return ;

    CubeCollisionModel* cubeModel = createPrevious<CubeCollisionModel>();
    const auto npoints = mstate->getSize();
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
        const typename TSphere<DataTypes>::Real distance = (typename TSphere<DataTypes>::Real)this->d_contactDistance.getValue();
        for (sofa::Size i=0; i<size; i++)
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
void SphereCollisionModel<DataTypes>::computeContinuousBoundingTree(SReal dt, int maxDepth)
{
    using sofa::type::Vec3 ;

    if(d_componentState.getValue() != ComponentState::Valid)
        return ;

    CubeCollisionModel* cubeModel = createPrevious<CubeCollisionModel>();
    const auto npoints = mstate->getSize();
    bool updated = false;
    if (npoints != size)
    {
        resize(npoints);
        updated = true;
        cubeModel->resize(0);
    }

    if (!isMoving() && !cubeModel->empty() && !updated)
        return; // No need to recompute BBox if immobile

    Vec3 minElem, maxElem;

    cubeModel->resize(size);
    if (!empty())
    {
        const typename TSphere<DataTypes>::Real distance = (typename TSphere<DataTypes>::Real)this->d_contactDistance.getValue();
        for (sofa::Size i=0; i<size; i++)
        {
            TSphere<DataTypes> p(this,i);
            const Vec3& pt = p.p();
            const Vec3 ptv = pt + p.v()*dt;

            for (int c = 0; c < 3; c++)
            {
                minElem[c] = pt[c];
                maxElem[c] = pt[c];
                if (ptv[c] > maxElem[c]) maxElem[c] = ptv[c];
                else if (ptv[c] < minElem[c]) minElem[c] = ptv[c];
            }

            typename TSphere<DataTypes>::Real r = p.r() + distance;
            cubeModel->setParentOf(i, minElem - Vec3(r,r,r), maxElem + Vec3(r,r,r));
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}

template <class DataTypes>
typename SphereCollisionModel<DataTypes>::Real SphereCollisionModel<DataTypes>::getRadius(const sofa::Index i) const
{
    if(i < this->d_radius.getValue().size())
        return d_radius.getValue()[i];
    else
        return (Real) d_defaultRadius.getValue();
}

template<class DataTypes>
void SphereCollisionModel<DataTypes>::computeBBox(const core::ExecParams* params, bool onlyVisible)
{
    SOFA_UNUSED(params);

    if(d_componentState.getValue() != ComponentState::Valid)
        return ;

    if( onlyVisible && !sofa::core::visual::VisualParams::defaultInstance()->displayFlags().getShowCollisionModels())
        return;

    const auto npoints = mstate->getSize();
    type::BoundingBox bbox;

    for(sofa::Size i = 0 ; i < npoints ; ++i )
    {
        const TSphere<DataTypes> t(this,i);
        const Coord& p = t.p();
        const Real r = t.r();

        bbox.include(p + type::Vec3{r,r,r});
        bbox.include(p - type::Vec3{r,r,r});
    }

    this->f_bbox.setValue(bbox);
}

} // namespace sofa::component::collision::geometry
