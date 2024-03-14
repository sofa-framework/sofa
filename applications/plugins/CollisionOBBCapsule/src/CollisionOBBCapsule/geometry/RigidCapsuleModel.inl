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
#include <CollisionOBBCapsule/geometry/RigidCapsuleModel.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/geometry/CubeModel.h>

namespace collisionobbcapsule::geometry
{

template<class MyReal>
CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::CapsuleCollisionModel():
      d_capsule_radii(initData(&d_capsule_radii, "radii","Radius of each capsule")),
      d_capsule_heights(initData(&d_capsule_heights,"heights","The capsule heights")),
      d_default_radius(initData(&d_default_radius,(Real)0.5,"defaultRadius","The default radius")),
      d_default_height(initData(&d_default_height,(Real)2,"dafaultHeight","The default height")),
      _mstate(nullptr)
{
    enum_type = CAPSULE_TYPE;
}

template<class MyReal>
CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::CapsuleCollisionModel(core::behavior::MechanicalState<DataTypes>* mstate):
    d_capsule_radii(initData(&d_capsule_radii, "radii","Radius of each capsule")),
    d_capsule_heights(initData(&d_capsule_heights,"heights","The capsule heights")),
    d_default_radius(initData(&d_default_radius,(Real)0.5,"defaultRadius","The default radius")),
    d_default_height(initData(&d_default_height,(Real)2,"dafaultHeight","The default height")),
    _mstate(mstate)
{
    enum_type = CAPSULE_TYPE;
}

template<class MyReal>
void CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::resize(sofa::Size size)
{
    this->core::CollisionModel::resize(size);

    VecReal & capsule_radii = *d_capsule_radii.beginEdit();
    VecReal & capsule_heights = *d_capsule_heights.beginEdit();

    if (capsule_radii.size() < size)
    {
        while(capsule_radii.size() < size)
            capsule_radii.push_back(d_default_radius.getValue());
    }
    else
    {
        capsule_radii.reserve(size);
    }

    if (capsule_heights.size() < size)
    {
        while(capsule_heights.size() < size)
            capsule_heights.push_back(d_default_height.getValue());
    }
    else
    {
        capsule_heights.reserve(size);
    }

    d_capsule_radii.endEdit();
    d_capsule_heights.endEdit();
}


template<class MyReal>
void CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::init()
{
    this->CollisionModel::init();
    _mstate = dynamic_cast< core::behavior::MechanicalState<DataTypes>* > (getContext()->getMechanicalState());
    if (_mstate==nullptr)
    {
        msg_error() << "CapsuleCollisionModel requires a Rigid Mechanical Model";
        return;
    }

    resize(_mstate->getSize());
}

template <class MyReal>
Size CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::nbCap()const
{
    return sofa::Size(d_capsule_radii.getValue().size());
}

template <class MyReal>
void CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::computeBoundingTree(int maxDepth)
{
    sofa::component::collision::geometry::CubeCollisionModel* cubeModel = createPrevious<sofa::component::collision::geometry::CubeCollisionModel>();
    const auto ncap = _mstate->getSize();

    bool updated = false;
    if (ncap != size)
    {
        resize(ncap);
        updated = true;
        cubeModel->resize(0);
    }

    if (!isMoving() && !cubeModel->empty() && !updated){
        return; // No need to recompute BBox if immobile
    }

    cubeModel->resize(ncap);
    if (!empty())
    {
        typename TCapsule<defaulttype::StdRigidTypes<3,MyReal> >::Real r;
        for (sofa::Size i=0; i<ncap; i++)
        {
            const Coord p1 = point1(i);
            const Coord p2 = point2(i);
            r = radius(i);

            type::Vec3 maxVec;
            type::Vec3 minVec;

            for(int dim = 0 ; dim < 3 ; ++dim){
                if(p1(dim) > p2(dim)){
                    maxVec(dim) = p1(dim) + r;
                    minVec(dim) = p2(dim) - r;
                }
                else{
                    maxVec(dim) = p2(dim) + r;
                    minVec(dim) = p1(dim) - r;
                }
            }

            cubeModel->setParentOf(i, minVec, maxVec);

        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}


template<class MyReal>
void CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::draw(const core::visual::VisualParams* vparams, sofa::Index index)
{
    sofa::type::RGBAColor col4f(getColor4f()[0], getColor4f()[1], getColor4f()[2], getColor4f()[3]);
    vparams->drawTool()->drawCapsule(point1(index),point2(index),(float)radius(index),col4f);
}

template<class MyReal>
void CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::draw(const core::visual::VisualParams* vparams)
{
    if (vparams->displayFlags().getShowCollisionModels())
    {
        sofa::type::RGBAColor col4f(getColor4f()[0], getColor4f()[1], getColor4f()[2], getColor4f()[3]);
        vparams->drawTool()->setPolygonMode(0,vparams->displayFlags().getShowWireFrame());//maybe ??
        vparams->drawTool()->setLightingEnabled(true); //Enable lightning

        for (sofa::Size i=0; i<size; i++){
            vparams->drawTool()->drawCapsule(point1(i),point2(i),(float)radius(i),col4f);
        }

        vparams->drawTool()->setLightingEnabled(false); //Disable lightning
    }

    if (getPrevious()!=nullptr && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);

    vparams->drawTool()->setPolygonMode(0,false);
}


template <class MyReal>
typename CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Real CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::defaultRadius() const
{
    return this->d_default_radius.getValue();
}

template <class MyReal>
const typename CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord & CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::center(sofa::Index i)const{
    return DataTypes::getCPos((_mstate->read(core::ConstVecCoordId::position())->getValue())[i]);
}

template <class MyReal>
typename CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Real CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::radius(sofa::Index i) const
{
    return this->d_capsule_radii.getValue()[i];
}

template <class MyReal>
typename CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::point1(sofa::Index i) const
{
    return  center(i) - axis(i) * height(i)/2.0;
}

template <class MyReal>
typename CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::point2(sofa::Index i) const
{
    return  center(i) + axis(i) * height(i)/2.0;
}

template <class MyReal>
typename TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> >::point1() const
{
    return this->model->point1(this->index);
}

template<class MyReal>
bool TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> >::shareSameVertex(const TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> > & )const{
    return false;
}

template <class MyReal>
typename TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> >::point2() const
{
    return this->model->point2(this->index);
}

template <class MyReal>
typename TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Real TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> >::radius() const
{
    return this->model->radius(this->index);
}


template<class MyReal>
const typename CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord & CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::velocity(sofa::Index index) const {
    return DataTypes::getDPos(((_mstate->read(core::ConstVecDerivId::velocity())->getValue()))[index]);
}


template<class MyReal>
const typename TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord & TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> >::v() const {return this->model->velocity(this->index);}

template<class MyReal>
const sofa::type::Quat<SReal> CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::orientation(sofa::Index index)const{
    return _mstate->read(core::ConstVecCoordId::position())->getValue()[index].getOrientation();
}

template<class MyReal>
typename CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::axis(sofa::Index index) const {
    Coord ax(0,1,0);

    const sofa::type::Quat<SReal> & ori = orientation(index);
    return ori.rotate(ax);
}


template<class MyReal>
typename CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Real CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::height(sofa::Index index) const {
    return ((d_capsule_heights.getValue()))[index];
}

template<class MyReal>
typename TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> >::axis() const {
    return this->model->axis(this->index);
}

template<class MyReal>
Data<typename CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::VecReal > & CapsuleCollisionModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::writeRadii(){
    return d_capsule_radii;
}

} // namespace collisionobbcapsule::geometry
