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
#include <SofaBaseCollision/RigidCapsuleModel.h>

#include <sofa/helper/system/config.h>
#include <sofa/helper/proximity.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <iostream>
#include <algorithm>

#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/CubeModel.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/simulation/Simulation.h>

namespace sofa
{

namespace component
{

namespace collision
{

template<class MyReal>
TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::TCapsuleModel():
      _capsule_radii(initData(&_capsule_radii, "radii","Radius of each capsule")),
      _capsule_heights(initData(&_capsule_heights,"heights","The capsule heights")),
      _default_radius(initData(&_default_radius,(Real)0.5,"defaultRadius","The default radius")),
      _default_height(initData(&_default_height,(Real)2,"dafaultHeight","The default height")),
      _mstate(NULL)
{
    enum_type = CAPSULE_TYPE;
}

template<class MyReal>
TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::TCapsuleModel(core::behavior::MechanicalState<DataTypes>* mstate):
    _capsule_radii(initData(&_capsule_radii, "radii","Radius of each capsule")),
    _capsule_heights(initData(&_capsule_heights,"heights","The capsule heights")),
    _default_radius(initData(&_default_radius,(Real)0.5,"defaultRadius","The default radius")),
    _default_height(initData(&_default_height,(Real)2,"dafaultHeight","The default height")),
    _mstate(mstate)
{
    enum_type = CAPSULE_TYPE;
}

template<class MyReal>
void TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::resize(int size)
{
    this->core::CollisionModel::resize(size);

    VecReal & capsule_radii = *_capsule_radii.beginEdit();
    VecReal & capsule_heights = *_capsule_heights.beginEdit();

    if ((int)capsule_radii.size() < size)
    {
        while((int)capsule_radii.size() < size)
            capsule_radii.push_back(_default_radius.getValue());
    }
    else
    {
        capsule_radii.reserve(size);
    }

    if ((int)capsule_heights.size() < size)
    {
        while((int)capsule_heights.size() < size)
            capsule_heights.push_back(_default_height.getValue());
    }
    else
    {
        capsule_heights.reserve(size);
    }

    _capsule_radii.endEdit();
    _capsule_heights.endEdit();
}


template<class MyReal>
void TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::init()
{
    this->CollisionModel::init();
    _mstate = dynamic_cast< core::behavior::MechanicalState<DataTypes>* > (getContext()->getMechanicalState());
    if (_mstate==NULL)
    {
        serr<<"TCapsuleModel requires a Rigid Mechanical Model" << sendl;
        return;
    }

    resize(_mstate->getSize());
}

template <class MyReal>
unsigned int TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::nbCap()const
{
    return _capsule_radii.getValue().size();
}

template <class MyReal>
void TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int ncap = _mstate->getSize();

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
        for (int i=0; i<ncap; i++)
        {
            const Coord p1 = point1(i);
            const Coord p2 = point2(i);
            r = radius(i);

            defaulttype::Vector3 maxVec;
            defaulttype::Vector3 minVec;

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
void TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::draw(const core::visual::VisualParams* vparams,int index)
{
    sofa::defaulttype::Vec<4,float> col4f(getColor4f());
    vparams->drawTool()->drawCapsule(point1(index),point2(index),(float)radius(index),col4f);
}

template<class MyReal>
void TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::draw(const core::visual::VisualParams* vparams)
{
    if (vparams->displayFlags().getShowCollisionModels())
    {
        sofa::defaulttype::Vec<4,float> col4f(getColor4f());
        vparams->drawTool()->setPolygonMode(0,vparams->displayFlags().getShowWireFrame());//maybe ??
        vparams->drawTool()->setLightingEnabled(true); //Enable lightning

        // Check topological modifications
        for (int i=0; i<size; i++){
            vparams->drawTool()->drawCapsule(point1(i),point2(i),(float)radius(i),col4f);
        }

        vparams->drawTool()->setLightingEnabled(false); //Disable lightning
    }

    if (getPrevious()!=NULL && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);

    vparams->drawTool()->setPolygonMode(0,false);
}


template <class MyReal>
typename TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Real TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::defaultRadius() const
{
    return this->_default_radius.getValue();
}

template <class MyReal>
const typename TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord & TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::center(int i)const{
    return DataTypes::getCPos((_mstate->read(core::ConstVecCoordId::position())->getValue())[i]);
}

template <class MyReal>
typename TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Real TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::radius(int i) const
{
    return this->_capsule_radii.getValue()[i];
}

template <class MyReal>
typename TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::point1(int i) const
{
    return  center(i) - axis(i) * height(i)/2.0;
}

template <class MyReal>
typename TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::point2(int i) const
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
const typename TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord & TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::velocity(int index) const {
    return DataTypes::getDPos(((_mstate->read(core::ConstVecDerivId::velocity())->getValue()))[index]);
}


template<class MyReal>
const typename TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord & TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> >::v() const {return this->model->velocity(this->index);}

template<class MyReal>
const sofa::defaulttype::Quaternion TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::orientation(int index)const{
    return _mstate->read(core::ConstVecCoordId::position())->getValue()[index].getOrientation();
}

template<class MyReal>
typename TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::axis(int index) const {
    Coord ax(0,1,0);

    const sofa::defaulttype::Quaternion & ori = orientation(index);
    return ori.rotate(ax);
}


template<class MyReal>
typename TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Real TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::height(int index) const {
    return ((_capsule_heights.getValue()))[index];
}

template<class MyReal>
typename TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord TCapsule<sofa::defaulttype::StdRigidTypes<3,MyReal> >::axis() const {
    return this->model->axis(this->index);
}

template<class MyReal>
Data<typename TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::VecReal > & TCapsuleModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::writeRadii(){
    return _capsule_radii;
}

}
}
}
