/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/CubeModel.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/simulation/Simulation.h>
#include <SofaBaseCollision/CapsuleModel.h>
namespace sofa
{

namespace component
{

namespace collision
{

template<class DataTypes>
TCapsuleModel<DataTypes>::TCapsuleModel():
      _capsule_radii(initData(&_capsule_radii, "listCapsuleRadii","Radius of each capsule")),
      _default_radius(initData(&_default_radius,(Real)0.5,"defaultRadius","The default radius")),
      _mstate(NULL)
{
    enum_type = CAPSULE_TYPE;
}

template<class DataTypes>
TCapsuleModel<DataTypes>::TCapsuleModel(core::behavior::MechanicalState<DataTypes>* mstate):
    _capsule_radii(initData(&_capsule_radii, "listCapsuleRadii","Radius of each capsule")),
    _default_radius(initData(&_default_radius,(Real)0.5,"defaultRadius","The default radius")),
    _mstate(mstate)
{
    enum_type = CAPSULE_TYPE;
}

template<class DataTypes>
void TCapsuleModel<DataTypes>::resize(int size)
{
    this->core::CollisionModel::resize(size);
    _capsule_points.resize(size);

    VecReal & capsule_radii = *_capsule_radii.beginEdit();

    if ((int)capsule_radii.size() < size)
    {
        while((int)capsule_radii.size() < size)
            capsule_radii.push_back(_default_radius.getValue());
    }
    else
    {
        capsule_radii.reserve(size);
    }

    _capsule_radii.endEdit();
}


template<class DataTypes>
void TCapsuleModel<DataTypes>::init()
{
    this->CollisionModel::init();
    _mstate = dynamic_cast< core::behavior::MechanicalState<DataTypes>* > (getContext()->getMechanicalState());
    if (_mstate==NULL)
    {
        serr<<"TCapsuleModel requires a Vec3 Mechanical Model" << sendl;
        return;
    }

    core::topology::BaseMeshTopology *bmt = getContext()->getMeshTopology();
    if (!bmt)
    {
        serr <<"CapsuleModel requires a MeshTopology" << sendl;
        return;
    }

    int nbEdges = bmt->getNbEdges();
    resize( nbEdges );

    for(int i = 0; i < nbEdges; ++i)
    {
        _capsule_points[i].first = bmt->getEdge(i)[0];
        _capsule_points[i].second= bmt->getEdge(i)[1];
    }
}

template <class DataTypes>
unsigned int TCapsuleModel<DataTypes>::nbCap()const
{
    return _capsule_radii.getValue().size();
}

template <class DataTypes>
void TCapsuleModel<DataTypes>::computeBoundingTree(int maxDepth)
{
    using namespace sofa::defaulttype;
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int ncap = getContext()->getMeshTopology()->getNbEdges();
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
        typename TCapsule<DataTypes>::Real r;

        //const typename TCapsule<DataTypes>::Real distance = (typename TCapsule<DataTypes>::Real)this->proximity.getValue();
        for (int i=0; i<ncap; i++)
        {
            const Coord p1 = point1(i);
            const Coord p2 = point2(i);
            r = radius(i);

            Vector3 maxVec;
            Vector3 minVec;

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


template<class DataTypes>
void TCapsuleModel<DataTypes>::draw(const core::visual::VisualParams* vparams,int index)
{
    sofa::defaulttype::Vec<4,float> col4f(getColor4f());
    vparams->drawTool()->drawCapsule(point1(index),point2(index),(float)radius(index),col4f);
}

template<class DataTypes>
void TCapsuleModel<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (vparams->displayFlags().getShowCollisionModels())
    {
        sofa::defaulttype::Vec<4,float> col4f(getColor4f());
        vparams->drawTool()->setPolygonMode(0,vparams->displayFlags().getShowWireFrame());//maybe ??
        vparams->drawTool()->setLightingEnabled(true); //Enable lightning

        // Check topological modifications
        //const int npoints = _mstate->getSize()/2;

        for (int i=0; i<size; i++){
            vparams->drawTool()->drawCapsule(point1(i),point2(i),(float)radius(i),col4f);
        }

        vparams->drawTool()->setLightingEnabled(false); //Disable lightning
    }

    if (getPrevious()!=NULL && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);

    vparams->drawTool()->setPolygonMode(0,false);
}


template <class DataTypes>
typename TCapsuleModel<DataTypes>::Real TCapsuleModel<DataTypes>::defaultRadius() const
{
    return this->_default_radius.getValue();
}

template <class DataTypes>
inline const typename TCapsuleModel<DataTypes>::Coord & TCapsuleModel<DataTypes>::point(int i)const{
    return this->_mstate->read(core::ConstVecCoordId::position())->getValue()[i];
}

template <class DataTypes>
typename TCapsuleModel<DataTypes>::Real TCapsuleModel<DataTypes>::radius(int i) const
{
    return this->_capsule_radii.getValue()[i];
}

template <class DataTypes>
const typename TCapsuleModel<DataTypes>::Coord & TCapsuleModel<DataTypes>::point1(int i) const
{
    return  point(_capsule_points[i].first);
}

template <class DataTypes>
const typename TCapsuleModel<DataTypes>::Coord & TCapsuleModel<DataTypes>::point2(int i) const
{
    return  point(_capsule_points[i].second);
}

template <class DataTypes>
int TCapsuleModel<DataTypes>::point1Index(int i) const
{
    return  _capsule_points[i].first;
}

template <class DataTypes>
int TCapsuleModel<DataTypes>::point2Index(int i) const
{
    return  _capsule_points[i].second;
}

template <class DataTypes>
typename TCapsule<DataTypes>::Coord TCapsule<DataTypes>::point1() const
{
    return this->model->point1(this->index);
}

template <class DataTypes>
typename TCapsule<DataTypes>::Coord TCapsule<DataTypes>::point2() const
{
    return this->model->point2(this->index);
}

template <class DataTypes>
typename TCapsule<DataTypes>::Real TCapsule<DataTypes>::radius() const
{
    return this->model->radius(this->index);
}


template<class DataTypes>
typename TCapsuleModel<DataTypes>::Deriv TCapsuleModel<DataTypes>::velocity(int index) const { return ((_mstate->read(core::ConstVecDerivId::velocity())->getValue())[_capsule_points[index].first] +
                                                                                       (_mstate->read(core::ConstVecDerivId::velocity())->getValue())[_capsule_points[index].second])/2.0;}

template<class DataTypes>
typename TCapsule<DataTypes>::Coord TCapsule<DataTypes>::v() const {return this->model->velocity(this->index);}

template<class DataTypes>
typename TCapsuleModel<DataTypes>::Coord TCapsuleModel<DataTypes>::axis(int index) const {
    Coord ax(point2(index) - point1(index));
    ax.normalize();
    return ax;
}

template<class DataTypes>
bool TCapsuleModel<DataTypes>::shareSameVertex(int i1,int i2)const{
    return _capsule_points[i1].first == _capsule_points[i2].first || _capsule_points[i1].first == _capsule_points[i2].second ||
            _capsule_points[i1].second == _capsule_points[i2].first || _capsule_points[i1].second == _capsule_points[i2].second;
}

template<class DataTypes>
bool TCapsule<DataTypes>::shareSameVertex(const TCapsule<DataTypes> & other)const{
    return (this->model == other.model) && this->model->shareSameVertex(this->index,other.index);
}

template<class DataTypes>
sofa::defaulttype::Quaternion TCapsuleModel<DataTypes>::orientation(int index) const {
    Coord ax(point2(index) - point1(index));
    ax.normalize();

    Coord x1(1,0,0);
    Coord x2(0,1,0);

    Coord rx1,rx2;

    if((rx1 = cross(x1,ax)).norm2() > 1e-6){
        rx1.normalize();
        rx2 = cross(rx1,ax);
    }
    else{
        rx1 = cross(x2,ax);
        rx2 = cross(rx1,ax);
    }

    return sofa::defaulttype::Quaternion::createQuaterFromFrame(rx1,ax,rx2);
}

template<class DataTypes>
typename TCapsuleModel<DataTypes>::Coord TCapsuleModel<DataTypes>::center(int index) const {
    return (point2(index) + point1(index))/2;
}

template<class DataTypes>
typename TCapsuleModel<DataTypes>::Real TCapsuleModel<DataTypes>::height(int index) const {
    return (point2(index) - point1(index)).norm();
}

template<class DataTypes>
typename TCapsule<DataTypes>::Coord TCapsule<DataTypes>::axis() const {
    return this->model->axis(this->index);
}

template<class DataTypes>
Data<typename TCapsuleModel<DataTypes>::VecReal > & TCapsuleModel<DataTypes>::writeRadii(){
    return _capsule_radii;
}

}
}
}
