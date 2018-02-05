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
#include <SofaBaseCollision/CylinderModel.h>

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
TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::TCylinderModel():
      _cylinder_radii(initData(&_cylinder_radii, "radii","Radius of each cylinder")),
      _cylinder_heights(initData(&_cylinder_heights,"heights","The cylinder heights")),
      _default_radius(initData(&_default_radius,(Real)0.5,"defaultRadius","The default radius")),
      _default_height(initData(&_default_height,(Real)2,"defaultHeight","The default height")),
      _default_local_axis(initData(&_default_local_axis,typename DataTypes::Vec3(0.0, 1.0, 0.0),"defaultLocalAxis", "The default local axis cylinder is modeled around")),
      _mstate(NULL)
{
    enum_type = CYLINDER_TYPE;
}

template<class MyReal>
TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::TCylinderModel(core::behavior::MechanicalState<DataTypes>* mstate):
    _cylinder_radii(initData(&_cylinder_radii, "radii","Radius of each cylinder")),
    _cylinder_heights(initData(&_cylinder_heights,"heights","The cylinder heights")),
    _default_radius(initData(&_default_radius,(Real)0.5,"defaultRadius","The default radius")),
    _default_height(initData(&_default_height,(Real)2,"dafaultHeight","The default height")),
    _default_local_axis(initData(&_default_local_axis,typename DataTypes::Vec3(0.0, 1.0, 0.0),"defaultLocalAxis", "The default local axis cylinder is modeled around")),
    _mstate(mstate)
{
    enum_type = CYLINDER_TYPE;
}

template<class MyReal>
void TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::resize(int size)
{
    this->core::CollisionModel::resize(size);

    VecReal & Cylinder_radii = *_cylinder_radii.beginEdit();
    VecReal & Cylinder_heights = *_cylinder_heights.beginEdit();
    VecAxisCoord & Cylinder_local_axes = *_cylinder_local_axes.beginEdit();

    if ((int)Cylinder_radii.size() < size)
    {
        while((int)Cylinder_radii.size() < size)
            Cylinder_radii.push_back(_default_radius.getValue());
    }
    else
    {
        Cylinder_radii.reserve(size);
    }

    if ((int)Cylinder_heights.size() < size)
    {
        while((int)Cylinder_heights.size() < size)
            Cylinder_heights.push_back(_default_height.getValue());
    }
    else
    {
        Cylinder_heights.reserve(size);
    }

    if ((int)Cylinder_local_axes.size() < size)
    {
        while((int)Cylinder_local_axes.size() < size)
            Cylinder_local_axes.push_back(_default_local_axis.getValue());
    }
    else
    {
        Cylinder_local_axes.reserve(size);
    }

    _cylinder_radii.endEdit();
    _cylinder_heights.endEdit();
    _cylinder_local_axes.endEdit();
}


template<class MyReal>
void TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::init()
{
    this->CollisionModel::init();
    _mstate = dynamic_cast< core::behavior::MechanicalState<DataTypes>* > (getContext()->getMechanicalState());
    if (_mstate==NULL)
    {
        serr<<"TCylinderModel requires a Rigid Mechanical Model" << sendl;
        return;
    }

    resize(_mstate->getSize());
}


template <class MyReal>
void TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::computeBoundingTree(int maxDepth)
{
    using namespace sofa::defaulttype;
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int ncyl = _mstate->getSize();

    bool updated = false;
    if (ncyl != size)
    {
        resize(ncyl);
        updated = true;
        cubeModel->resize(0);
    }

    if (!isMoving() && !cubeModel->empty() && !updated){
        return; // No need to recompute BBox if immobile
    }

    Vector3 minVec,maxVec;
    cubeModel->resize(ncyl);
    if (!empty())
    {
        typename TCylinder<StdRigidTypes<3,MyReal> >::Real r;

        //const typename TCylinder<StdRigidTypes<3,MyReal> >::Real distance = (typename TCylinder<StdRigidTypes<3,MyReal> >::Real)this->proximity.getValue();
        for (int i=0; i<ncyl; i++)
        {
            r = radius(i);
            SReal h2 = height(i)/2.0;

            Vector3 p1(center(i));
            Vector3 p2(center(i));

            Vector3 ax = axis(i);

            p1 += h2 * ax;
            p2 -= h2 * ax;

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
void TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::draw(const core::visual::VisualParams* vparams,int i)
{
    using namespace sofa::defaulttype;
    Vec<4,float> colour(getColor4f());
    SReal h2 = height(i)/2.0;

    Vector3 p1(center(i));
    Vector3 p2(center(i));

    Vector3 ax = axis(i);

    p1 += h2 * ax;
    p2 -= h2 * ax;

    vparams->drawTool()->drawCylinder(p2,p1,(float)radius(i),colour);
}

template<class MyReal>
void TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::draw(const core::visual::VisualParams* vparams)
{
    if (vparams->displayFlags().getShowCollisionModels())
    {
        //vparams->drawTool()->setPolygonMode(0,vparams->displayFlags().getShowWireFrame());//maybe ??
        //vparams->drawTool()->setLightingEnabled(true); //Enable lightning

        // Check topological modifications
        //const int npoints = _mstate->getSize()/2;

        for (int i=0; i<size; i++){
            draw(vparams,i);
        }

        //vparams->drawTool()->setLightingEnabled(false); //Disable lightning
    }

    if (getPrevious()!=NULL && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);

    //vparams->drawTool()->setPolygonMode(0,false);
}


template <class MyReal>
typename TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Real TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::defaultRadius() const
{
    return this->_default_radius.getValue();
}

template <class MyReal>
const typename TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord & TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::center(int i)const{
    return DataTypes::getCPos((_mstate->read(core::ConstVecCoordId::position())->getValue())[i]);
}

template <class MyReal>
typename TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Real TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::radius(int i) const
{
    return this->_cylinder_radii.getValue()[i];
}

template <class MyReal>
typename TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::point1(int i) const
{
    return  center(i) - axis(i) * height(i)/2.0;
}

template <class MyReal>
typename TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::point2(int i) const
{
    return  center(i) + axis(i) * height(i)/2.0;
}

template <class MyReal>
typename TCylinder<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord TCylinder<sofa::defaulttype::StdRigidTypes<3,MyReal> >::point1() const
{
    return this->model->point1(this->index);
}

template <class MyReal>
typename TCylinder<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord TCylinder<sofa::defaulttype::StdRigidTypes<3,MyReal> >::point2() const
{
    return this->model->point2(this->index);
}

template <class MyReal>
typename TCylinder<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Real TCylinder<sofa::defaulttype::StdRigidTypes<3,MyReal> >::radius() const
{
    return this->model->radius(this->index);
}


template<class MyReal>
const typename TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord & TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::velocity(int index) const {
    return DataTypes::getDPos(((_mstate->read(core::ConstVecDerivId::velocity())->getValue()))[index]);
}


template<class MyReal>
const typename TCylinder<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord & TCylinder<sofa::defaulttype::StdRigidTypes<3,MyReal> >::v() const {return this->model->velocity(this->index);}

template<class MyReal>
const sofa::defaulttype::Quaternion TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::orientation(int index)const{
    return _mstate->read(core::ConstVecCoordId::position())->getValue()[index].getOrientation();
}

template<class MyReal>
typename TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::axis(int index) const {
    Coord ax = _cylinder_local_axes.getValue()[index];

    const sofa::defaulttype::Quaternion & ori = orientation(index);
    return ori.rotate(ax);
}

template<class MyReal>
typename TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::local_axis(int index) const {
    Coord ax = _cylinder_local_axes.getValue()[index];
    return ax;
}

template<class MyReal>
typename TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Real TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::height(int index) const {
    return ((_cylinder_heights.getValue()))[index];
}

template<class MyReal>
typename TCylinder<sofa::defaulttype::StdRigidTypes<3,MyReal> >::Coord TCylinder<sofa::defaulttype::StdRigidTypes<3,MyReal> >::axis() const {
    return this->model->axis(this->index);
}

template<class MyReal>
Data<typename TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::VecReal > & TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::writeRadii(){
    return _cylinder_radii;
}

template<class MyReal>
Data<typename TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::VecReal > & TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::writeHeights(){
    return _cylinder_heights;
}

template<class MyReal>
Data<typename TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::VecAxisCoord > & TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> >::writeLocalAxes(){
    return _cylinder_local_axes;
}

}
}
}
