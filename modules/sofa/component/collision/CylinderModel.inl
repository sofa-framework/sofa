#include <sofa/component/collision/CylinderModel.h>

#include <sofa/helper/system/config.h>
#include <sofa/helper/proximity.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/collision/Intersection.inl>
#include <iostream>
#include <algorithm>

#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/collision/CubeModel.h>
#include <sofa/core/ObjectFactory.h>

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


template<class MyReal>
TCylinderModel<StdRigidTypes<3,MyReal> >::TCylinderModel():
      _cylinder_radii(initData(&_cylinder_radii, "radii","Radius of each cylinder")),
      _cylinder_heights(initData(&_cylinder_heights,"heights","The cylinder heights")),
      _default_radius(initData(&_default_radius,(Real)0.5,"defaultRadius","The default radius")),
      _default_height(initData(&_default_height,(Real)2,"dafaultHeight","The default height")),
      _mstate(NULL)
{
    enum_type = CYLINDER_TYPE;
}

template<class MyReal>
TCylinderModel<StdRigidTypes<3,MyReal> >::TCylinderModel(core::behavior::MechanicalState<DataTypes>* mstate):
    _cylinder_radii(initData(&_cylinder_radii, "radii","Radius of each cylinder")),
    _cylinder_heights(initData(&_cylinder_heights,"heights","The cylinder heights")),
    _default_radius(initData(&_default_radius,(Real)0.5,"defaultRadius","The default radius")),
    _default_height(initData(&_default_height,(Real)2,"dafaultHeight","The default height")),
    _mstate(mstate)
{
    enum_type = CYLINDER_TYPE;
}

template<class MyReal>
void TCylinderModel<StdRigidTypes<3,MyReal> >::resize(int size)
{
    this->core::CollisionModel::resize(size);

    VecReal & Cylinder_radii = *_cylinder_radii.beginEdit();
    VecReal & Cylinder_heights = *_cylinder_heights.beginEdit();

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

    _cylinder_radii.endEdit();
    _cylinder_heights.endEdit();
}


template<class MyReal>
void TCylinderModel<StdRigidTypes<3,MyReal> >::init()
{
    this->CollisionModel::init();
    _mstate = dynamic_cast< core::behavior::MechanicalState<DataTypes>* > (getContext()->getMechanicalState());
    if (_mstate==NULL)
    {
        serr<<"TCylinderModel requires a Rigid Mechanical Model" << sendl;
        return;
    }

    resize(_mstate->getX()->size());
}


template <class MyReal>
void TCylinderModel<StdRigidTypes<3,MyReal> >::computeBoundingTree(int maxDepth)
{
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int ncyl = _mstate->getX()->size();

    bool updated = false;
    if (ncyl != size)
    {
        resize(ncyl);
        updated = true;
        cubeModel->resize(0);
    }

    if (!isMoving() && !cubeModel->empty() && !updated){
        std::cout<<"immobile..."<<std::endl;
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
void TCylinderModel<StdRigidTypes<3,MyReal> >::draw(const core::visual::VisualParams* vparams,int i)
{
    Vec<4,float> colour(getColor4f());
    SReal h2 = height(i)/2.0;

    Vector3 p1(center(i));
    Vector3 p2(center(i));

    Vector3 ax = axis(i);

    p1 += h2 * ax;
    p2 -= h2 * ax;

    vparams->drawTool()->drawCylinder(p1,p2,(float)radius(i),colour);
}

template<class MyReal>
void TCylinderModel<StdRigidTypes<3,MyReal> >::draw(const core::visual::VisualParams* vparams)
{
    if (vparams->displayFlags().getShowCollisionModels())
    {
        //vparams->drawTool()->setPolygonMode(0,vparams->displayFlags().getShowWireFrame());//maybe ??
        //vparams->drawTool()->setLightingEnabled(true); //Enable lightning

        // Check topological modifications
        //const int npoints = _mstate->getX()->size()/2;

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
typename TCylinderModel<StdRigidTypes<3,MyReal> >::Real TCylinderModel<StdRigidTypes<3,MyReal> >::defaultRadius() const
{
    return this->_default_radius.getValue();
}

template <class MyReal>
const typename TCylinderModel<StdRigidTypes<3,MyReal> >::Coord & TCylinderModel<StdRigidTypes<3,MyReal> >::center(int i)const{
    return DataTypes::getCPos((*(_mstate->getX()))[i]);
}

template <class MyReal>
typename TCylinderModel<StdRigidTypes<3,MyReal> >::Real TCylinderModel<StdRigidTypes<3,MyReal> >::radius(int i) const
{
    return this->_cylinder_radii.getValue()[i];
}

template <class MyReal>
typename TCylinderModel<StdRigidTypes<3,MyReal> >::Coord TCylinderModel<StdRigidTypes<3,MyReal> >::point1(int i) const
{
    return  center(i) - axis(i) * height(i)/2.0;
}

template <class MyReal>
typename TCylinderModel<StdRigidTypes<3,MyReal> >::Coord TCylinderModel<StdRigidTypes<3,MyReal> >::point2(int i) const
{
    return  center(i) + axis(i) * height(i)/2.0;
}

template <class MyReal>
typename TCylinder<StdRigidTypes<3,MyReal> >::Coord TCylinder<StdRigidTypes<3,MyReal> >::point1() const
{
    return this->model->point1(this->index);
}

template <class MyReal>
typename TCylinder<StdRigidTypes<3,MyReal> >::Coord TCylinder<StdRigidTypes<3,MyReal> >::point2() const
{
    return this->model->point2(this->index);
}

template <class MyReal>
typename TCylinder<StdRigidTypes<3,MyReal> >::Real TCylinder<StdRigidTypes<3,MyReal> >::radius() const
{
    return this->model->radius(this->index);
}


template<class MyReal>
const typename TCylinderModel<StdRigidTypes<3,MyReal> >::Coord & TCylinderModel<StdRigidTypes<3,MyReal> >::velocity(int index) const {
    return DataTypes::getDPos(((*(_mstate->getV())))[index]);
}


template<class MyReal>
const typename TCylinder<StdRigidTypes<3,MyReal> >::Coord & TCylinder<StdRigidTypes<3,MyReal> >::v() const {return this->model->velocity(this->index);}

template<class MyReal>
const Quaternion TCylinderModel<StdRigidTypes<3,MyReal> >::orientation(int index)const{
    return (*_mstate->getX())[index].getOrientation();
}

template<class MyReal>
typename TCylinderModel<StdRigidTypes<3,MyReal> >::Coord TCylinderModel<StdRigidTypes<3,MyReal> >::axis(int index) const {
    Coord ax(0,1,0);

    const Quaternion & ori = orientation(index);
    return ori.rotate(ax);
}


template<class MyReal>
typename TCylinderModel<StdRigidTypes<3,MyReal> >::Real TCylinderModel<StdRigidTypes<3,MyReal> >::height(int index) const {
    return ((_cylinder_heights.getValue()))[index];
}

template<class MyReal>
typename TCylinder<StdRigidTypes<3,MyReal> >::Coord TCylinder<StdRigidTypes<3,MyReal> >::axis() const {
    return this->model->axis(this->index);
}

template<class MyReal>
Data<typename TCylinderModel<StdRigidTypes<3,MyReal> >::VecReal > & TCylinderModel<StdRigidTypes<3,MyReal> >::writeRadii(){
    return _cylinder_radii;
}

}
}
}


