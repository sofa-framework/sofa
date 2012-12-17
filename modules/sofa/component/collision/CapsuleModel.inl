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
#include <sofa/component/collision/CapsuleModel.h>
namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace helper;


template<class DataTypes>
TCapsuleModel<DataTypes>::TCapsuleModel():
      _capsule_radii(initData(&_capsule_radii, "listCapsuleRadii","Radius of each capsule")),
      _default_radius(initData(&_default_radius,(Real)0.5,"defaultRadius","The default radius")),
      _mstate(NULL)
{
}

template<class DataTypes>
TCapsuleModel<DataTypes>::TCapsuleModel(core::behavior::MechanicalState<DataTypes>* mstate):
    _capsule_radii(initData(&_capsule_radii, "listCapsuleRadii","Radius of each capsule")),
    _default_radius(initData(&_default_radius,(Real)0.5,"defaultRadius","The default radius")),
    _mstate(mstate)
{
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
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int ncap = _mstate->getX()->size()/2;
    bool updated = false;
    if (ncap != size)
    {
        resize(ncap);
        updated = true;
        cubeModel->resize(0);
    }

    if (!isMoving() && !cubeModel->empty() && !updated){
        std::cout<<"immobile..."<<std::endl;
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
    Vec<4,float> col4f(getColor4f());
    vparams->drawTool()->drawCapsule(point1(index),point2(index),(float)radius(index),col4f);
}

template<class DataTypes>
void TCapsuleModel<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (vparams->displayFlags().getShowCollisionModels())
    {
        Vec<4,float> col4f(getColor4f());
        vparams->drawTool()->setPolygonMode(0,vparams->displayFlags().getShowWireFrame());//maybe ??
        vparams->drawTool()->setLightingEnabled(true); //Enable lightning

        // Check topological modifications
        const int npoints = _mstate->getX()->size()/2;

        for (int i=0; i<npoints; i++){
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
    return (*(_mstate->getX()))[i];
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


}
}
}

