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
#pragma once
#include <CollisionOBBCapsule/geometry/OBBModel.h>

#include <sofa/component/collision/geometry/CubeModel.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

namespace collisionobbcapsule::geometry
{

template<class DataTypes>
OBBCollisionModel<DataTypes>::OBBCollisionModel():
    ext(initData(&ext,"extents","Extents in x,y and z directions")),
    default_ext(initData(&default_ext,(Real)(1.0), "defaultExtent","Default extent")),
    _mstate(nullptr)
{
    enum_type = OBB_TYPE;
}

template<class DataTypes>
OBBCollisionModel<DataTypes>::OBBCollisionModel(sofa::core::behavior::MechanicalState<DataTypes>* mstate):
    ext(initData(&ext, "extents","Extents in x,y and z directions")),
    default_ext(initData(&default_ext,(Real)(1.0), "defaultExtent","Default extent")),
    _mstate(mstate)
{
    enum_type = OBB_TYPE;
}


template<class DataTypes>
void OBBCollisionModel<DataTypes>::init()
{
    this->CollisionModel::init();
    _mstate = dynamic_cast< core::behavior::MechanicalState<DataTypes>* > (getContext()->getMechanicalState());
    if (_mstate==nullptr)
    {
        msg_error()<<"OBBCollisionModel requires a Rigid Mechanical Model";
        return;
    }

    const auto npoints = _mstate->getSize();
    resize(npoints);
}


template<class DataTypes>
void OBBCollisionModel<DataTypes>::resize(sofa::Size size){
    this->core::CollisionModel::resize(size);

    VecCoord & vext = *(ext.beginEdit());

    if (vext.size() < size)
    {
        while(vext.size() < size)
            vext.push_back(Coord(default_ext.getValue(),default_ext.getValue(),default_ext.getValue()));
    }
    else
    {
        vext.reserve(size);
    }

    ext.endEdit();
}


template<class DataTypes>
void OBBCollisionModel<DataTypes>::computeBoundingTree(int maxDepth){
    sofa::component::collision::geometry::CubeCollisionModel* cubeModel = createPrevious<sofa::component::collision::geometry::CubeCollisionModel>();
    const auto npoints = _mstate->getSize();
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
        const typename OBBCollisionModel<DataTypes>::Real distance = (typename OBBCollisionModel<DataTypes>::Real)this->proximity.getValue();

        std::vector<Coord> vs;
        vs.reserve(8);
        for (sofa::Size i=0; i<size; i++)
        {
            vs.clear();
            vertices(i,vs);

            Coord minElem = vs[0];
            Coord maxElem = vs[0];

            for(int j = 1 ; j < 8 ; ++j){
                for(int jj = 0 ; jj < 3 ; ++jj){
                    if(minElem[jj] > vs[j][jj])
                        minElem[jj] = vs[j][jj];
                    else if(maxElem[jj] < vs[j][jj])
                        maxElem[jj] = vs[j][jj];
                }
            }

            for(int jj = 0 ; jj < 3 ; ++jj){
                minElem[jj] -= distance;
                maxElem[jj] += distance;
            }

            cubeModel->setParentOf(i, minElem, maxElem);
        }
        cubeModel->computeBoundingTree(maxDepth);
    }
}


template<class DataTypes>
void OBBCollisionModel<DataTypes>::draw(const sofa::core::visual::VisualParams* vparams, sofa::Index index){

    using namespace sofa::type;
    using namespace sofa::defaulttype;

    std::vector<Coord> p;
    vertices(index,p);

    sofa::type::RGBAColor col4f(getColor4f()[0], getColor4f()[1], getColor4f()[2], getColor4f()[3]);

    std::vector<Vec3> n;
    n.push_back(axis(index,1));
    n.push_back(axis(index,0));
    n.push_back(-n[0]);
    n.push_back(-n[2]);
    n.push_back(n.front());

    std::vector<Vec3> points;
    points.push_back(p[3]);
    points.push_back(p[0]);
    points.push_back(p[2]);
    points.push_back(p[1]);
    points.push_back(p[6]);
    points.push_back(p[5]);
    points.push_back(p[7]);
    points.push_back(p[4]);
    points.push_back(p[3]);
    points.push_back(p[0]);

    vparams->drawTool()->drawTriangleStrip(points,n, col4f);

    n.clear();
    points.clear();

    points.push_back(p[1]);
    points.push_back(p[0]);
    points.push_back(p[5]);
    points.push_back(p[4]);

    n.push_back(-axis(index,2));
    n.push_back(n.back());

    vparams->drawTool()->drawTriangleStrip(points,n,col4f);

    n.clear();
    points.clear();

    points.push_back(p[6]);
    points.push_back(p[7]);
    points.push_back(p[2]);
    points.push_back(p[3]);

    n.push_back(axis(index,2));
    n.push_back(n.back());

    vparams->drawTool()->drawTriangleStrip(points,n,col4f);
}

template<class DataTypes>
void OBBCollisionModel<DataTypes>::draw(const sofa::core::visual::VisualParams* vparams){
    if (vparams->displayFlags().getShowCollisionModels())
    {
        vparams->drawTool()->setPolygonMode(0,vparams->displayFlags().getShowWireFrame());

        const auto npoints = _mstate->getSize();
        vparams->drawTool()->setLightingEnabled(true); //Enable lightning
        for(sofa::Size i = 0 ; i < npoints ; ++i )
            draw(vparams,i);
        vparams->drawTool()->setLightingEnabled(false); //Disable lightning
    }

    if (getPrevious()!=nullptr && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);

    vparams->drawTool()->setPolygonMode(0,false);
}


template <class DataTypes>
inline typename TOBB<DataTypes>::Coord TOBB<DataTypes>::generalCoordinates(const Coord &c)const{
    return this->model->generalCoordinates(c,this->index);
}

template <class DataTypes>
inline bool TOBB<DataTypes>::onSurface(const Coord &c)const{
    Coord loc = this->localCoordinates(c);
    for(int i = 0 ; i < 3 ; ++i){
        if(loc[i] > extent(i) + 1e-6 || loc[i] < - extent(i) - 1e-6)
            return false;
        else if(fabs(this->extent(i) - fabs(loc[i])) < 1e-6)
            return true;
    }

    return false;
}

template <class DataTypes>
inline typename OBBCollisionModel<DataTypes>::Coord OBBCollisionModel<DataTypes>::generalCoordinates(const Coord &c, sofa::Index index)const{
    return orientation(index).rotate(c) + center(index);
}


template <class DataTypes>
inline typename OBBCollisionModel<DataTypes>::Coord OBBCollisionModel<DataTypes>::localCoordinates(const Coord &c, sofa::Index index)const{
    return orientation(index).inverseRotate(c - center(index));
}

template <class DataTypes>
inline typename TOBB<DataTypes>::Coord TOBB<DataTypes>::localCoordinates(const Coord & c)const{
    return this->model->localCoordinates(c,this->index);
}

template <class DataTypes>
inline const typename OBBCollisionModel<DataTypes>::Coord & OBBCollisionModel<DataTypes>::lvelocity(sofa::Index index)const{
    return (_mstate->read(core::vec_id::read_access::velocity)->getValue())[index].getLinear();
}

template <class DataTypes>
inline const typename TOBB<DataTypes>::Coord & TOBB<DataTypes>::v()const{
    return this->model->lvelocity(this->index);
}

template<class DataTypes>
inline typename OBBCollisionModel<DataTypes>::Coord OBBCollisionModel<DataTypes>::axis(sofa::Index index,int dim)const{
    Coord unit;
    if(dim == 0){
        unit[0] = 1;
        unit[1] = 0;
        unit[2] = 0;
    }
    else if(dim == 1){
        unit[0] = 0;
        unit[1] = 1;
        unit[2] = 0;
    }
    else{
        unit[0] = 0;
        unit[1] = 0;
        unit[2] = 1;
    }

    return orientation(index).rotate(unit);
}

template<class DataTypes>
inline typename OBBCollisionModel<DataTypes>::Coord OBBCollisionModel<DataTypes>::vertex(sofa::Index index,int num)const{
    Real s0 = extent(index,0);
    Real s1 = extent(index,1);
    Real s2 = extent(index,2);

    if(num == 0){
        s0*=-1;
        s1*=1;
        s2*=-1;
    }
    else if(num == 1){
        s0*=1;
        s1*=1;
        s2*=-1;
    }
    else if(num == 2){
        s0*=1;
        s1*=1;
        s2*=1;
    }
    else if(num == 3){
        s0*=-1;
        s1*=1;
        s2*=1;
    }
    else if(num == 4){
        s0*=-1;
        s1*=-1;
        s2*=-1;
    }
    else if(num == 5){
        s0*=1;
        s1*=-1;
        s2*=-1;
    }
    else if(num == 6){
        s0*=1;
        s1*=-1;
        s2*=1;
    }
    else{
        s0*=-1;
        s1*=-1;
        s2*=1;
    }

    return center(index) + s0 * axis(index,0) + s1 * axis(index,1) + s2 * axis(index,2);
}

template<class DataTypes>
inline void OBBCollisionModel<DataTypes>::axes(sofa::Index index,Coord * v_axes)const{
    v_axes[0] = axis(index,0);
    v_axes[1] = axis(index,1);
    v_axes[2] = axis(index,2);
}

template<class DataTypes>
inline void TOBB<DataTypes>::axes(Coord * v_axes)const{
    this->model->axes(this->index,v_axes);
}

template<class DataTypes>
inline void OBBCollisionModel<DataTypes>::vertices(sofa::Index index,std::vector<Coord> & vs)const{
    Coord a0(axis(index,0) * extent(index,0));
    Coord a1(axis(index,1) * extent(index,1));
    Coord a2(axis(index,2) * extent(index,2));

    const Coord & c = center(index);

    vs.push_back(c + a0 - a1 - a2);
    vs.push_back(c + a0 + a1 - a2);
    vs.push_back(c + a0 + a1 + a2);
    vs.push_back(c + a0 - a1 + a2);
    vs.push_back(c - a0 - a1 - a2);
    vs.push_back(c - a0 + a1 - a2);
    vs.push_back(c - a0 + a1 + a2);
    vs.push_back(c - a0 - a1 + a2);
}

template<class DataTypes>
inline const typename OBBCollisionModel<DataTypes>::Coord & OBBCollisionModel<DataTypes>::center(sofa::Index index)const{
    return _mstate->read(core::vec_id::read_access::position)->getValue()[index].getCenter();
}

template<class DataTypes>
inline const typename OBBCollisionModel<DataTypes>::Quaternion & OBBCollisionModel<DataTypes>::orientation(sofa::Index index)const{
    return _mstate->read(core::vec_id::read_access::position)->getValue()[index].getOrientation();
}

template<class DataTypes>
inline typename OBBCollisionModel<DataTypes>::Real OBBCollisionModel<DataTypes>::extent(sofa::Index index,int dim)const{
    return ((ext.getValue())[index])[dim];
}

template<class DataTypes>
inline const typename OBBCollisionModel<DataTypes>::Coord & OBBCollisionModel<DataTypes>::extents(sofa::Index index)const{
    return (ext.getValue())[index];
}

template<class DataTypes>
inline typename TOBB<DataTypes>::Coord TOBB<DataTypes>::axis(int dim)const{
    return this->model->axis(this->index,dim);
}

template<class DataTypes>
inline typename TOBB<DataTypes>::Real TOBB<DataTypes>::extent(int dim)const{
    return this->model->extent(this->index,dim);
}

template<class DataTypes>
inline const typename TOBB<DataTypes>::Coord & TOBB<DataTypes>::extents()const{
    return this->model->extents(this->index);
}

template<class DataTypes>
inline const typename TOBB<DataTypes>::Coord & TOBB<DataTypes>::center()const{
    return this->model->center(this->index);
}

template<class DataTypes>
inline const typename TOBB<DataTypes>::Quaternion & TOBB<DataTypes>::orientation()const{
    return this->model->orientation(this->index);
}

template <class DataTypes>
inline Data<typename OBBCollisionModel<DataTypes>::VecCoord> & OBBCollisionModel<DataTypes>::writeExtents(){return ext;}

template <class DataTypes>
inline void TOBB<DataTypes>::vertices(std::vector<Coord> & vs)const{return this->model->vertices(this->index,vs);}

template <class DataTypes>
inline void TOBB<DataTypes>::showVertices()const{
    std::vector<Coord> vs;
    vertices(vs);

    std::stringstream tmpmsg ;
    tmpmsg << "vertices:"<< msgendl;
    for(unsigned int i = 0 ; i < vs.size() ; ++i)
        tmpmsg<<"-"<<vs[i]<<msgendl;
    dmsg_info("TOBB<DataTypes") << tmpmsg.str() ;
}

template <class DataTypes>
void OBBCollisionModel<DataTypes>::computeBBox(const core::ExecParams*, bool onlyVisible)
{
    if( !onlyVisible ) return;


    static const Real max_real = std::numeric_limits<Real>::max();
    static const Real min_real = std::numeric_limits<Real>::lowest();
    Real maxBBox[3] = {min_real,min_real,min_real};
    Real minBBox[3] = {max_real,max_real,max_real};


    std::vector<Coord> p;
    const auto npoints = _mstate->getSize();
    for(sofa::Size i = 0 ; i < npoints ; ++i )
    {
        vertices(i,p);
        for (unsigned int j=0; j<8; j++)
        {
            for (int c=0; c<3; c++)
            {
                if (p[j][c] > maxBBox[c]) maxBBox[c] = Real(p[j][c]);
                if (p[j][c] < minBBox[c]) minBBox[c] = Real(p[j][c]);
            }
        }
    }

    this->f_bbox.setValue(sofa::type::TBoundingBox<Real>(minBBox,maxBBox));

}

} // namespace collisionobbcapsule::geometry
