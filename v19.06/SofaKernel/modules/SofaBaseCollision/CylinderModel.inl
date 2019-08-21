/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/proximity.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseCollision/CubeModel.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/Simulation.h>

namespace sofa
{

namespace component
{

namespace collision
{

using sofa::core::objectmodel::ComponentState;


template<class DataTypes>
TCylinderModel<DataTypes>::TCylinderModel():
      d_cylinder_radii(initData(&d_cylinder_radii, "radii","Radius of each cylinder")),
      d_cylinder_heights(initData(&d_cylinder_heights,"heights","The cylinder heights")),
      d_default_radius(initData(&d_default_radius,Real(0.5),"defaultRadius","The default radius")),
      d_default_height(initData(&d_default_height,Real(2),"defaultHeight","The default height")),
      d_default_local_axis(initData(&d_default_local_axis,typename DataTypes::Vec3(0.0, 1.0, 0.0),"defaultLocalAxis", "The default local axis cylinder is modeled around")),
      m_mstate(NULL)
{
    enum_type = CYLINDER_TYPE;
}

template<class DataTypes>
TCylinderModel<DataTypes>::TCylinderModel(core::behavior::MechanicalState<DataTypes>* mstate)
    : TCylinderModel()
{
    m_mstate = mstate;
    enum_type = CYLINDER_TYPE;
}

template<class DataTypes>
void TCylinderModel<DataTypes>::resize(int size)
{
    this->core::CollisionModel::resize(size);

    VecReal & Cylinder_radii = *d_cylinder_radii.beginEdit();
    VecReal & Cylinder_heights = *d_cylinder_heights.beginEdit();
    VecAxisCoord & Cylinder_local_axes = *d_cylinder_local_axes.beginEdit();

    if (int(Cylinder_radii.size()) < size)
    {
        while(int(Cylinder_radii.size())< size)
            Cylinder_radii.push_back(d_default_radius.getValue());
    }
    else
    {
        Cylinder_radii.reserve(size);
    }

    if (int(Cylinder_heights.size()) < size)
    {
        while(int(Cylinder_heights.size()) < size)
            Cylinder_heights.push_back(d_default_height.getValue());
    }
    else
    {
        Cylinder_heights.reserve(size);
    }

    if (int(Cylinder_local_axes.size()) < size)
    {
        while(int(Cylinder_local_axes.size()) < size)
            Cylinder_local_axes.push_back(d_default_local_axis.getValue());
    }
    else
    {
        Cylinder_local_axes.reserve(size);
    }

    d_cylinder_radii.endEdit();
    d_cylinder_heights.endEdit();
    d_cylinder_local_axes.endEdit();
}

template<class DataTypes>
void TCylinderModel<DataTypes>::init()
{
    this->CollisionModel::init();
    m_mstate = dynamic_cast< core::behavior::MechanicalState<DataTypes>* > (getContext()->getMechanicalState());
    if (m_mstate==NULL)
    {
        msg_error() << "TCylinderModel requires a Rigid Mechanical Model";
        m_componentstate = ComponentState::Invalid;
        return;
    }

    resize(m_mstate->getSize());
}


template<class DataTypes>
void TCylinderModel<DataTypes>::computeBoundingTree(int maxDepth)
{
    using namespace sofa::defaulttype;
    CubeModel* cubeModel = createPrevious<CubeModel>();
    const int ncyl = m_mstate->getSize();

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
        typename TCylinder<DataTypes>::Real r;
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


template<class DataTypes>
void TCylinderModel<DataTypes>::draw(const core::visual::VisualParams* vparams,int i)
{
    using namespace sofa::defaulttype;
    Vec<4,float> colour(getColor4f());
    SReal h2 = height(i)/2.0;

    Vector3 p1(center(i));
    Vector3 p2(center(i));

    Vector3 ax = axis(i);

    p1 += h2 * ax;
    p2 -= h2 * ax;

    vparams->drawTool()->drawCylinder(p2,p1,float(radius(i)),colour);
}

template<class DataTypes>
void TCylinderModel<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (vparams->displayFlags().getShowCollisionModels())
    {

        for (int i=0; i<size; i++){
            draw(vparams,i);
        }

    }

    if (getPrevious()!=NULL && vparams->displayFlags().getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);
}


template<class DataTypes>
typename TCylinderModel<DataTypes>::Real TCylinderModel< DataTypes >::defaultRadius() const
{
    return this->d_default_radius.getValue();
}

template<class DataTypes>
const typename TCylinderModel<DataTypes>::Coord & TCylinderModel< DataTypes >::center(int i)const{
    return DataTypes::getCPos((m_mstate->read(core::ConstVecCoordId::position())->getValue())[i]);
}

template<class DataTypes>
typename TCylinderModel<DataTypes>::Real TCylinderModel< DataTypes >::radius(int i) const
{
    return this->d_cylinder_radii.getValue()[i];
}

template<class DataTypes>
typename TCylinderModel<DataTypes>::Coord TCylinderModel< DataTypes >::point1(int i) const
{
    return  center(i) - axis(i) * height(i)/2.0;
}

template<class DataTypes>
typename TCylinderModel<DataTypes>::Coord TCylinderModel< DataTypes >::point2(int i) const
{
    return  center(i) + axis(i) * height(i)/2.0;
}

template<class DataTypes>
typename TCylinder<DataTypes>::Coord TCylinder< DataTypes >::point1() const
{
    return this->model->point1(this->index);
}

template<class DataTypes>
typename TCylinder<DataTypes>::Coord TCylinder<DataTypes >::point2() const
{
    return this->model->point2(this->index);
}

template<class DataTypes>
typename TCylinder<DataTypes>::Real TCylinder<DataTypes >::radius() const
{
    return this->model->radius(this->index);
}


template<class DataTypes>
const typename TCylinderModel<DataTypes>::Coord & TCylinderModel<DataTypes >::velocity(int index) const {
    return DataTypes::getDPos(((m_mstate->read(core::ConstVecDerivId::velocity())->getValue()))[index]);
}


template<class DataTypes>
const typename TCylinder<DataTypes>::Coord & TCylinder<DataTypes >::v() const {return this->model->velocity(this->index);}

template<class DataTypes>
const sofa::defaulttype::Quaternion TCylinderModel<DataTypes >::orientation(int index)const{
    return m_mstate->read(core::ConstVecCoordId::position())->getValue()[index].getOrientation();
}

template<class DataTypes>
typename TCylinderModel<DataTypes>::Coord TCylinderModel<DataTypes >::axis(int index) const {
    Coord ax = d_cylinder_local_axes.getValue()[index];

    const sofa::defaulttype::Quaternion & ori = orientation(index);
    return ori.rotate(ax);
}

template<class DataTypes>
typename TCylinderModel<DataTypes>::Coord TCylinderModel<DataTypes>::local_axis(int index) const {
    Coord ax = d_cylinder_local_axes.getValue()[index];
    return ax;
}

template<class DataTypes>
typename TCylinderModel<DataTypes>::Real TCylinderModel<DataTypes>::height(int index) const {
    return ((d_cylinder_heights.getValue()))[index];
}

template<class DataTypes>
 typename TCylinder<DataTypes>::Coord TCylinder<DataTypes >::axis() const {
    return this->model->axis(this->index);
}

template<class DataTypes>
Data< typename TCylinderModel<DataTypes>::VecReal> & TCylinderModel<DataTypes >::writeRadii(){
    return d_cylinder_radii;
}

template<class DataTypes>
Data< typename TCylinderModel<DataTypes>::VecReal > & TCylinderModel<DataTypes >::writeHeights(){
    return d_cylinder_heights;
}

template<class DataTypes>
Data< typename TCylinderModel<DataTypes>::VecAxisCoord > & TCylinderModel<DataTypes >::writeLocalAxes(){
    return d_cylinder_local_axes;
}

}
}
}
