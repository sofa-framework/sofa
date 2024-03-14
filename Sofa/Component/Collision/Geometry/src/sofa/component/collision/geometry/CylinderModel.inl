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
#include <sofa/component/collision/geometry/CylinderModel.h>
#include <sofa/component/collision/geometry/CubeModel.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/DisplayFlags.h>
#include <sofa/helper/visual/DrawTool.h>

namespace sofa::component::collision::geometry
{

using sofa::core::objectmodel::ComponentState;


template<class DataTypes>
CylinderCollisionModel<DataTypes>::CylinderCollisionModel():
      d_cylinder_radii(initData(&d_cylinder_radii, "radii","Radius of each cylinder")),
      d_cylinder_heights(initData(&d_cylinder_heights,"heights","The cylinder heights")),
      d_default_radius(initData(&d_default_radius,Real(0.5),"defaultRadius","The default radius")),
      d_default_height(initData(&d_default_height,Real(2),"defaultHeight","The default height")),
      d_default_local_axis(initData(&d_default_local_axis,typename DataTypes::Vec3(0.0, 1.0, 0.0),"defaultLocalAxis", "The default local axis cylinder is modeled around")),
      m_mstate(nullptr)
{
    enum_type = CYLINDER_TYPE;
}

template<class DataTypes>
CylinderCollisionModel<DataTypes>::CylinderCollisionModel(core::behavior::MechanicalState<DataTypes>* mstate)
    : CylinderCollisionModel()
{
    m_mstate = mstate;
    enum_type = CYLINDER_TYPE;
}

template<class DataTypes>
void CylinderCollisionModel<DataTypes>::resize(sofa::Size size)
{
    this->core::CollisionModel::resize(size);

    VecReal & Cylinder_radii = *d_cylinder_radii.beginEdit();
    VecReal & Cylinder_heights = *d_cylinder_heights.beginEdit();
    VecAxisCoord & Cylinder_local_axes = *d_cylinder_local_axes.beginEdit();

    if (Cylinder_radii.size() < size)
    {
        while(Cylinder_radii.size()< size)
            Cylinder_radii.push_back(d_default_radius.getValue());
    }
    else
    {
        Cylinder_radii.reserve(size);
    }

    if (Cylinder_heights.size() < size)
    {
        while(Cylinder_heights.size() < size)
            Cylinder_heights.push_back(d_default_height.getValue());
    }
    else
    {
        Cylinder_heights.reserve(size);
    }

    if (Cylinder_local_axes.size() < size)
    {
        while(Cylinder_local_axes.size() < size)
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
void CylinderCollisionModel<DataTypes>::init()
{
    this->CollisionModel::init();
    m_mstate = dynamic_cast< core::behavior::MechanicalState<DataTypes>* > (getContext()->getMechanicalState());
    if (m_mstate==nullptr)
    {
        msg_error() << "CylinderCollisionModel requires a Rigid Mechanical Model";
        d_componentState.setValue(ComponentState::Invalid);
        return;
    }

    resize(m_mstate->getSize());
}


template<class DataTypes>
void CylinderCollisionModel<DataTypes>::computeBoundingTree(int maxDepth)
{
    using namespace sofa::type;
    using namespace sofa::defaulttype;
    CubeCollisionModel* cubeModel = createPrevious<CubeCollisionModel>();
    const auto ncyl = m_mstate->getSize();

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

    Vec3 minVec,maxVec;
    cubeModel->resize(ncyl);
    if (!empty())
    {
        typename TCylinder<DataTypes>::Real r;
        for (sofa::Size i=0; i<ncyl; i++)
        {
            r = radius(i);
            SReal h2 = height(i)/2.0;

            Vec3 p1(center(i));
            Vec3 p2(center(i));

            Vec3 ax = axis(i);

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
void CylinderCollisionModel<DataTypes>::draw(const core::visual::VisualParams* vparams, sofa::Index i)
{
    using namespace sofa::type;
    using namespace sofa::defaulttype;
    const sofa::type::RGBAColor colour(getColor4f()[0], getColor4f()[1], getColor4f()[2], getColor4f()[3]);
    const SReal h2 = height(i)/2.0;

    Vec3 p1(center(i));
    Vec3 p2(center(i));

    const Vec3 ax = axis(i);

    p1 += h2 * ax;
    p2 -= h2 * ax;

    sofa::core::visual::visualparams::getDrawTool(vparams)->drawCylinder(p2,p1,float(radius(i)),colour);
}

template<class DataTypes>
void CylinderCollisionModel<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    const auto df = sofa::core::visual::visualparams::getDisplayFlags(vparams);
    if (df.getShowCollisionModels())
    {

        for (sofa::Index i=0; i<size; i++){
            draw(vparams,i);
        }

    }

    if (getPrevious()!=nullptr && df.getShowBoundingCollisionModels())
        getPrevious()->draw(vparams);
}


template<class DataTypes>
typename CylinderCollisionModel<DataTypes>::Real CylinderCollisionModel< DataTypes >::defaultRadius() const
{
    return this->d_default_radius.getValue();
}

template<class DataTypes>
const typename CylinderCollisionModel<DataTypes>::Coord & CylinderCollisionModel< DataTypes >::center(sofa::Index i)const{
    return DataTypes::getCPos((m_mstate->read(core::ConstVecCoordId::position())->getValue())[i]);
}

template<class DataTypes>
typename CylinderCollisionModel<DataTypes>::Real CylinderCollisionModel< DataTypes >::radius(sofa::Index i) const
{
    return this->d_cylinder_radii.getValue()[i];
}

template<class DataTypes>
typename CylinderCollisionModel<DataTypes>::Coord CylinderCollisionModel< DataTypes >::point1(sofa::Index i) const
{
    return  center(i) - axis(i) * height(i)/2.0;
}

template<class DataTypes>
typename CylinderCollisionModel<DataTypes>::Coord CylinderCollisionModel< DataTypes >::point2(sofa::Index i) const
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
const typename CylinderCollisionModel<DataTypes>::Coord & CylinderCollisionModel<DataTypes >::velocity(sofa::Index index) const {
    return DataTypes::getDPos(((m_mstate->read(core::ConstVecDerivId::velocity())->getValue()))[index]);
}


template<class DataTypes>
const typename TCylinder<DataTypes>::Coord & TCylinder<DataTypes >::v() const {return this->model->velocity(this->index);}

template<class DataTypes>
const sofa::type::Quat<SReal> CylinderCollisionModel<DataTypes >::orientation(sofa::Index index)const{
    return m_mstate->read(core::ConstVecCoordId::position())->getValue()[index].getOrientation();
}

template<class DataTypes>
typename CylinderCollisionModel<DataTypes>::Coord CylinderCollisionModel<DataTypes >::axis(sofa::Index index) const {
    Coord ax = d_cylinder_local_axes.getValue()[index];

    const sofa::type::Quat<SReal> & ori = orientation(index);
    return ori.rotate(ax);
}

template<class DataTypes>
typename CylinderCollisionModel<DataTypes>::Coord CylinderCollisionModel<DataTypes>::local_axis(sofa::Index index) const {
    Coord ax = d_cylinder_local_axes.getValue()[index];
    return ax;
}

template<class DataTypes>
typename CylinderCollisionModel<DataTypes>::Real CylinderCollisionModel<DataTypes>::height(sofa::Index index) const {
    return ((d_cylinder_heights.getValue()))[index];
}

template<class DataTypes>
 typename TCylinder<DataTypes>::Coord TCylinder<DataTypes >::axis() const {
    return this->model->axis(this->index);
}

template<class DataTypes>
Data< typename CylinderCollisionModel<DataTypes>::VecReal> & CylinderCollisionModel<DataTypes >::writeRadii(){
    return d_cylinder_radii;
}

template<class DataTypes>
Data< typename CylinderCollisionModel<DataTypes>::VecReal > & CylinderCollisionModel<DataTypes >::writeHeights(){
    return d_cylinder_heights;
}

template<class DataTypes>
Data< typename CylinderCollisionModel<DataTypes>::VecAxisCoord > & CylinderCollisionModel<DataTypes >::writeLocalAxes(){
    return d_cylinder_local_axes;
}

} // namespace sofa::component::collision::geometry
