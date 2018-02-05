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
#ifndef SOFA_COMPONENT_COLLISION_CYLINDERMODEL_H
#define SOFA_COMPONENT_COLLISION_CYLINDERMODEL_H
#include "config.h"

#include <sofa/core/CollisionModel.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/accessor.h>


namespace sofa
{

namespace component
{

namespace collision
{

template<class DataTypes>
class TCylinderModel;

template<class DataTypes>
class TCylinder;

/**
  *A Cylinder can be viewed as a segment with a radius, here the segment is
  *defined by its apexes.
  */
template< class MyReal>
class TCylinder<sofa::defaulttype::StdRigidTypes<3,MyReal> > : public core::TCollisionElementIterator< TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> > >
{
public:
    typedef sofa::defaulttype::StdRigidTypes<3,MyReal> DataTypes;
    typedef typename DataTypes::Real   Real;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::CPos Coord;
    typedef typename DataTypes::VecCoord VecCoord;

    typedef TCylinderModel<DataTypes> ParentModel;

    TCylinder(ParentModel* model, int index);

    explicit TCylinder(const core::CollisionElementIterator& i);

    Coord axis()const;

    Real radius() const;

    Coord point1() const;
    Coord point2() const;

    const Coord & v()const;
};


/**
  *CylinderModel templated by RigidTypes (frames), direction is given by Y direction of the frame.
  */
template< class MyReal>
class TCylinderModel<sofa::defaulttype::StdRigidTypes<3,MyReal> > : public core::CollisionModel
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TCylinderModel, SOFA_TEMPLATE2(sofa::defaulttype::StdRigidTypes, 3, MyReal)), core::CollisionModel);


    typedef sofa::defaulttype::StdRigidTypes<3,MyReal> DataTypes;
    typedef DataTypes InDataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename  DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::CPos Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecReal VecReal;
    typedef typename helper::vector<typename DataTypes::Vec3> VecAxisCoord;

    typedef TCylinder<DataTypes> Element;
    friend class TCylinder<DataTypes>;
protected:
    Data<VecReal> _cylinder_radii;
    Data<VecReal> _cylinder_heights;
    Data<VecAxisCoord> _cylinder_local_axes;

    Data<Real> _default_radius;
    Data<Real> _default_height;
    Data<Coord> _default_local_axis;

    TCylinderModel();
    TCylinderModel(core::behavior::MechanicalState<DataTypes>* mstate );
public:
    virtual void init() override;

    // -- CollisionModel interface

    virtual void resize(int size) override;

    virtual void computeBoundingTree(int maxDepth=0) override;

    //virtual void computeContinuousBoundingTree(SReal dt, int maxDepth=0);

    void draw(const core::visual::VisualParams* vparams,int index) override;

    void draw(const core::visual::VisualParams* vparams) override;


    core::behavior::MechanicalState<DataTypes>* getMechanicalState() { return _mstate; }

    Real radius(int index) const;

    const Coord & center(int i)const;

    //Returns the direction of the cylinder at index index
    Coord axis(int index)const;
    //Returns the direction of the cylinder at index in local coordinates
    Coord local_axis(int index) const;

    const sofa::defaulttype::Quaternion orientation(int index)const;

    Real height(int index)const;

    Coord point1(int i) const;

    Coord point2(int i) const;

    Real defaultRadius()const;

    const Coord & velocity(int index)const;

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL && context->getMechanicalState() != NULL)
            return false;

        return BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const TCylinderModel<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    Data<VecReal>& writeRadii();
    Data<VecReal>& writeHeights();
    Data<VecAxisCoord>& writeLocalAxes();

protected:
    core::behavior::MechanicalState<DataTypes>* _mstate;
};


template<class MyReal>
inline TCylinder<sofa::defaulttype::StdRigidTypes<3,MyReal> >::TCylinder(ParentModel* model, int index)
    : core::TCollisionElementIterator<ParentModel>(model, index)
{}

template<class MyReal>
inline TCylinder<sofa::defaulttype::StdRigidTypes<3,MyReal> >::TCylinder(const core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<ParentModel>(static_cast<ParentModel*>(i.getCollisionModel()), i.getIndex())
{
}

typedef TCylinderModel<sofa::defaulttype::Rigid3Types> CylinderModel;
typedef TCylinder<sofa::defaulttype::Rigid3Types> Cylinder;

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_COLLISION_CYLINDERMODEL_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_COLLISION_API TCylinder<defaulttype::Rigid3dTypes>;
extern template class SOFA_BASE_COLLISION_API TCylinderModel<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_COLLISION_API TCylinder<defaulttype::Rigid3fTypes>;
extern template class SOFA_BASE_COLLISION_API TCylinderModel<defaulttype::Rigid3fTypes>;
#endif
#endif

} // namespace collision

} // namespace component

} // namespace sofa

#endif
