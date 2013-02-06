#ifndef RIGIDSPHEREMODEL_H
#define RIGIDSPHEREMODEL_H
/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/core/CollisionModel.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/component/component.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/accessor.h>
//#include <sofa/component/collision/RigidContactMapper.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

template <class TReal>
class TRigidSphereModel;

template<class TReal>
class TRigidSphere : public core::TCollisionElementIterator< TRigidSphereModel<TReal> >
{
public:
    typedef StdRigidTypes<3,TReal> DataTypes;
    typedef typename DataTypes::Coord Coord;
    typedef typename Coord::Pos Pos;
    typedef typename DataTypes::Quat Quaternion;
    typedef typename DataTypes::VecCoord VecCoord;

    typedef TRigidSphereModel<TReal > ParentModel;

    TRigidSphere(ParentModel* model, int index);

    explicit TRigidSphere(core::CollisionElementIterator& i);

    const Pos& center() const;
    const Pos& p() const;
    const Quaternion& orientation()const;

    //const Coord& pFree() const;
    //const Coord& v() const;

    /// Return true if the element stores a free position vector
    //bool hasFreePosition() const;

    TReal r() const;
};

template<class TReal>
class TRigidSphereModel : public core::CollisionModel
{
public:
    typedef StdRigidTypes<3,TReal> DataTypes;
    typedef DataTypes InDataTypes;
    SOFA_CLASS(SOFA_TEMPLATE(TRigidSphereModel,TReal),core::CollisionModel);

    typedef typename DataTypes::Coord::Pos Pos;
    typedef typename DataTypes::VecReal VecReal;
    typedef typename DataTypes::Quat Quaternion;
    typedef TRigidSphere<TReal> Element;

    friend class TRigidSphere<TReal>;
protected:
    TRigidSphereModel();

    TRigidSphereModel(core::behavior::MechanicalState<DataTypes>* _mstate );
public:
    virtual void init();

    // -- CollisionModel interface

    virtual void resize(int size);

    virtual void computeBoundingTree(int maxDepth=0);

    //virtual void computeContinuousBoundingTree(double dt, int maxDepth=0);

    void draw(const core::visual::VisualParams*,int index);

    void draw(const core::visual::VisualParams* vparams);


    core::behavior::MechanicalState<DataTypes>* getMechanicalState() { return mstate; }

    const VecReal& getR() const { return this->radius.getValue(); }

    TReal getRadius(const int i) const;

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL && context->getMechanicalState() != NULL)
            return false;

        return BaseObject::canCreate(obj, context, arg);
    }

    template<class T>
    static typename T::SPtr create(T*, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        typename T::SPtr obj;
        core::behavior::MechanicalState<DataTypes>* _mstate = NULL;

        if( context)
        {
            _mstate = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState());
            if (_mstate)
                obj = sofa::core::objectmodel::New<T>(_mstate);
            else
                obj = sofa::core::objectmodel::New<T>();

            context->addObject(obj);
        }

        if (arg) obj->parse(arg);

        return obj;
    }


    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const TRigidSphereModel<TReal>* = NULL)
    {
        return DataTypes::Name();
    }

    Data< VecReal > radius;
    Data< SReal > defaultRadius;

protected:
    core::behavior::MechanicalState<DataTypes>* mstate;
};


template<class TReal>
inline TRigidSphere<TReal>::TRigidSphere(ParentModel* model, int index)
    : core::TCollisionElementIterator<ParentModel>(model, index)
{}

template<class TReal>
inline TRigidSphere<TReal>::TRigidSphere(core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<ParentModel>(static_cast<ParentModel*>(i.getCollisionModel()), i.getIndex())
{
}


template <class TReal>
inline const typename TRigidSphere<TReal>::Pos& TRigidSphere<TReal>::center() const { return (*this->model->mstate->getX())[this->index].getCenter(); }

template <class TReal>
inline const typename TRigidSphere<TReal>::Quaternion& TRigidSphere<TReal>::orientation() const { return (*this->model->mstate->getX())[this->index].getOrientation();}

template <class TReal>
inline const typename TRigidSphere<TReal>::Pos& TRigidSphere<TReal>::p() const { return (*this->model->mstate->getX())[this->index].getCenter(); }

//template<class TReal>
//inline const typename TSphere<DataTypes>::Coord& TSphere<DataTypes>::pFree() const { return (*this->model->mstate->read(core::ConstVecCoordId::freePosition())).getValue()[this->index]; }

//template<class DataTypes>
//inline const typename TSphere<DataTypes>::Coord& TSphere<DataTypes>::v() const { return (*this->model->mstate->getV())[this->index]; }

template<class TReal>
inline TReal TRigidSphere<TReal>::r() const { return (TReal) this->model->getRadius((unsigned)this->index); }

//template<class TReal>
//inline bool TSphere<DataTypes>::hasFreePosition() const { return this->model->mstate->read(core::ConstVecCoordId::freePosition())->isSet(); }


typedef TRigidSphereModel<SReal> RigidSphereModel;
typedef TRigidSphere<SReal> RigidSphere;

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_BUILD_BASE_COLLISION)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_COLLISION_API TRigidSphereModel<double>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_COLLISION_API TRigidSphereModel<float>;
#endif
#endif

}
}
}

#endif // RIGIDSPHEREMODEL_H
