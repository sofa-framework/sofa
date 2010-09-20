/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_COLLISION_SPHEREMODEL_H
#define SOFA_COMPONENT_COLLISION_SPHEREMODEL_H

#include <sofa/core/CollisionModel.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/component/component.h>
#include <sofa/defaulttype/VecTypes.h>

#include <vector>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;

template<class DataTypes>
class TSphereModel;

template<class TDataTypes>
class TSphere : public core::TCollisionElementIterator< TSphereModel<TDataTypes> >
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::Real   Real;
    typedef typename DataTypes::Coord Coord;
    typedef TSphereModel<DataTypes> ParentModel;

    TSphere(ParentModel* model, int index);

    explicit TSphere(core::CollisionElementIterator& i);

    const Coord& center() const;
    const Coord& p() const;
    const Coord& pFree() const;
    const Coord& v() const;
    Real r() const;
};

template< class TDataTypes>
class TSphereModel : public core::CollisionModel
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(TSphereModel, TDataTypes), core::CollisionModel);
    typedef TDataTypes DataTypes;
    typedef DataTypes InDataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename  DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecReal VecReal;
    typedef TSphere<DataTypes> Element;
    friend class TSphere<DataTypes>;

    TSphereModel();

    TSphereModel(core::behavior::MechanicalState<TDataTypes>* _mstate );

    virtual void init();

    // -- CollisionModel interface

    virtual void resize(int size);

    virtual void computeBoundingTree(int maxDepth=0);

    virtual void computeContinuousBoundingTree(double dt, int maxDepth=0);

    void draw(int index);

    void draw();

    virtual void drawColourPicking(const ColourCode method);

    virtual sofa::defaulttype::Vector3 getPositionFromWeights(int index, Real /*a*/, Real /*b*/, Real /*c*/);

    core::behavior::MechanicalState<DataTypes>* getMechanicalState() { return mstate; }

    virtual bool load(const char* filename);

    int addSphere(const Vector3& pos, Real r);
    void setSphere(int i, const Vector3& pos, Real r);

    const VecReal& getR() const { return this->radius.getValue(); }

    Real getRadius(const int i) const;
    void setRadius(const int i, const Real r);
    void setRadius(const Real r);

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::behavior::MechanicalState<TDataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

    template<class T>
    static void create(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        core::behavior::MechanicalState<TDataTypes>* _mstate = NULL;
        if( context)
        {
            _mstate = dynamic_cast<core::behavior::MechanicalState<TDataTypes>*>(context->getMechanicalState());
            if (_mstate) obj = new T(_mstate);
            else obj = new T();
            context->addObject(obj);
        }
        if (arg) obj->parse(arg);
    }


    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const TSphereModel<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:

    core::behavior::MechanicalState<DataTypes>* mstate;

    Data< VecReal > radius;
    Data< SReal > defaultRadius;

    sofa::core::objectmodel::DataFileName filename;

    class Loader;
};

template<class DataTypes>
inline TSphere<DataTypes>::TSphere(ParentModel* model, int index)
    : core::TCollisionElementIterator<ParentModel>(model, index)
{}

template<class DataTypes>
inline TSphere<DataTypes>::TSphere(core::CollisionElementIterator& i)
    : core::TCollisionElementIterator<ParentModel>(static_cast<ParentModel*>(i.getCollisionModel()), i.getIndex())
{
}

template<class DataTypes>
inline const typename DataTypes::Coord& TSphere<DataTypes>::center() const { return (*this->model->mstate->getX())[this->index]; }

template<class DataTypes>
inline const typename DataTypes::Coord& TSphere<DataTypes>::p() const { return (*this->model->mstate->getX())[this->index]; }

template<class DataTypes>
inline const typename DataTypes::Coord& TSphere<DataTypes>::pFree() const { return (*this->model->mstate->getXfree())[this->index]; }

template<class DataTypes>
inline const typename DataTypes::Coord& TSphere<DataTypes>::v() const { return (*this->model->mstate->getV())[this->index]; }

template<class DataTypes>
inline typename DataTypes::Real TSphere<DataTypes>::r() const { return (Real) this->model->getRadius((unsigned)this->index); }

using namespace sofa::defaulttype;
typedef TSphereModel<Vec3Types> SphereModel;
typedef TSphere<Vec3Types> Sphere;

#if defined(WIN32) && !defined(SOFA_BUILD_COMPONENT_COLLISION)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_COLLISION_API TSphereModel<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_COLLISION_API TSphereModel<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace collision

} // namespace component

} // namespace sofa

#endif
