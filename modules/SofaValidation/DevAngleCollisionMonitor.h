/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_MISC_DEVANGLECOLLISIONMONITOR_H
#define SOFA_COMPONENT_MISC_DEVANGLECOLLISIONMONITOR_H
#include "config.h"

#include <sofa/helper/vector.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <SofaValidation/DevMonitor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <SofaMeshCollision/PointModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaBaseCollision/NewProximityIntersection.h>
#include <SofaBaseCollision/BruteForceDetection.h>

namespace sofa
{

namespace component
{

namespace misc
{

template <class TDataTypes>
class DevAngleCollisionMonitor: public virtual DevMonitor<sofa::defaulttype::Vec1Types>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(DevAngleCollisionMonitor,TDataTypes), SOFA_TEMPLATE(DevMonitor,sofa::defaulttype::Vec1Types));

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;

    Data < Real > maxDist;
protected:
    DevAngleCollisionMonitor();
    virtual ~DevAngleCollisionMonitor() { };
public:
    void init();
    void eval();

    /// Retrieve the associated MechanicalState (First model)
    core::behavior::MechanicalState<DataTypes>* getMState1() { return mstate1; }
    core::behavior::BaseMechanicalState* getMechModel1() { return mstate1; }

    /// Retrieve the associated MechanicalState (Second model)
    core::behavior::MechanicalState<defaulttype::Vec3Types>* getMState2() { return mstate2; }
    core::behavior::BaseMechanicalState* getMechModel2() { return mstate2; }


    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (arg->getAttribute("object1") || arg->getAttribute("object2"))
        {
            if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(arg->findObject(arg->getAttribute("object1",".."))) == NULL)
                return false;
            if (dynamic_cast<core::behavior::MechanicalState<defaulttype::Vec3Types>*>(arg->findObject(arg->getAttribute("object2",".."))) == NULL)
                return false;
        }
        else
        {
            if (dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
                return false;
        }
        return core::objectmodel::BaseObject::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T* tObj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        typename T::SPtr obj = core::objectmodel::BaseObject::create(tObj, context, arg);

        if (arg && (arg->getAttribute("object1") || arg->getAttribute("object2")))
        {
            obj->mstate1 = dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(arg->findObject(arg->getAttribute("object1","..")));
            obj->mstate2 = dynamic_cast<core::behavior::MechanicalState<defaulttype::Vec3Types>*>(arg->findObject(arg->getAttribute("object2","..")));
        }

        return obj;
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const DevAngleCollisionMonitor<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:
    /// First model mechanical state
    core::behavior::MechanicalState<DataTypes> *mstate1;
    /// Second model mechanical state
    core::behavior::MechanicalState<defaulttype::Vec3Types> *mstate2;

    /// Point model of first object
    sofa::component::collision::PointModel *pointsCM;
    /// Surface model of second object
    sofa::component::collision::TriangleModel *surfaceCM;

    sofa::component::collision::NewProximityIntersection::SPtr intersection;
    sofa::component::collision::BruteForceDetection::SPtr detection;
    typedef core::collision::TDetectionOutputVector< sofa::component::collision::TriangleModel, sofa::component::collision::PointModel> ContactVector;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MISC_DEVANGLECOLLISIONMONITOR_CPP)
#ifndef SOFA_FLOAT
extern template class DevAngleCollisionMonitor<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class DevAngleCollisionMonitor<defaulttype::Rigid3fTypes>;
#endif
#endif

} // namespace misc

} // namespace component

} // namespace sofa

#endif
