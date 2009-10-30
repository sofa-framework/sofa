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
#ifndef SOFA_COMPONENT_MISC_DEVANGLECOLLISIONMONITOR_H
#define SOFA_COMPONENT_MISC_DEVANGLECOLLISIONMONITOR_H

#include <sofa/helper/vector.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/component/misc/DevMonitor.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/component/collision/PointModel.h>
#include <sofa/component/collision/TriangleModel.h>
#include <sofa/component/collision/NewProximityIntersection.h>
#include <sofa/component/collision/BruteForceDetection.h>

namespace sofa
{

namespace component
{

namespace misc
{

template <class TDataTypes>
class DevAngleCollisionMonitor: public virtual DevMonitor<sofa::defaulttype::Vec1dTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(DevAngleCollisionMonitor,TDataTypes), SOFA_TEMPLATE(DevMonitor,sofa::defaulttype::Vec1dTypes));

    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;

    Data < Real > maxDist;

    DevAngleCollisionMonitor();
    virtual ~DevAngleCollisionMonitor() { };

    void init();
    void eval();

    /// Retrieve the associated MechanicalState (First model)
    core::componentmodel::behavior::MechanicalState<DataTypes>* getMState1() { return mstate1; }
    core::componentmodel::behavior::BaseMechanicalState* getMechModel1() { return mstate1; }

    /// Retrieve the associated MechanicalState (Second model)
    core::componentmodel::behavior::MechanicalState<defaulttype::Vec3dTypes>* getMState2() { return mstate2; }
    core::componentmodel::behavior::BaseMechanicalState* getMechModel2() { return mstate2; }


    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (arg->getAttribute("object1") || arg->getAttribute("object2"))
        {
            if (dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>*>(arg->findObject(arg->getAttribute("object1",".."))) == NULL)
                return false;
            if (dynamic_cast<core::componentmodel::behavior::MechanicalState<defaulttype::Vec3dTypes>*>(arg->findObject(arg->getAttribute("object2",".."))) == NULL)
                return false;
        }
        else
        {
            if (dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
                return false;
        }
        return core::objectmodel::BaseObject::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static void create(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        core::objectmodel::BaseObject::create(obj, context, arg);
        if (arg && (arg->getAttribute("object1") || arg->getAttribute("object2")))
        {
            obj->mstate1 = dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>*>(arg->findObject(arg->getAttribute("object1","..")));
            obj->mstate2 = dynamic_cast<core::componentmodel::behavior::MechanicalState<defaulttype::Vec3dTypes>*>(arg->findObject(arg->getAttribute("object2","..")));
        }
        else if (context)
        {
            /*            obj->mstate1 =
                        obj->mstate2 =
                        dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState());*/
        }
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
    core::componentmodel::behavior::MechanicalState<DataTypes> *mstate1;
    /// Second model mechanical state
    core::componentmodel::behavior::MechanicalState<defaulttype::Vec3dTypes> *mstate2;

    /// Point model of first object
    sofa::component::collision::PointModel *pointsCM;
    /// Surface model of second object
    sofa::component::collision::TriangleModel *surfaceCM;

    sofa::component::collision::NewProximityIntersection * intersection;
    sofa::component::collision::BruteForceDetection* detection;
    typedef core::componentmodel::collision::TDetectionOutputVector< sofa::component::collision::TriangleModel, sofa::component::collision::PointModel> ContactVector;
};

} // namespace misc

} // namespace component

} // namespace sofa

#endif
