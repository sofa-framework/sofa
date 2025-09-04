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
#include <sofa/core/config.h>
#include <iosfwd>


namespace sofa::helper::visual { class DrawTool; }

namespace sofa::core
{


class BaseState;
class BaseMapping;
class BehaviorModel;
class CollisionModel;
class CollisionElementIterator;
class DataEngine;

class ExecParams;
class ConstraintParams;
namespace constraintparams
{
SOFA_CORE_API const ConstraintParams* defaultInstance();
SOFA_CORE_API ExecParams* castToExecParams(sofa::core::ConstraintParams*);
SOFA_CORE_API const ExecParams* castToExecParams(const sofa::core::ConstraintParams*);
}

class ExecParams;
namespace execparams
{
SOFA_CORE_API ExecParams* defaultInstance();
}

class MechanicalParams;
namespace mechanicalparams
{
SOFA_CORE_API const MechanicalParams* defaultInstance();
SOFA_CORE_API ExecParams* castToExecParams(sofa::core::MechanicalParams*);
SOFA_CORE_API const ExecParams* castToExecParams(const sofa::core::MechanicalParams*);

SOFA_CORE_API SReal kFactor(const sofa::core::MechanicalParams*);
SOFA_CORE_API SReal bFactor(const sofa::core::MechanicalParams*);
SOFA_CORE_API SReal kFactorIncludingRayleighDamping(const sofa::core::MechanicalParams*, SReal d);
SOFA_CORE_API SReal mFactorIncludingRayleighDamping(const sofa::core::MechanicalParams*, SReal d);
SOFA_CORE_API SReal dt(const sofa::core::MechanicalParams*);
}

}

namespace sofa::core::objectmodel
{
class Base;
class BaseObject;
class BaseNode;
class BaseContext;
class BaseData;
class BaseLink;
class BaseClass;
class AbstractDataLink;
class Event;
class BaseClass;
class AbstractDataLink;
class ContextObject;
class ConfigurationSetting;

template<class TOwnerType, class TDestType, unsigned TFlags>
class SingleLink;

class Tag;
SOFA_CORE_API std::ostream& operator<<(std::ostream& o, const Tag& t);
SOFA_CORE_API std::istream& operator>>(std::istream& i, Tag& t);
}


namespace sofa::core::behavior
{
class BaseForceField;
class BaseMass;
class BaseMechanicalState;
class BaseAnimationLoop;
class BaseLagrangianConstraint;
class BaseConstraintSet;
class ConstraintSolver;
class ConstraintResolution;
class OdeSolver;
class BaseLinearSolver;
class BaseInteractionForceField;
class BaseProjectiveConstraintSet;
class BaseInteractionProjectiveConstraintSet;
class BaseInteractionConstraint;
class BaseMechanicalState;
class LinearSolver;
class MultiMatrixAccessor;

template<class T>
class MechanicalState;
}

namespace sofa::core::topology
{
class BaseMeshTopology;
class BaseTopologyObject;
class Topology;
class TopologyHandler;
class TopologyChange;
SOFA_CORE_API std::ostream& operator<< ( std::ostream& out, const sofa::core::topology::TopologyChange* t );
SOFA_CORE_API std::istream& operator>> ( std::istream& in, sofa::core::topology::TopologyChange*& t );
SOFA_CORE_API std::istream& operator>> ( std::istream& in, const sofa::core::topology::TopologyChange*& );
}

namespace sofa::core::visual
{
class VisualLoop;
class VisualModel;
class VisualManager;
class Shader;

class FlagTreeItem;
SOFA_CORE_API std::ostream& operator<< ( std::ostream& os, const FlagTreeItem& root );
SOFA_CORE_API std::istream& operator>> ( std::istream& in, FlagTreeItem& root );

class DisplayFlags;
SOFA_CORE_API std::ostream& operator<< ( std::ostream& os, const DisplayFlags& flags );
SOFA_CORE_API std::istream& operator>> ( std::istream& in, DisplayFlags& flags );

using sofa::helper::visual::DrawTool;

class VisualParams;
namespace visualparams
{
SOFA_CORE_API VisualParams* defaultInstance();

SOFA_CORE_API ExecParams* castToExecParams(sofa::core::visual::VisualParams*);
SOFA_CORE_API const ExecParams* castToExecParams(const sofa::core::visual::VisualParams*);

SOFA_CORE_API sofa::core::visual::DrawTool* getDrawTool(VisualParams* params);
SOFA_CORE_API sofa::core::visual::DisplayFlags& getDisplayFlags(VisualParams* params);
SOFA_CORE_API sofa::core::visual::DrawTool* getDrawTool(const VisualParams* params);
SOFA_CORE_API const sofa::core::visual::DisplayFlags& getDisplayFlags(const VisualParams* params);
}
}

namespace sofa::core::collision
{
class CollisionGroupManager;
class ContactManager;
class Detection;
class Intersection;
class Pipeline;
}

namespace sofa::core::loader
{
class BaseLoader;
}

namespace sofa::core
{
/////////////////////////////// CORE::OPAQUE FUNCTION /////////////////////////////////////////////////
///
/// CORE::OPAQUE function are a groupe of function that make "opaque" some of the common sofa behaviors.
///
/// Core::Opaque functions are:
///     - Base* sofa::core::castToBase(T*) replace dynamic_cast<Base*>(T*);
///     - T* sofa::core::castTo(Base*) replace dynamic_cast<T*>(Base*);
///     - sofa:core::objectmodel::base::GetClass<T>() replace T::GetClass();
///
/// These functions are called "opaque" as they work with only forward declaration of the involved
/// types in comparison to class methods the requires the full class declaration to be used.
///
/// It is highly recommended to use as much as possible opaque function in header files as this
/// allow to reduce the dependency tree.
///
/// Opaque function may be slower at runtime (by one function call) but this is true only if LTO isn't
/// able to optimize them properly. If you have experience/feedback with LTO please join the discussion
/// in https://github.com/sofa-framework/sofa/discussions/1822
///////////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////
/// Defines the baseline functions for a type all the types in-herit from Base.
/// These are non-opaque function that needs to be specialized in order to implement
/// an opaque version for a given type.

/// Dynamic cast from the type parameter B* into Base*
template<class Source>
sofa::core::objectmodel::Base* castToBase(Source*b){ return b; }

/// Dynamic cast from Base* into the type parameter Dest
template<class Dest>
Dest castTo(sofa::core::objectmodel::Base* base){ return dynamic_cast<Dest>(base); }

namespace objectmodel::base
{
/// Returns the BaseClass* from type parameter B, hiding B::GetClass()
template<class B>
const sofa::core::objectmodel::BaseClass* GetClass(){return B::GetClass(); }
}
///////////////////////////////////////////////////////////////////////////////////////

/// Declares the opaque function signature for a type that in-herit from Base.
///
/// Example of use:
/// Doing:
///     SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(MyType)
/// Will add the following functions:
///     MyType* castTo<MyType>(sofa::core::objectmodel::Base*)
///     sofa::core::objectmodel::Base* castToBase(MyType*)
///     BaseClass* sofa::core::objectmodel::base::GetClass()
///
/// Once declare it is mandatory to also define the same functions.
/// For that you must use SOFA_DEFINE_OPAQUE_FUNCTION_BETWEEN_BASE_AND
#define SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(TYPENAME) \
    template<> SOFA_CORE_API TYPENAME* castTo(sofa::core::objectmodel::Base* base); \
    SOFA_CORE_API sofa::core::objectmodel::Base* castToBase(TYPENAME* b); \
    namespace objectmodel::base { template<> SOFA_CORE_API const sofa::core::objectmodel::BaseClass* GetClass<TYPENAME>(); }

/// Define the opaque function signature for a type that in-herit from Base.
///
/// Example of use:
/// Doing:
///     SOFA_DEFINE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(MyType)
/// Will add the following functions:
///     MyType* dynamiCastBaseTo(sofa::core::objectmodel::Base*) { ... }
///     sofa::core::objectmodel::Base* dynamiCastBaseFrom(MyType*) {... }
///     BaseClass* sofa::core::objectmodel::base::GetClass() { ... }
///
#define SOFA_DEFINE_OPAQUE_FUNCTIONS_BETWEEN_BASE_AND(TYPENAME) \
    template<> \
    TYPENAME* castTo(sofa::core::objectmodel::Base* base) \
{ return dynamic_cast<TYPENAME*>(base); } \
    sofa::core::objectmodel::Base* castToBase(TYPENAME* b) \
{ return b; } \
    namespace objectmodel::base { template<> const sofa::core::objectmodel::BaseClass* GetClass<TYPENAME>() \
{ return TYPENAME::GetClass(); } }

SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::BaseState);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::BaseMapping);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::BehaviorModel);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::CollisionModel);

SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::objectmodel::BaseObject);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::objectmodel::ContextObject);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::objectmodel::ConfigurationSetting);

SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::behavior::BaseAnimationLoop);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::behavior::BaseMass);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::behavior::OdeSolver);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::behavior::ConstraintSolver);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::behavior::BaseLinearSolver);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::behavior::BaseMechanicalState);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::behavior::BaseForceField);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::behavior::BaseInteractionForceField);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::behavior::BaseProjectiveConstraintSet);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::behavior::BaseConstraintSet);

SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::topology::Topology);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::topology::BaseMeshTopology);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::topology::BaseTopologyObject);

SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::collision::Pipeline);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::visual::VisualLoop);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::visual::Shader);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::visual::VisualModel);
SOFA_DECLARE_OPAQUE_FUNCTION_BETWEEN_BASE_AND(sofa::core::visual::VisualManager);
}

namespace sofa::core::objectmodel::basecontext
{
SOFA_CORE_API SReal getDt(sofa::core::objectmodel::BaseContext* context);
SOFA_CORE_API SReal getTime(sofa::core::objectmodel::BaseContext* context);
}

namespace sofa::core::objectmodel::basecontext
{
SOFA_CORE_API SReal getDt(sofa::core::objectmodel::BaseContext* context);
SOFA_CORE_API SReal getTime(sofa::core::objectmodel::BaseContext* context);
}
