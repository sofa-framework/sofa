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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "Binding.h"
#include "Binding_SofaModule.h"

#include "Binding_BaseData.h"
#include "Binding_Base.h"
#include "Binding_BaseContext.h"
#include "Binding_Context.h"
#include "Binding_Node.h"
#include "Binding_GNode.h"
#include "Binding_Vec3.h"
#include "Binding_BaseObjectDescription.h"

#include <Python.h>

PyObject *SofaPythonModule = 0;




void bindSofaPythonModule()
{
    //PyImport_AppendInittab( (char*)"Sofa", &initSofa );

    SofaPythonModule = SP_INIT_MODULE(Sofa)

            SP_ADD_CLASS(SofaPythonModule,BaseData)
            SP_ADD_CLASS(SofaPythonModule,Vec3)
            SP_ADD_CLASS(SofaPythonModule,BaseObjectDescription)

            SP_ADD_CLASS(SofaPythonModule,Base)
            SP_ADD_CLASS(SofaPythonModule,BaseContext)
            SP_ADD_CLASS(SofaPythonModule,Context)
            SP_ADD_CLASS(SofaPythonModule,Node)
            SP_ADD_CLASS(SofaPythonModule,GNode)
            /*
                    SP_ADD_CLASS(SofaPythonModule,BaseObject)
                        SP_ADD_CLASS(SofaPythonModule,BaseController)
                            SP_ADD_CLASS(SofaPythonModule,Controller)
                                SP_ADD_CLASS(SofaPythonModule,ScriptController)
                                    SP_ADD_CLASS(SofaPythonModule,PythonScriptController)
            */
}








#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/helper/Factory.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/component/component.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/component/typedef/MechanicalState_double.h>
#include <sofa/component/typedef/Mass_double.h>
#include <sofa/component/typedef/Particles_double.h>
#include <sofa/gui/GUIManager.h>
#include <sofa/gui/SofaGUI.h>

#include "PythonScriptController.h"

        using namespace sofa::core;
using namespace sofa::core::objectmodel;
using namespace sofa::simulation;
using namespace sofa::simulation::tree;
using namespace sofa::defaulttype;
using namespace sofa::component::odesolver;
using namespace sofa::component::container;
using namespace sofa::component::controller;
using namespace sofa::core::behavior;
using namespace sofa::core::visual;
using namespace sofa::gui;

//sofa::simulation::xml::BaseElement::NodeFactory* getNodeFactory() {return sofa::simulation::xml::BaseElement::NodeFactory::getInstance();}

// ce qui suit est la SEULE manière de binder avec boost.python des fonctions surchargées
// on passe par un pointeur en spécifiant les paramètres
void (Base::*setName1)(const std::string&) = &Base::setName;
//BaseObject* (BaseContext::*getBaseObject)(const std::string& path) const = &BaseContext::get<BaseObject>;




/*
BOOST_PYTHON_MODULE( Sofa )
{
    def ("createObject", createObject);//, return_value_policy<reference_existing_object>());
    def ("getObject", getObject);//, return_value_policy<reference_existing_object>());
    def ("getChildNode", getChildNode);//, return_value_policy<reference_existing_object>());

    // send message to the GUI...
    def ("sendGUIMessage", sendGUIMessage);

    // TODO: double héritage BaseMechanicalState & State<sofa::defaulttype::Vec3Types>
    class_ <MechanicalState<sofa::defaulttype::Vec3Types> , MechanicalState<sofa::defaulttype::Vec3Types>::SPtr, bases<BaseObject>, boost::noncopyable>("MechanicalState", no_init)
            ;

    class_ <MechanicalObject3 , MechanicalObject3::SPtr, bases<MechanicalState<sofa::defaulttype::Vec3Types> >, boost::noncopyable>("MechanicalObject", no_init)
            .def("resize",&MechanicalObject3::resize)
            ;

    // héritage multiple:
    // UniformMass<T> / Mass<T> / ForceField<T>+BaseMass...
    // ...ForceField<T> / BaseForceFiled / BaseObject
    // ...BaseMass / BaseObject
    class_ <UniformMass3 , UniformMass3::SPtr, bases< BaseObject >, boost::noncopyable>("MechanicalObject", no_init)
            .def("setMass",&UniformMass3::setMass)
            ;

    // BaseObjectDescription n'a pas de smart pointer ?? bon bah ok alors
    class_ <BaseObjectDescription>("BaseObjectDescription",init<const char*,const char*>())
            .add_property("name",
                          &BaseObjectDescription::getName,
                          &BaseObjectDescription::setName)

            .def("getAttribute",&BaseObjectDescription::getAttribute)
            .def("setAttribute",&BaseObjectDescription::setAttribute)
            ;


    class_ <tristate>("tristate",init<int>())
            ;

    class_ <DisplayFlags, boost::noncopyable>("DisplayFlags", no_init)
            //.def("setShowAll",&DisplayFlags::setShowAll)
            //.def("GetShowAll",&DisplayFlags::getShowAll)
            .add_property("showAll",
                          &DisplayFlags::getShowAll,
                          make_function(&DisplayFlags::setShowAll,return_value_policy<reference_existing_object>()))
            .add_property("showVisual",
                          &DisplayFlags::getShowVisual,
                          make_function(&DisplayFlags::setShowVisual,return_value_policy<reference_existing_object>()))
            .add_property("showVisualModels",
                          &DisplayFlags::getShowVisualModels,
                          make_function(&DisplayFlags::setShowVisualModels,return_value_policy<reference_existing_object>()))
            .add_property("showBehavior",
                          &DisplayFlags::getShowBehavior,
                          make_function(&DisplayFlags::setShowBehavior,return_value_policy<reference_existing_object>()))
            .add_property("showBehaviorModels",
                          &DisplayFlags::getShowBehaviorModels,
                          make_function(&DisplayFlags::setShowBehaviorModels,return_value_policy<reference_existing_object>()))
            .add_property("showForceFields",
                          &DisplayFlags::getShowForceFields,
                          make_function(&DisplayFlags::setShowForceFields,return_value_policy<reference_existing_object>()))
            .add_property("showInteractionForceFields",
                          &DisplayFlags::getShowInteractionForceFields,
                          make_function(&DisplayFlags::setShowInteractionForceFields,return_value_policy<reference_existing_object>()))
            .add_property("showCollision",
                          &DisplayFlags::getShowCollision,
                          make_function(&DisplayFlags::setShowCollision,return_value_policy<reference_existing_object>()))
            .add_property("showCollisionModels",
                          &DisplayFlags::getShowCollisionModels,
                          make_function(&DisplayFlags::setShowCollisionModels,return_value_policy<reference_existing_object>()))
            .add_property("showBoundingCollisionModels",
                          &DisplayFlags::getShowBoundingCollisionModels,
                          make_function(&DisplayFlags::setShowBoundingCollisionModels,return_value_policy<reference_existing_object>()))
            .add_property("showMapping",
                          &DisplayFlags::getShowMapping,
                          make_function(&DisplayFlags::setShowMapping,return_value_policy<reference_existing_object>()))
            .add_property("showMappings",
                          &DisplayFlags::getShowMappings,
                          make_function(&DisplayFlags::setShowMappings,return_value_policy<reference_existing_object>()))
            .add_property("showMechanicalMappings",
                          &DisplayFlags::getShowMechanicalMappings,
                          make_function(&DisplayFlags::setShowMechanicalMappings,return_value_policy<reference_existing_object>()))
            .add_property("showOptions",
                          &DisplayFlags::getShowOptions,
                          make_function(&DisplayFlags::setShowOptions,return_value_policy<reference_existing_object>()))
            .add_property("showWireFrame",
                          &DisplayFlags::getShowWireFrame,
                          make_function(&DisplayFlags::setShowWireFrame,return_value_policy<reference_existing_object>()))
            .add_property("showNormals",
                          &DisplayFlags::getShowNormals,
                          make_function(&DisplayFlags::setShowNormals,return_value_policy<reference_existing_object>()))
            ;

    // DisplayFlags
    // note: pour les overloads de beginEdit, voir "BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(george_overloads, wack_em, 1, 3)" plus haut
    class_ <TData<DisplayFlags>, bases<BaseData>, boost::noncopyable>("TDataDisplayFlags", no_init)
            ;
    class_ <Data<DisplayFlags>, bases<TData<DisplayFlags> >, boost::noncopyable>("DataDisplayFlags", no_init)
            .def("beginEdit",&Data<DisplayFlags>::beginEdit,return_value_policy<reference_existing_object>())
            .def("endEdit",&Data<DisplayFlags>::endEdit)
            ;

    // et enfin, mon tout nouveau composant Python qui envoie du steack de poney:
    class_ <BaseController, BaseController::SPtr, bases<BaseObject>, boost::noncopyable>("BaseController", no_init)
            ;
    class_ <Controller, Controller::SPtr, bases<BaseController>, boost::noncopyable>("Controller", no_init)
            ;

    class_ <PythonScriptController, PythonScriptController::SPtr, bases<Controller>, boost::noncopyable>("PythonController", no_init)

            ;
}
*/

