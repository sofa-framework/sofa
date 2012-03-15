#include "PythonBindings.h"


#include <boost/python.hpp>
using namespace boost::python;

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

// the function to be wrapped to python...
void HelloCWorld(char *s)
{
    printf("Hello, C World! \"%s\"\n",s);
}





class testClass
{
protected:
    std::string mName;
public:
    const std::string getName() {return mName;}
    void setName(const std::string n) {mName=n;}
};

//sofa::simulation::xml::BaseElement::NodeFactory* getNodeFactory() {return sofa::simulation::xml::BaseElement::NodeFactory::getInstance();}

// ce qui suit est la SEULE manière de binder avec boost.python des fonctions surchargées
// on passe par un pointeur en spécifiant les paramètres
void (Base::*setName1)(const std::string&) = &Base::setName;

// factory!
BaseObject::SPtr createObject(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
//BaseObject* createObject(objectmodel::BaseContext* context, objectmodel::BaseObjectDescription* arg)
{
    printf("<PYTHON>createObject '%s' of type '%s' in node '%s'\n",
            arg->getName().c_str(),
            arg->getAttribute("type",""),
            context->getName().c_str());
    BaseObject::SPtr obj = ObjectFactory::getInstance()->createObject(context,arg);//.get();
    if (obj==0) printf("<PYTHON>createObject ERROR\n");
    return obj;
}


BOOST_PYTHON_MODULE( Sofa )
{
    def ("HelloBoost" , HelloCWorld ) ; // une simple fonction HelloWorld

//    def ("createObject", createObject);
    def ("createObject", createObject);//, return_value_policy<reference_existing_object>());

    class_ <Base, Base::SPtr, boost::noncopyable>("Base", no_init)
    //      .def("setName",setName1)
    //      .def("getName",&Base::getName,return_value_policy<copy_const_reference>())
    .add_property("name",
            make_function(&Base::getName,return_value_policy<copy_const_reference>()),
            setName1)
    .def("findData",&Base::findData,return_value_policy<reference_existing_object>())
    ;

    class_ <BaseNode, BaseNode::SPtr, bases<Base>, boost::noncopyable>("BaseNode", no_init)
    ;

    class_ <BaseObject, BaseObject::SPtr, bases<Base>, boost::noncopyable>("BaseObject", no_init)
    ;

    class_ <BaseContext, BaseContext::SPtr, bases<Base>, boost::noncopyable>("BaseContext", no_init)
    .def("isActive",&BaseContext::isActive)
    .def("setActive",&BaseContext::setActive)
    .def("getTime",&BaseContext::getTime)
    .def("getDt",&BaseContext::getDt)
    .def("getAnimate",&BaseContext::getAnimate)
    .def("getGravity",&BaseContext::getGravity,return_value_policy<copy_const_reference>())
    .def("setGravity",&BaseContext::setGravity)
    .def("getRootContext",&BaseContext::getRootContext,return_value_policy<reference_existing_object>())
    .def("setAnimate",&BaseContext::setAnimate)
    ;

    class_ <Context, Context::SPtr, bases<BaseContext>, boost::noncopyable>("Context", no_init)
    ;


    // STRANGE BEHAVIOUR: en mettant bases<BaseNode> avant bases<Context>,
    // crash dans le script python : "AttributeError: 'GNode' object has no attribute 'getRootContext'"
    // ... il semblerait que l'héritage multiple ne soit pas supporté par python.... arg
    // on contourne en shuntant la classe "baseNode". Eh ouais.
    class_ <Node, Node::SPtr, bases<BaseNode,Context>, boost::noncopyable>("Node", no_init)
    ;

    // POUAH ! dégueulasse de mettre des double directement
    class_ <Vec3d>("Vec3",init<double,double,double>())
    ;

    // inutile de déclarer coord3 et deriv3, ça rentre en conflit avec Vec3...
//    class_ <Deriv3>("Deriv3",init<double,double,double>())
//            ;

    class_ <GNode, GNode::SPtr, bases<Node>, boost::noncopyable>("GNode", no_init)
    .def("createChild",&GNode::createChild,return_value_policy<reference_existing_object>())
    .def("addChild",&GNode::addChild)
    .def("removeChild",&GNode::removeChild)
    .def("moveChild",&GNode::moveChild)
    .def("addObject",&GNode::addObject)
    .def("removeObject",&GNode::removeObject)
    .def("detachFromGraph",&GNode::detachFromGraph)
    ;

    class_ <OdeSolver, OdeSolver::SPtr, bases<BaseObject>, boost::noncopyable>("OdeSolver", no_init)
    ;

    class_ <EulerSolver, EulerSolver::SPtr, bases<OdeSolver>, boost::noncopyable>("EulerSolver", no_init)
    ;

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


    // BaseData, TData<T>, Data<T>...
    class_ <BaseData, boost::noncopyable>("BaseData", no_init)
    .add_property("name",
            make_function(&BaseData::getName,return_value_policy<copy_const_reference>()),
            &BaseData::setName)
    .add_property("owner",
            make_function(&BaseData::getOwner,return_value_policy<reference_existing_object>()),
            &BaseData::setOwner)
    .add_property("value",
            &BaseData::getValueString,
            &BaseData::read)

    .def("getValueTypeString",&BaseData::getValueTypeString)
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

/*
BOOST_PYTHON_MODULE( SofaBaseNode )
{

    class_ <sofa::core::objectmodel::BaseNode, boost::noncopyable>("BaseNode", no_init) // private constructor...
            .def("getRoot",&sofa::core::objectmodel::BaseNode::getRoot)
      //      .def("setName",&Base::setName)
            ;

}
*/


void registerSofaPythonModule()
{
    PyImport_AppendInittab( (char*)"Sofa", &initSofa );
    //PyImport_AppendInittab( "SofaSimulationTree", &initSofaSimulationTree );
}
