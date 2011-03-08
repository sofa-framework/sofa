declare(newmat,		extlibs/newmat)
declare(tinyxml,	extlibs/tinyxml)
declare(eigen,		extlibs/eigen-3.0-beta2/eigen.pro)

!contains(DEFINES,SOFA_HAVE_FLOWVR){
	declare(miniFlowVR, extlibs/miniFlowVR)
}


declare(sofahelper,				framework/sofa/helper,			newmat)
declare(sofadefaulttype,	framework/sofa/defaulttype,	sofahelper)
declare(sofacore,					framework/sofa/core, 				sofadefaulttype sofahelper)

declare(sofasimulation,		modules/sofa/simulation/common,	sofacore)
declare(sofatree,					modules/sofa/simulation/tree,		sofasimulation sofacore)
declare(sofabgl,					modules/sofa/simulation/bgl,		sofasimulation sofacore)

declare(sofacomponentbase,					modules/sofa/component/libbase.pro,		sofatree sofacore miniFlowVR)
declare(sofacomponentbehaviormodel,	modules/sofa/component/behaviormodel, sofatree sofacore)
declare(sofacomponentcontextobject, modules/sofa/component/contextobject,	sofatree sofacore)
declare(sofacomponentengine,				modules/sofa/component/engine,				sofacomponentcollision sofacomponentbase sofasimulation sofacore)
declare(sofacomponentfem,						modules/sofa/component/fem,						sofacomponentbase sofatree sofasimulation sofacore newmat eigen)
declare(sofacomponentforcefield,		modules/sofa/component/forcefield,		sofacomponentbase sofacore newmat eigen)
declare(sofacomponentloader,				modules/sofa/component/loader,				sofatree sofasimulation sofacore tinyxml)
declare(sofacomponentmapping,				modules/sofa/component/mapping,				sofacomponentforcefield sofacomponentvisualmodel sofacomponentbase sofatree sofasimulation sofacore)
declare(sofacomponentmass,					modules/sofa/component/mass,					sofacomponentbase sofatree sofasimulation sofacore)
declare(sofacomponentodesolver,			modules/sofa/component/odesolver,			sofatree sofasimulation sofacore)
declare(sofacomponentvisualmodel,		modules/sofa/component/visualmodel,		sofacomponentbase sofatree sofasimulation sofacore tinyxml)

declare(sofacomponentconstraintset,	modules/sofa/component/constraintset, sofacomponentforcefield	sofacomponentodesolver sofacomponentlinearsolver sofacomponentmass sofacomponentbase sofacore eigen)
declare(sofacomponentlinearsolver,	modules/sofa/component/linearsolver,	sofacomponentforcefield	sofacomponentodesolver sofacomponentbase sofacore newmat eigen)

declare(sofacomponentinteractionforcefield,		modules/sofa/component/interactionforcefield,		sofacomponentforcefield sofacomponentbase sofacore)
declare(sofacomponentprojectiveconstraintset,	modules/sofa/component/projectiveconstraintset, sofacomponentlinearsolver sofacomponentforcefield	sofacomponentodesolver sofacomponentmass sofacomponentbase sofacore)
declare(sofacomponentmastersolver,	modules/sofa/component/mastersolver,	sofacomponentlinearsolver sofacomponentconstraintset sofacomponentprojectiveconstraintset sofacomponentbase sofasimulation sofacore)

declare(sofacomponentcontroller, modules/sofa/component/controller, \
	sofacomponentmastersolver sofacomponentinteractionforcefield \
	sofacomponentforcefield sofacomponentconstraintset \
	sofacomponentprojectiveconstraintset sofacomponentbase \
	sofatree sofasimulation sofacore)

declare(sofacomponentcollision, modules/sofa/component/collision, \
	sofacomponentvisualmodel sofacomponentconstraintset sofacomponentprojectiveconstraintset \
	sofacomponentmapping sofacomponentinteractionforcefield sofacomponentforcefield \
	sofacomponentodesolver sofacomponentlinearsolver sofacomponentbase \
	sofabgl sofatree sofasimulation sofacore miniFlowVR eigen newmat)

declare(sofacomponentmisc, modules/sofa/component/misc,	\
	sofacomponentmastersolver sofacomponentfem sofacomponentinteractionforcefield \
	sofacomponentcontextobject sofacomponentbehaviormodel sofacomponentlinearsolver \
	sofacomponentodesolver sofacomponentcontroller sofacomponentvisualmodel \
	sofacomponentmass sofacomponentforcefield sofacomponentmapping \
	sofacomponentprojectiveconstraintset sofacomponentconstraintset sofacomponentcollision \
	sofacomponentbase sofasimulation sofatree sofacore newmat)

declare(sofacomponentconfigurationsetting, modules/sofa/component/configurationsetting, \
	sofatree sofasimulation sofacomponentbase \
	sofacomponentmastersolver sofacomponentinteractionforcefield sofacomponentcontextobject \
	sofacomponentbehaviormodel sofacomponentlinearsolver sofacomponentodesolver \
	sofacomponentcontroller sofacomponentvisualmodel sofacomponentmass sofacomponentforcefield \
	sofacomponentmapping sofacomponentprojectiveconstraintset sofacomponentconstraintset \
	sofacomponentcollision sofacomponentmisc sofacore)

declare(sofacomponent, modules/sofa/component/libcomponent.pro, \
	sofacomponentloader sofacomponentbase sofacomponentmastersolver \
	sofacomponentfem sofacomponentinteractionforcefield sofacomponentcontextobject \
	sofacomponentbehaviormodel sofacomponentlinearsolver sofacomponentodesolver \
	sofacomponentbase sofacomponentcontroller sofacomponentvisualmodel \
	sofacomponentmass sofacomponentforcefield sofacomponentmapping \
	sofacomponentprojectiveconstraintset sofacomponentconstraintset \
	sofacomponentcollision sofacomponentmisc sofacomponentconfigurationsetting \
	sofacomponentengine sofatree sofasimulation sofacore)
