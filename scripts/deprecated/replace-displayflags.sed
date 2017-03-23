#!/bin/sed -f 



s/(\([[:blank:]]*const core::visual::VisualParams\*\)\([[:blank:]]*\))/(\1 vparams)/g


s/this->getContext()->getShowVisualModels()/vparams->displayFlags().getShowVisualModels()/g
s/getContext()->getShowVisualModels()/vparams->displayFlags().getShowVisualModels()/g

s/this->getContext()->getShowBehaviorModels()/vparams->displayFlags().getShowBehaviorModels()/g
s/getContext()->getShowBehaviorModels()/vparams->displayFlags().getShowBehaviorModels()/g

s/this->getContext()->getShowForceFields()/vparams->displayFlags().getShowForceFields()/g
s/getContext()->getShowForceFields()/vparams->displayFlags().getShowForceFields()/g

s/this->getContext()->getShowInteractionForceFields()/vparams->displayFlags().getShowInteractionForceFields()/g
s/getContext()->getShowInteractionForceFields()/vparams->displayFlags().getShowInteractionForceFields()/g

s/this->getContext()->getShowCollisionModels()/vparams->displayFlags().getShowCollisionModels()/g
s/getContext()->getShowCollisionModels()/vparams->displayFlags().getShowCollisionModels()/g

s/this->getContext()->getShowBoundingCollisionModels()/vparams->displayFlags().getShowBoundingCollisionModels()/g
s/getContext()->getShowBoundingCollisionModels()/vparams->displayFlags().getShowBoundingCollisionModels()/g

s/this->getContext()->getShowMappings()/vparams->displayFlags().getShowMappings()/g
s/getContext()->getShowMappings()/vparams->displayFlags().getShowMappings()/g

s/this->getContext()->getShowMechanicalMappings()/vparams->displayFlags().getShowMechanicalMappings()/g
s/getContext()->getShowMechanicalMappings()/vparams->displayFlags().getShowMechanicalMappings()/g

s/this->getContext()->getShowWireFrame()/vparams->displayFlags().getShowWireFrame()/g
s/getContext()->getShowWireFrame()/vparams->displayFlags().getShowWireFrame()/g

s/this->getContext()->getShowNormals()/vparams->displayFlags().getShowNormals()/g
s/getContext()->getShowNormals()/vparams->displayFlags().getShowNormals()/g

s/this->getContext()->getShowProcessorColor()/vparams->displayFlags().getShowProcessorColor()/g
s/getContext()->getShowProcessorColor()/vparams->displayFlags().getShowProcessorColor()/g







