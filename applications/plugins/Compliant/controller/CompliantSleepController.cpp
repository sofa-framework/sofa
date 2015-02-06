#include "CompliantSleepController.h"

#include <sofa/core/ObjectFactory.h>
#include <compliance/DiagonalCompliance.h>
#include <compliance/UniformCompliance.h>

namespace sofa
{

namespace component
{

namespace controller
{

template <class DataTypes>
bool ComplianceTester<DataTypes>::canConvert(core::objectmodel::BaseObject* o)
{
	return dynamic_cast< DataTypes* >(o) != NULL;
}

CompliantSleepController::CompliantSleepController()
{
#ifndef SOFA_DOUBLE
	addCompliance< forcefield::DiagonalCompliance<defaulttype::Vec6fTypes> >();
	addCompliance< forcefield::UniformCompliance<sofa::defaulttype::Vec1fTypes> >();
#endif
#ifndef SOFA_FLOAT
	addCompliance< forcefield::DiagonalCompliance<defaulttype::Vec6dTypes> >();
	addCompliance< forcefield::UniformCompliance<sofa::defaulttype::Vec1dTypes> >();
#endif
}

CompliantSleepController::~CompliantSleepController()
{
}

void CompliantSleepController::collectWakeupPairs(std::vector<BaseContexts>& wakeupPairs)
{
	SleepController::collectWakeupPairs(wakeupPairs);

	GetConstrainedContextPairs(core::ExecParams::defaultInstance(), this, wakeupPairs).execute(getContext()->getRootContext());
}

bool CompliantSleepController::isCompliance(core::objectmodel::BaseObject* o) const
{
	for (unsigned int i = 0, nbTesters = m_complianceTesters.size(); i < nbTesters; ++i)
	{
		ComplianceTesterPtr tester = m_complianceTesters[i];
		if (tester->canConvert(o))
		{
			return true;
		}
	}
	return false;
}

GetConstrainedContextPairs::GetConstrainedContextPairs(const core::ExecParams* params, CompliantSleepController* sleepController, std::vector<CompliantSleepController::BaseContexts>& wakeupPairs)
	: simulation::Visitor(params)
	, m_sleepController(sleepController)
	, m_wakeupPairs(wakeupPairs)
{
}

void GetConstrainedContextPairs::processNodeBottomUp(simulation::Node* node)
{
	// Simplified and more generic joint detection, based on the fact that when creating a multimapping,
	// the mapping node MUST be added as a child of all mapped nodes.
	// It may detect incorrectly configured relations as valid joints, but it's an acceptable tradeoff as those should not happen anyway,
	// and the only negative impact is that some nodes may be maintained awake that would otherwise be able to sleep.

	simulation::Node::Parents parents = node->getParents();
	if (parents.size() == 2)
	{
		m_sleepController->addWakeupPair(m_wakeupPairs, parents[0]->getContext(), true, parents[1]->getContext(), true);
	}

	/*
	m_processNode = false;
	for_each(this, node, node->object, &GetConstrainedContextPairs::processObject);
	if (!m_processNode)
		return;

	simulation::Node* searchNode = node;
	while (searchNode)
	{
		m_mapping = NULL;
		for_each(this, searchNode, searchNode->object, &GetConstrainedContextPairs::processMapping);
		searchNode = NULL;

		if (m_mapping)
		{
			helper::vector<core::BaseState*> states = m_mapping->getFrom();
			switch (states.size())
			{
				case 0:
					// bad mapping?
					break;
				case 1:
					// this state is just mapped on another one, we should test the parent state
					// (typically, this will be a mapping that converts to the dofs the compliance applies on, like a DistanceMapping)
					searchNode = dynamic_cast<simulation::Node*>(states[0]->getContext());
					break;
				case 2:
				{
					// multi-mapping + compliance creates a dependency between the two input mechanical states, this is what we track.
					m_sleepController->addWakeupPair(m_wakeupPairs, states[0]->getContext(), true, states[1]->getContext(), true);
					break;
				}
				default:
					// FIXME: I don't know the system well enough to decice what to do when encountering a multi-mapping with more than two inputs.
					// We may want to enumerate the context pairs in each individual dof mapping, but how?
					break;
			}
		}
	}
	*/
}

void GetConstrainedContextPairs::processObject(simulation::Node* /*node*/, core::objectmodel::BaseObject* o)
{
	if (m_sleepController->isCompliance(o))
	{
		m_processNode = true;
	}
}

void GetConstrainedContextPairs::processMapping(simulation::Node* /*node*/, core::objectmodel::BaseObject* o)
{
	core::BaseMapping* mapping = dynamic_cast< core::BaseMapping* >(o);
	if (mapping != NULL)
	{
		m_mapping = mapping;
	}
}

int CompliantSleepControllerClass = core::RegisterObject("A controller that puts node into sleep when the objects are not moving, and wake them up again when there are in collision with a moving object (compatible with compliant specific constraints)")
	.add< CompliantSleepController >();

SOFA_DECL_CLASS(CompliantSleepController);

} // namespace controller

} // namepace component

} // namespace sofa