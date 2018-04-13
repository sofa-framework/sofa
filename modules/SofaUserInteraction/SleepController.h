/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_CONTROLLER_SLEEPCONTROLLER_H
#define SOFA_COMPONENT_CONTROLLER_SLEEPCONTROLLER_H
#include "config.h"

#include <SofaUserInteraction/initUserInteraction.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/simulation/Visitor.h>

namespace sofa
{

	
namespace core
{

namespace behavior
{
class BaseMechanicalState;
} // namespace behavior

} // namespace core

namespace component
{

namespace controller
{

/**
 * @brief As the mechanical states are templated, we use a templated class to test each possibilty
 */
class BaseStateTester
{
public:
	virtual bool canConvert(core::behavior::BaseMechanicalState* baseState) = 0;
	virtual bool wantsToSleep(core::behavior::BaseMechanicalState* baseState, SReal speedThreshold, SReal rotationThreshold) = 0;
};

template <class DataTypes>
class StateTester : public BaseStateTester
{
public:
    virtual ~StateTester();
	virtual bool canConvert(core::behavior::BaseMechanicalState* baseState);
	virtual bool wantsToSleep(core::behavior::BaseMechanicalState* baseState, SReal speedThreshold, SReal rotationThreshold);
};

/**
 * @brief 	A component that will survey the graph for nodes/contexts that can sleep.
 *			From those objects their mechanical states will be extracted and their velocities 
 *			are compared to a set threshold and if under the threshold, put to sleep.
 *			A minimum time is set before putting a node to sleep if it was just woken up, so 
 *			objects with no initial velocity aren't put to sleep right away.
 *			If a sleeping object is in contact with another object, it's woken up.
 */
class SOFA_USER_INTERACTION_API SleepController : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(SleepController, core::objectmodel::BaseObject);

	virtual void init() override;
	virtual void reset() override;
	virtual void handleEvent(core::objectmodel::Event*) override;

	Data<double> d_minTimeSinceWakeUp; ///< Do not do anything before objects have been moving for this duration
	Data<SReal> d_speedThreshold; ///< Speed value under which we consider a particule to be immobile
	Data<SReal> d_rotationThreshold; ///< If non null, this is the rotation speed value under which we consider a particule to be immobile

protected:
    SleepController();
    virtual ~SleepController();

	void putNodesToSleep();
	void wakeUpNodes();
	void updateTimeSinceWakeUp();
	void updateSleepStatesRecursive();

	// Return a parent node that can change its sleeping state (if it exists, else return the same node)
	core::objectmodel::BaseContext* getParentContextThatCanSleep(core::objectmodel::BaseContext* context);

	template <class DataTypes> void addType()
	{ m_stateTesters.push_back(StateTesterPtr(new StateTester<DataTypes>)); }

	typedef std::vector<core::behavior::BaseMechanicalState*> StatesThatCanSleep;
	StatesThatCanSleep m_statesThatCanSleep; // The states we will test each timestep

	typedef std::vector<core::objectmodel::BaseContext*> BaseContexts;
	BaseContexts m_contextsThatCanSleep; // The nodes corresponding to m_statesThanCanSleep 

	std::vector<double> m_timeSinceWakeUp; // For each monitored node, the duration since it has awaken
	std::vector<bool> m_initialState; // The initial state of each node we are monitoring (for reset)

	typedef std::shared_ptr<BaseStateTester> StateTesterPtr;
	typedef std::vector<StateTesterPtr> StateTesters;
	StateTesters m_stateTesters; // All supported templates
	StateTesters m_correspondingTesters; // The correct template for each state of the list m_statesThanCanSleep

	virtual void collectWakeupPairs(std::vector<BaseContexts>& wakeupPairs);
	void addWakeupPair(std::vector<BaseContexts>& wakeupPairs, core::objectmodel::BaseContext* context1, bool moving1, core::objectmodel::BaseContext* context2, bool moving2);
};

/**
 * @brief A visitor that gets a list of all base mechanical state that can sleep
 */
class SOFA_USER_INTERACTION_API GetStatesThatCanSleep : public simulation::Visitor
{
public:
	GetStatesThatCanSleep(const core::ExecParams* params, std::vector<core::behavior::BaseMechanicalState*>& states);

	virtual void processNodeBottomUp(simulation::Node* node);

protected:
	std::vector<core::behavior::BaseMechanicalState*>& m_states;
};

/**
 * @brief A visitor that sets the sleep state of all nodes based on their parents being asleep.
 */
class SOFA_USER_INTERACTION_API UpdateAllSleepStates : public simulation::Visitor
{
public:
	UpdateAllSleepStates(const core::ExecParams* params);

	virtual Visitor::Result processNodeTopDown(simulation::Node* node);
};

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONTROLLER_SLEEPCONTROLLER_H
