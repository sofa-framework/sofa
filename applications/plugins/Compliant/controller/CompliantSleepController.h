#ifndef SOFA_COMPONENT_CONTROLLER_COMPLIANTSLEEPCONTROLLER_H
#define SOFA_COMPONENT_CONTROLLER_COMPLIANTSLEEPCONTROLLER_H

#include <Compliant/config.h>
#include <SofaUserInteraction/SleepController.h>

namespace sofa
{

namespace component
{

namespace controller
{

/**
 * @brief As the compliance states are templated, we use a templated class to test each possibilty
 */
class BaseComplianceTester
{
protected:
    virtual ~BaseComplianceTester() {}
public:
	virtual bool canConvert(core::objectmodel::BaseObject* o) = 0;
};

template <class DataTypes>
class ComplianceTester : public BaseComplianceTester
{
public:
	virtual bool canConvert(core::objectmodel::BaseObject* o);
};

/**
 * @brief 	A derived sleep controller that recognizes compliant constraints mapping to wakeup objects.
 */
class SOFA_Compliant_API CompliantSleepController : public SleepController
{
public:
    SOFA_CLASS(CompliantSleepController, SleepController);

protected:
    CompliantSleepController();
    virtual ~CompliantSleepController();

	virtual void collectWakeupPairs(std::vector<BaseContexts>& wakeupPairs);

	bool isCompliance(core::objectmodel::BaseObject* o) const;

	template <class DataTypes> void addCompliance()
		{ m_complianceTesters.push_back(ComplianceTesterPtr(new ComplianceTester<DataTypes>())); }

	typedef std::shared_ptr<BaseComplianceTester> ComplianceTesterPtr;
	typedef std::vector<ComplianceTesterPtr> ComplianceTesters;
	ComplianceTesters m_complianceTesters; // All supported templates

	friend class GetConstrainedContextPairs;
};

class SOFA_Compliant_API GetConstrainedContextPairs : public simulation::Visitor
{
public:
	GetConstrainedContextPairs(const core::ExecParams* params, CompliantSleepController* sleepController, std::vector<CompliantSleepController::BaseContexts>& wakeupPairs);

	virtual void processNodeBottomUp(simulation::Node* node);

protected:
	void processObject(simulation::Node* node, core::objectmodel::BaseObject* o);
	void processMapping(simulation::Node* node, core::objectmodel::BaseObject* o);

	CompliantSleepController* m_sleepController;
	std::vector<CompliantSleepController::BaseContexts>& m_wakeupPairs;
	bool m_processNode;
	core::BaseMapping* m_mapping;
};

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONTROLLER_CONTROLLER_H
