#ifndef SOFA_COMPONENT_MISC_COMPARESTATE_H
#define SOFA_COMPONENT_MISC_COMPARESTATE_H

#include <sofa/component/misc/ReadState.h>
#include <sofa/simulation/common/Visitor.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

/** Compare State vectors from file at each timestep
*/
class CompareState: public ReadState
{
public:
    CompareState();

    void handleEvent(sofa::core::objectmodel::Event* event);
    void processCompareState();

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::componentmodel::behavior::BaseMechanicalState*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

    double getError() {return totalError_X + totalError_V;}
protected :
    double totalError_X;
    double totalError_V;
};

/// Create CompareState component in the graph each time needed
class CompareStateCreator: public Visitor
{
public:
    CompareStateCreator() : sceneName(""), counterCompareState(0), createInMapping(false) {}
    CompareStateCreator(std::string &n, bool i=true, int c=0 ) { sceneName=n; init=i; counterCompareState=c; }
    virtual Result processNodeTopDown( simulation::Node*  );

    void setSceneName(std::string &n) { sceneName = n; }
    void setCounter(int c) { counterCompareState = c; }
    void setCreateInMapping(bool b) { createInMapping=b; }
protected:
    void addCompareState(sofa::core::componentmodel::behavior::BaseMechanicalState *ms, simulation::Node* gnode);
    bool init;
    std::string sceneName;
    int counterCompareState; //avoid to have two same files if two mechanical objects has the same name
    bool createInMapping;
};

class CompareStateResult: public Visitor
{
public:
    CompareStateResult() { error=0; }
    virtual Result processNodeTopDown( simulation::Node*  );

    double getError() { return error; }
protected:
    double error;
};

} // namespace misc

} // namespace component

} // namespace sofa

#endif
