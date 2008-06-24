#ifndef SOFA_COMPONENT_MISC_COMPARESTATE_H
#define SOFA_COMPONENT_MISC_COMPARESTATE_H

#include <sofa/component/misc/ReadState.h>


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

} // namespace misc

} // namespace component

} // namespace sofa

#endif
