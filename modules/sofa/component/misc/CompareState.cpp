#ifndef SOFA_COMPONENT_MISC_COMPARESTATE_INL
#define SOFA_COMPONENT_MISC_COMPARESTATE_INL

#include <sofa/component/misc/CompareState.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/UpdateMappingVisitor.h>

#include <sstream>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace misc
{

int CompareStateClass = core::RegisterObject("Compare State vectors from a reference frame to the associated Mechanical State")
        .add< CompareState >();
CompareState::CompareState(): ReadState()
{
    totalError_X=0.0;
    totalError_V=0.0;
}



void CompareState::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (/* simulation::AnimateBeginEvent* ev = */ dynamic_cast<simulation::AnimateBeginEvent*>(event))
    {
        processCompareState();
    }
    if (/* simulation::AnimateEndEvent* ev = */ dynamic_cast<simulation::AnimateEndEvent*>(event))
    {
    }
}
void CompareState::processCompareState()
{
    if (infile && mmodel)
    {
        double time = getContext()->getTime() + f_shift.getValue();
        lastTime = time;
        std::vector<std::string> validLines;
        std::string line, cmd;
        while (nextTime <= time && !infile->eof())
        {
            getline(*infile, line);
            //std::cout << "line= "<<line<<std::endl;
            std::istringstream str(line);
            str >> cmd;
            if (cmd == "T=")
            {
                str >> nextTime;
                if (nextTime <= time) validLines.clear();
            }

            if (nextTime <= time) validLines.push_back(line);
        }

        for (std::vector<std::string>::iterator it=validLines.begin(); it!=validLines.end(); ++it)
        {
            std::istringstream str(*it);
            cmd.clear();
            str >> cmd;
            if (cmd == "X=")
            {
                totalError_X += mmodel->compareX(str);
            }
            else if (cmd == "V=")
            {
                totalError_V += mmodel->compareV(str);
            }
        }
    }
}

} // namespace misc

} // namespace component

} // namespace sofa

#endif
