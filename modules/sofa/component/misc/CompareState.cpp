/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef SOFA_COMPONENT_MISC_COMPARESTATE_INL
#define SOFA_COMPONENT_MISC_COMPARESTATE_INL

#include <sofa/component/misc/CompareState.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/UpdateMappingVisitor.h>

#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>

#include <sstream>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/common/xml/XML.h>

namespace sofa
{

namespace component
{

namespace misc
{

namespace
{
/*anonymous namespace for utility functions*/





/*
look for potential CompareStateFile formatted likewise
%0_%1_%2_mstate.txt.gz
with
- %0 the current scene name
- %1 the current comparestate counter value
- %2 the name of the mstate which will undergo comparizons.
*/
std::string lookForValidCompareStateFile( const std::string& sceneName,
        const std::string& mstateName,
        const int counterCompareState,
        const std::string& extension)
{
    using namespace sofa::helper::system;

    std::ostringstream ofilename;
    ofilename << sceneName << "_" << counterCompareState << "_" << mstateName  << "_mstate" << extension ;

    std::string result;
    std::string testFilename = ofilename.str();
    std::ostringstream errorlog;
    if( DataRepository.findFile(testFilename,"",&errorlog ) )
    {
        result = ofilename.str();
        return result;
    }

    // from here we look for a closest match in terms of mstateName.

    std::string parentDir = SetDirectory::GetParentDir(testFilename.c_str());
    std::string fileName  = SetDirectory::GetFileName(testFilename.c_str());

    const int& numDefault = sofa::simulation::xml::numDefault;

    for( int i = 0; i<numDefault; ++i)
    {
        std::ostringstream oss;
        oss << sceneName << "_" << counterCompareState << "_default" << i << "_mstate" << extension ;
        std::string testFile = oss.str();
        if(DataRepository.findFile(testFile,"",&errorlog))
        {
            result = testFile;
            break;
        }
    }

    return result;
}

}



int CompareStateClass = core::RegisterObject("Compare State vectors from a reference frame to the associated Mechanical State")
        .add< CompareState >();

CompareState::CompareState(): ReadState()
{
    totalError_X=0.0;
    totalError_V=0.0;
    dofError_X=0.0;
    dofError_V=0.0;
}


//-------------------------------- handleEvent-------------------------------------------
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

//-------------------------------- processCompareState------------------------------------
void CompareState::processCompareState()
{
    double time = getContext()->getTime() + f_shift.getValue();
    time += getContext()->getDt() * 0.001;
    //lastTime = time+0.00001;
    std::vector<std::string> validLines;
    if (!this->readNext(time, validLines)) return;
    for (std::vector<std::string>::iterator it=validLines.begin(); it!=validLines.end(); ++it)
    {
        std::istringstream str(*it);
        std::string cmd;
        str >> cmd;

        double currentError=0;
        if (cmd.compare("X=") == 0)
        {
            //<TO REMOVE>
            //currentError = mmodel->compareX(str);
            currentError = mmodel->compareVec(core::VecId::position(), str);

            totalError_X +=currentError;
            dofError_X +=currentError/(double)this->mmodel->getSize();
        }
        else if (cmd.compare("V=") == 0)
        {
            //<TO REMOVE>
            //currentError = mmodel->compareV(str);
            currentError = mmodel->compareVec(core::VecId::velocity(), str);
            totalError_V +=currentError;
            dofError_V += currentError/(double)this->mmodel->getSize();
        }
    }

    sout << "totalError_X = " << totalError_X << ", totalError_V = " << totalError_V << sendl;
}






CompareStateCreator::CompareStateCreator(const core::ExecParams* params)
    : Visitor(params)
    , sceneName("")
#ifdef SOFA_HAVE_ZLIB
    , extension(".txt.gz")
#else
    , extension(".txt")
#endif
    , createInMapping(false)
    , init(true)
    , counterCompareState(0)
{
}

CompareStateCreator::CompareStateCreator(const std::string &n, const core::ExecParams* params, bool i, int c)
    : Visitor(params)
    , sceneName(n)
#ifdef SOFA_HAVE_ZLIB
    , extension(".txt.gz")
#else
    , extension(".txt")
#endif
    , createInMapping(false)
    , init(i)
    , counterCompareState(c)
{
}

//Create a Compare State component each time a mechanical state is found
simulation::Visitor::Result CompareStateCreator::processNodeTopDown( simulation::Node* gnode)
{
    using namespace sofa::defaulttype;
    sofa::core::behavior::BaseMechanicalState * mstate = gnode->mechanicalState;
    if (!mstate)   return simulation::Visitor::RESULT_CONTINUE;
    core::behavior::OdeSolver *isSimulated;
    mstate->getContext()->get(isSimulated);
    if (!isSimulated) return simulation::Visitor::RESULT_CONTINUE;

    //We have a mechanical state
    addCompareState(mstate, gnode);
    return simulation::Visitor::RESULT_CONTINUE;
}



void CompareStateCreator::addCompareState(sofa::core::behavior::BaseMechanicalState *ms, simulation::Node* gnode)
{
    sofa::core::objectmodel::BaseContext* context = gnode->getContext();
    sofa::core::BaseMapping *mapping; context->get(mapping);
    if (createInMapping || mapping== NULL)
    {
        sofa::component::misc::CompareState *rs; context->get(rs, core::objectmodel::BaseContext::Local);
        if (  rs == NULL )
        {
            rs = new sofa::component::misc::CompareState(); gnode->addObject(rs);
        }



        std::string validFilename = lookForValidCompareStateFile(sceneName, ms->getName(), counterCompareState, extension);

        rs->f_filename.setValue(validFilename);  rs->f_listening.setValue(false); //Deactivated only called by extern functions
        if (init) rs->init();

        ++counterCompareState;
    }
}



//Create a Compare State component each time a mechanical state is found
simulation::Visitor::Result CompareStateResult::processNodeTopDown( simulation::Node* gnode)
{
    sofa::component::misc::CompareState *cv;
    gnode->get(cv);
    if (!cv)   return simulation::Visitor::RESULT_CONTINUE;
    //We have a mechanical state
    error += cv->getTotalError();
    errorByDof += cv->getErrorByDof();
    numCompareState++;
    return simulation::Visitor::RESULT_CONTINUE;
}



} // namespace misc

} // namespace component

} // namespace sofa

#endif
