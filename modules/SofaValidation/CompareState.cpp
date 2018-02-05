/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

#ifndef SOFA_COMPONENT_MISC_COMPARESTATE_INL
#define SOFA_COMPONENT_MISC_COMPARESTATE_INL

#include <SofaValidation/CompareState.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/core/ObjectFactory.h>
#include <SofaSimulationCommon/xml/XML.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>

#include <sstream>
#include <algorithm>
using namespace sofa::defaulttype;

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
        const std::string& extension,
        const std::string defaultName ="default")
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

    const int& numDefault = sofa::simulation::xml::numDefault;

    for( int i = 0; i<numDefault; ++i)
    {
        std::ostringstream oss;
        oss << sceneName << "_" << counterCompareState << "_" << defaultName << i << "_mstate" << extension ;
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
    if (/* simulation::AnimateBeginEvent* ev = */simulation::AnimateBeginEvent::checkEventType(event))
    {
        processCompareState();
    }
    if (/* simulation::AnimateEndEvent* ev = */simulation::AnimateEndEvent::checkEventType(event))
    {

    }
}

//-------------------------------- processCompareState------------------------------------
void CompareState::processCompareState()
{
    SReal time = getContext()->getTime() + f_shift.getValue();
    time += getContext()->getDt() * 0.001;
    //lastTime = time+0.00001;
    std::vector<std::string> validLines;
    if (!nextValidLines.empty() && last_time == getContext()->getTime())
        validLines.swap(nextValidLines);
    else
    {
        last_time = getContext()->getTime();
        if (!this->readNext(time, validLines)) return;
    }
    for (std::vector<std::string>::iterator it=validLines.begin(); it!=validLines.end(); ++it)
    {
        std::istringstream str(*it);
        std::string cmd;
        str >> cmd;

        double currentError=0;
        if (cmd.compare("X=") == 0)
        {
            last_X = *it;
            currentError = mmodel->compareVec(core::VecId::position(), str);


            totalError_X +=currentError;

            double dsize = (double)this->mmodel->getSize();
            if (dsize != 0.0)
                dofError_X +=currentError/dsize;
        }
        else if (cmd.compare("V=") == 0)
        {
            last_V = *it;
            currentError = mmodel->compareVec(core::VecId::velocity(), str);
            totalError_V +=currentError;

            double dsize = (double)this->mmodel->getSize();
            if (dsize != 0.0)
                dofError_V += currentError/dsize;
        }
    }

    sout << "totalError_X = " << totalError_X << ", totalError_V = " << totalError_V << sendl;
}

//-------------------------------- processCompareState------------------------------------
void CompareState::draw(const core::visual::VisualParams* vparams)
{
    SReal time = getContext()->getTime() + f_shift.getValue();
    time += getContext()->getDt() * 0.001;
    //lastTime = time+0.00001;
    if (nextValidLines.empty() && last_time != getContext()->getTime())
    {
        last_time = getContext()->getTime();
        if (!this->readNext(time, nextValidLines))
            nextValidLines.clear();
        else
        {
            for (std::vector<std::string>::iterator it=nextValidLines.begin(); it!=nextValidLines.end(); ++it)
            {
                std::istringstream str(*it);
                std::string cmd;
                str >> cmd;
                if (cmd.compare("X=") == 0)
                {
                    last_X = *it;
                }
                else if (cmd.compare("V=") == 0)
                {
                    last_V = *it;
                }
            }
        }
    }

    if (mmodel && !last_X.empty())
    {
        core::VecCoordId refX(core::VecCoordId::V_FIRST_DYNAMIC_INDEX);
        mmodel->vAvail(vparams, refX);
        mmodel->vAlloc(vparams, refX);
        std::istringstream str(last_X);
        std::string cmd;
        str >> cmd;
        mmodel->readVec(refX, str);

        const core::objectmodel::BaseData* dataX = mmodel->baseRead(core::VecCoordId::position());
        const core::objectmodel::BaseData* dataRefX = mmodel->baseRead(refX);
        if (dataX && dataRefX)
        {
            const sofa::defaulttype::AbstractTypeInfo* infoX = dataX->getValueTypeInfo();
            const sofa::defaulttype::AbstractTypeInfo* infoRefX = dataRefX->getValueTypeInfo();
            const void* valueX = dataX->getValueVoidPtr();
            const void* valueRefX = dataRefX->getValueVoidPtr();
            if (valueX && infoX && infoX->ValidInfo() && valueRefX && infoRefX && infoRefX->ValidInfo())
            {
                int ncX = infoX->size();
                int ncRefX = infoRefX->size();
                int sizeX = infoX->size(valueX);
                int sizeRefX = infoRefX->size(valueRefX);
                if (ncX > 1 && ncRefX > 1)
                {
                    int nc = std::min(3,std::min(ncX,ncRefX));
                    int nbp = std::min(sizeX/ncX, sizeRefX/ncRefX);

                    std::vector< Vector3 > points;
                    points.resize(nbp*2);
                    for(int p=0; p<nbp; ++p)
                    {
                        Vector3& pX = points[2*p+0];
                        Vector3& pRefX = points[2*p+1];
                        for (int c=0; c<nc; ++c)
                            pX[c] = infoX->getScalarValue(valueX, p*ncX+c);
                        for (int c=0; c<nc; ++c)
                            pRefX[c] = infoRefX->getScalarValue(valueRefX, p*ncRefX+c);
                    }
                    vparams->drawTool()->drawLines(points, 1, Vec<4,float>(1.0f,0.0f,0.5f,1.0f));
                }
            }
        }

        mmodel->vFree(vparams, refX);
    }
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
        sofa::component::misc::CompareState::SPtr rs; context->get(rs, core::objectmodel::BaseContext::Local);
        if (  rs == NULL )
        {
            rs = sofa::core::objectmodel::New<sofa::component::misc::CompareState>(); gnode->addObject(rs);
        }

        // compatibility:
        std::string validFilename = lookForValidCompareStateFile(sceneName, ms->getName(), counterCompareState, extension);
        if(validFilename.empty())
        {
            // look for a file which closest match the shortName of this mechanicalState.
            validFilename = lookForValidCompareStateFile(sceneName, ms->getName(), counterCompareState, extension,
                    ms->getClass()->shortName);
        }
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
