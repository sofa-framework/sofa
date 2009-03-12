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
#ifndef SOFA_COMPONENT_MISC_READSTATE_INL
#define SOFA_COMPONENT_MISC_READSTATE_INL

#include <sofa/component/misc/ReadState.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/UpdateMappingVisitor.h>

#include <sstream>

namespace sofa
{

namespace component
{

namespace misc
{

ReadState::ReadState()
    : f_filename( initData(&f_filename, "filename", "output file name"))
    , f_interval( initData(&f_interval, 0.0, "interval", "time duration between inputs"))
    , f_shift( initData(&f_shift, 0.0, "shift", "shift between times in the file and times when they will be read"))
    , f_loop( initData(&f_loop, false, "loop", "set to 'true' to re-read the file when reaching the end"))
    , mmodel(NULL)
    , infile(NULL)
    , nextTime(0)
{
    this->f_listening.setValue(true);
}

ReadState::~ReadState()
{
    if (infile)
        delete infile;
}

void ReadState::init()
{
//     mmodel = dynamic_cast<core::componentmodel::behavior::MechanicalState<DataTypes>*>(this->getContext()->getMechanicalState());
    reset();
}

void ReadState::reset()
{
    mmodel = dynamic_cast< sofa::core::componentmodel::behavior::BaseMechanicalState* >(this->getContext()->getMechanicalState());
    if (infile)
        delete infile;
    const std::string& filename = f_filename.getValue();
    if (!filename.empty())
    {
        infile = new std::ifstream(filename.c_str());
        if( !infile->is_open() )
        {
            serr << "Error opening file "<<filename<<sendl;
            delete infile;
            infile = NULL;
        }
    }
    nextTime = 0;
}

void ReadState::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (/* simulation::AnimateBeginEvent* ev = */ dynamic_cast<simulation::AnimateBeginEvent*>(event))
    {
        processReadState();
    }
    if (/* simulation::AnimateEndEvent* ev = */ dynamic_cast<simulation::AnimateEndEvent*>(event))
    {

    }
}



void ReadState::setTime(double time)
{
    if (time < nextTime) {reset(); nextTime=0.0;}
}

void ReadState::processReadState(double time)
{
    if (time == lastTime) return;
    setTime(time);
    processReadState();
}

void ReadState::processReadState()
{
    static double totalTime = 0.0;
    bool updated = false;

    if (infile && mmodel)
    {
        double time = getContext()->getTime() + f_shift.getValue();
        lastTime = time;
        std::vector<std::string> validLines;
        std::string line, cmd;
        while (nextTime <= time && !infile->eof())
        {
            getline(*infile, line);
            //sout << "line= "<<line<<sendl;
            std::istringstream str(line);
            str >> cmd;
            if (cmd == "T=")
            {
                str >> nextTime;
                nextTime += totalTime;
                if (nextTime <= time)
                    validLines.clear();
            }

            if (nextTime <= time)
                validLines.push_back(line);
        }

        for (std::vector<std::string>::iterator it=validLines.begin(); it!=validLines.end(); ++it)
        {
            std::istringstream str(*it);
            cmd.clear();
            str >> cmd;
            if (cmd == "X=")
            {
                mmodel->readX(str);
                updated = true;
            }
            else if (cmd == "V=")
            {
                mmodel->readV(str);
                updated = true;
            }
        }
    }

    if (f_loop.getValue() && infile->eof())
    {
        infile->clear();
        infile->seekg(0);
        totalTime = nextTime;
    }

    if (updated)
    {
        //sout<<"update from file"<<sendl;
        sofa::simulation::MechanicalPropagatePositionAndVelocityVisitor action1;
        this->getContext()->executeVisitor(&action1);
        sofa::simulation::UpdateMappingVisitor action2;
        this->getContext()->executeVisitor(&action2);
    }
}

} // namespace misc

} // namespace component

} // namespace sofa

#endif
