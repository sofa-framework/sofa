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
#ifndef SOFA_COMPONENT_MISC_READSTATE_INL
#define SOFA_COMPONENT_MISC_READSTATE_INL

#include <SofaGeneralLoader/ReadState.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>

#include <string.h>
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
    , f_scalePos( initData(&f_scalePos, 1.0, "scalePos", "scale the input mechanical object"))
    , mmodel(NULL)
    , infile(NULL)
#ifdef SOFA_HAVE_ZLIB
    , gzfile(NULL)
#endif
    , nextTime(0)
    , lastTime(0)
    , loopTime(0)
{
    this->f_listening.setValue(true);
}

ReadState::~ReadState()
{
    if (infile)
        delete infile;
#ifdef SOFA_HAVE_ZLIB
    if (gzfile)
        gzclose(gzfile);
#endif
}

void ReadState::init()
{
    reset();
}

void ReadState::reset()
{
    mmodel = this->getContext()->getMechanicalState();
    if (infile)
    {
        delete infile;
        infile = NULL;
    }
#ifdef SOFA_HAVE_ZLIB
    if (gzfile)
    {
        gzclose(gzfile);
        gzfile = NULL;
    }
#endif

    const std::string& filename = f_filename.getFullPath();
    if (filename.empty())
    {
        serr << "ERROR: empty filename"<<sendl;
    }
#ifdef SOFA_HAVE_ZLIB
    else if (filename.size() >= 3 && filename.substr(filename.size()-3)==".gz")
    {
        gzfile = gzopen(filename.c_str(),"rb");
        if( !gzfile )
        {
            serr << "Error opening compressed file "<<filename<<sendl;
        }
    }
#endif
    else
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
    lastTime = 0;
    loopTime = 0;
}

void ReadState::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (/* simulation::AnimateBeginEvent* ev = */simulation::AnimateBeginEvent::checkEventType(event))
    {
        processReadState();
    }
    if (/* simulation::AnimateEndEvent* ev = */simulation::AnimateEndEvent::checkEventType(event))
    {

    }
}



void ReadState::setTime(double time)
{
    if (time+getContext()->getDt()*0.5 < lastTime) {reset();}
}

void ReadState::processReadState(double time)
{
    if (time == lastTime) return;
    setTime(time);
    processReadState();
}

bool ReadState::readNext(double time, std::vector<std::string>& validLines)
{
    if (!mmodel) return false;
    if (!infile
#ifdef SOFA_HAVE_ZLIB
        && !gzfile
#endif
       )
        return false;
    lastTime = time;
    validLines.clear();
    std::string line, cmd;
    while (nextTime <= time)
    {
#ifdef SOFA_HAVE_ZLIB
        if (gzfile)
        {
            if (gzeof(gzfile))
            {
                if (!f_loop.getValue())
                    break;
                gzrewind(gzfile);
                loopTime = nextTime;
            }
            //getline(gzfile, line);
            line.clear();
            char buf[4097];
            buf[0] = '\0';
            while (gzgets(gzfile,buf,sizeof(buf))!=NULL && buf[0])
            {
                size_t l = strlen(buf);
                if (buf[l-1] == '\n')
                {
                    buf[l-1] = '\0';
                    line += buf;
                    break;
                }
                else
                {
                    line += buf;
                    buf[0] = '\0';
                }
            }
        }
        else
#endif
            if (infile)
            {
                if (infile->eof())
                {
                    if (!f_loop.getValue())
                        break;
                    infile->clear();
                    infile->seekg(0);
                    loopTime = nextTime;
                }
                getline(*infile, line);
            }
        //sout << "line= "<<line<<sendl;
        std::istringstream str(line);
        str >> cmd;
        if (cmd == "T=")
        {
            str >> nextTime;
            nextTime += loopTime;
            //sout << "next time: " << nextTime << sendl;
            if (nextTime <= time)
                validLines.clear();
        }
        if (nextTime <= time)
            validLines.push_back(line);
    }
    return true;
}

void ReadState::processReadState()
{
    double time = getContext()->getTime() + f_shift.getValue();
    std::vector<std::string> validLines;
    if (!readNext(time, validLines)) return;
    bool updated = false;
    for (std::vector<std::string>::iterator it=validLines.begin(); it!=validLines.end(); ++it)
    {
        std::istringstream str(*it);
        std::string cmd;
        str >> cmd;
        if (cmd == "X=")
        {
            mmodel->readVec(core::VecId::position(), str);            
            mmodel->applyScale(f_scalePos.getValue(), f_scalePos.getValue(), f_scalePos.getValue());

            updated = true;
        }
        else if (cmd == "V=")
        {
            mmodel->readVec(core::VecId::velocity(), str);
            updated = true;
        }
    }

    if (updated)
    {
        //sout<<"update from file"<<sendl;
        sofa::simulation::MechanicalProjectPositionAndVelocityVisitor action0(core::MechanicalParams::defaultInstance());
        this->getContext()->executeVisitor(&action0);
        sofa::simulation::MechanicalPropagateOnlyPositionAndVelocityVisitor action1(core::MechanicalParams::defaultInstance());
        this->getContext()->executeVisitor(&action1);
        sofa::simulation::UpdateMappingVisitor action2(core::MechanicalParams::defaultInstance());
        this->getContext()->executeVisitor(&action2);
    }
}

} // namespace misc

} // namespace component

} // namespace sofa

#endif
