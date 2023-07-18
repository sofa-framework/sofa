/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/playback/ReadState.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalProjectPositionAndVelocityVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalProjectPositionAndVelocityVisitor;

#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndVelocityVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalPropagateOnlyPositionAndVelocityVisitor;

#include <cstring>
#include <sstream>

namespace sofa::component::playback
{

using type::Vec3;

ReadState::ReadState()
    : d_filename( initData(&d_filename, "filename", "output file name"))
    , d_interval( initData(&d_interval, 0.0, "interval", "time duration between inputs"))
    , d_shift( initData(&d_shift, 0.0, "shift", "shift between times in the file and times when they will be read"))
    , d_loop( initData(&d_loop, false, "loop", "set to 'true' to re-read the file when reaching the end"))
    , d_scalePos( initData(&d_scalePos, 1.0, "scalePos", "scale the input mechanical object"))
    , d_rotation( initData(&d_rotation, Vec3(0.,0.,0.), "rotation", "rotate the input mechanical object"))
    , d_translation( initData(&d_translation, Vec3(0.,0.,0.), "translation", "translate the input mechanical object"))
    , mmodel(nullptr)
    , infile(nullptr)
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
    , gzfile(nullptr)
#endif
    , nextTime(0)
    , lastTime(0)
    , loopTime(0)
{
    this->f_listening.setValue(true);
    d_scalePos.setGroup("Transformation");
    d_rotation.setGroup("Transformation");
    d_translation.setGroup("Transformation");
}

ReadState::~ReadState()
{
    if (infile)
        delete infile;
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
    if (gzfile)
        gzclose(gzfile);
#endif
}

void ReadState::init()
{
    reset();
}

void ReadState::bwdInit()
{
    processReadState();
}


void ReadState::reset()
{
    mmodel = this->getContext()->getMechanicalState();
    if (infile)
    {
        delete infile;
        infile = nullptr;
    }
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
    if (gzfile)
    {
        gzclose(gzfile);
        gzfile = nullptr;
    }
#endif

    const std::string& filename = d_filename.getFullPath();
    if (filename.empty())
    {
        msg_error() << "ERROR: empty filename";
    }
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
    else if (filename.size() >= 3 && filename.substr(filename.size()-3)==".gz")
    {
        gzfile = gzopen(filename.c_str(),"rb");
        if( !gzfile )
        {
            msg_error() << "Error opening compressed file "<<filename;
        }
    }
#endif
    else
    {
        infile = new std::ifstream(filename.c_str());
        if( !infile->is_open() )
        {
            msg_error() << "Error opening file "<<filename;
            delete infile;
            infile = nullptr;
        }
    }
    nextTime = 0;
    lastTime = 0;
    loopTime = 0;
}

void ReadState::handleEvent(sofa::core::objectmodel::Event* event)
{
    if (simulation::AnimateBeginEvent::checkEventType(event))
    {
        processReadState();
    }
    if (simulation::AnimateEndEvent::checkEventType(event))
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
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
        && !gzfile
#endif
       )
        return false;
    lastTime = time;
    validLines.clear();
    std::string line, cmd;
    while (nextTime <= time)
    {
#if SOFA_COMPONENT_PLAYBACK_HAVE_ZLIB
        if (gzfile)
        {
            if (gzeof(gzfile))
            {
                if (!d_loop.getValue())
                    break;
                gzrewind(gzfile);
                loopTime = nextTime;
            }
            //getline(gzfile, line);
            line.clear();
            char buf[4097];
            buf[0] = '\0';
            while (gzgets(gzfile,buf,sizeof(buf))!=nullptr && buf[0])
            {
                const size_t l = strlen(buf);
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
                    if (!d_loop.getValue())
                        break;
                    infile->clear();
                    infile->seekg(0);
                    loopTime = nextTime;
                }
                getline(*infile, line);
            }

        std::istringstream str(line);
        str >> cmd;
        if (cmd == "T=")
        {
            str >> nextTime;
            nextTime += loopTime;

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
    double time = getContext()->getTime() + d_shift.getValue();
    std::vector<std::string> validLines;
    if (!readNext(time, validLines)) return;
    bool updated = false;

    const double scale = d_scalePos.getValue();
    const Vec3& rotation = d_rotation.getValue();
    const Vec3& translation = d_translation.getValue();

    for (std::vector<std::string>::iterator it=validLines.begin(); it!=validLines.end(); ++it)
    {
        std::istringstream str(*it);
        std::string cmd;
        str >> cmd;
        if (cmd == "X=")
        {
            mmodel->readVec(core::VecId::position(), str);
            mmodel->applyScale(scale,scale,scale);
            mmodel->applyRotation(rotation[0],rotation[1],rotation[2]);
            mmodel->applyTranslation(translation[0],translation[1],translation[2]);

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
        MechanicalProjectPositionAndVelocityVisitor action0(core::mechanicalparams::defaultInstance());
        this->getContext()->executeVisitor(&action0);
        MechanicalPropagateOnlyPositionAndVelocityVisitor action1(core::mechanicalparams::defaultInstance());
        this->getContext()->executeVisitor(&action1);
        sofa::simulation::UpdateMappingVisitor action2(core::mechanicalparams::defaultInstance());
        this->getContext()->executeVisitor(&action2);
    }
}

} // namespace sofa::component::playback
