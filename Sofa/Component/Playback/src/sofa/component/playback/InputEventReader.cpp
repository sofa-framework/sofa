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
#include <sofa/component/playback/InputEventReader.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <fstream>

#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>

namespace sofa::component::playback
{

// Register in the Factory
int InputEventReaderClass = core::RegisterObject("Read events from file")
        .add< InputEventReader >();

InputEventReader::InputEventReader()
    : filename( initData(&filename, std::string("/dev/input/mouse2"), "filename", "input events file name"))
    , inverseSense(initData(&inverseSense, false, "inverseSense", "inverse the sense of the mouvement"))
    , p_printEvent(initData(&p_printEvent, false, "printEvent", "Print event informations"))
//, timeout( initData(&timeout, 0, "timeout", "time out to get an event from file" ))
    , p_key1(initData(&p_key1, '0', "key1","Key event generated when the left pedal is pressed"))
    , p_key2(initData(&p_key2, '1', "key2","Key event generated when the right pedal is pressed"))
    , p_writeEvents(initData(&p_writeEvents, false , "writeEvents","If true, write incoming events ; if false, read events from that file (if an output filename is provided)"))
    , p_outputFilename(initData(&p_outputFilename, "outputFilename","Other filename where events will be stored (or read)"))
    , inFile(nullptr), outFile(nullptr)
    , fd(-1)
    , deplX(0), deplY(0)
    , pedalValue(-1)
    , currentPedalState(NO_PEDAL)
    , oldPedalState(NO_PEDAL)
{
}
void InputEventReader::doBaseObjectInit()
{
#ifdef __linux__
    if((fd = open(filename.getFullPath().c_str(), O_RDONLY)) < 0)
        msg_error() << "Impossible to open the file: " << filename.getValue();
#endif

    if(p_outputFilename.isSet())
    {
        if (p_writeEvents.getValue())
        {
            outFile = new std::ofstream();
            outFile->open(p_outputFilename.getFullPath().c_str());
            if( !outFile->is_open() )
            {
                msg_error() << "File " <<p_outputFilename.getFullPath() << " not writable";
                delete outFile;
                outFile = nullptr;
            }
        }
        else
        {
            inFile = new std::ifstream(p_outputFilename.getFullPath().c_str(), std::ifstream::in | std::ifstream::binary);
            if( !inFile->is_open() )
            {
                msg_error() << "File " <<p_outputFilename.getFullPath() << " not readable";
                delete inFile;
                inFile = nullptr;
            }
        }
    }
}

InputEventReader::~InputEventReader()
{
#ifdef __linux__
    if (fd >= 0)
        close(fd);
#endif
}

void InputEventReader::manageEvent(const input_event &ev)
{
#ifndef __linux__
    (void)ev;
#endif
#ifdef __linux__
    if (p_printEvent.getValue())
        msg_error() << "event type 0x" << std::hex << ev.type << std::dec << " code 0x" << std::hex << ev.code << std::dec << " value " << ev.value;

    if (ev.type == EV_REL)
    {
        switch (ev.code)
        {
        case REL_X:
            if (inverseSense.getValue())
            {
                deplX -= ev.value; break;
            }
            else
            {
                deplX += ev.value; break;
            }
        case REL_Y:
            if (inverseSense.getValue())
            {
                deplY -= ev.value; break;
            }
            else
            {
                deplY += ev.value; break;
            }
        }
    }

    pedalValue = -1;

    if (ev.type == EV_ABS)
    {
        switch (ev.code)
        {
        case ABS_Y:
            pedalValue = ev.value;
            oldPedalState = currentPedalState;
            //TODO: less specific
            if (pedalValue > 156)
            {
                //Left
                currentPedalState = LEFT_PEDAL;
            }
            else if (pedalValue < 100)
            {
                //Right
                currentPedalState = RIGHT_PEDAL;
            }
            else
            {
                //Nothing
                currentPedalState = NO_PEDAL;
            }
            break;
        }
    }
#endif
}

void InputEventReader::getInputEvents()
{
#ifdef __linux__
    //read device
    if (!inFile)
    {
        if (fd < 0) return;
        pollfd pfd;
        pfd.fd = fd;
        pfd.events = POLLIN;
        pfd.revents = 0;
        struct input_event ev, temp;

        while (poll(&pfd, 1, 0 /*timeout.getValue()*/)>0 && (pfd.revents & POLLIN))
        {
            if (read(fd, &temp, sizeof(struct input_event)) == -1)
                msg_error() << "Read function return an error.";

            memcpy(&ev, &temp, sizeof(struct input_event));

            manageEvent(ev);

            if (p_writeEvents.getValue())
            {
                if(outFile->good())
                {
                    *outFile << ev.type << " " << ev.code << " " << ev.value << " ";
                }
            }
        }
        if (p_writeEvents.getValue())
        {
            if(outFile->good())
            {
                *outFile << std::endl;
            }
        }
    }
    else //read file
    {
        if (!inFile->eof())
        {
            std::string line;
            std::getline(*inFile, line);
            std::istringstream ln(line);

            while(!ln.eof())
            {
                struct input_event ev;
                ln >> ev.type >> ev.code >> ev.value;
                manageEvent(ev);
            }
        }
    }

#endif
}

void InputEventReader::handleEvent(core::objectmodel::Event *event)
{
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
    {
        getInputEvents();
        //Mouse event
        if (deplX || deplY)
        {
            sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::Move, deplX, deplY);
            deplX = 0;
            deplY = 0;
            getContext()->propagateEvent(core::execparams::defaultInstance(), &mouseEvent);
        }

        //Pedals Value
        //get root
        if (currentPedalState != NO_PEDAL)
        {
            if (oldPedalState == NO_PEDAL)
            {
                if (currentPedalState == LEFT_PEDAL)
                {
                    sofa::core::objectmodel::KeypressedEvent ev(p_key1.getValue());
                    this->getContext()->propagateEvent(core::execparams::defaultInstance(), &ev);
                }
                else
                {
                    sofa::core::objectmodel::KeypressedEvent ev(p_key2.getValue());
                    this->getContext()->propagateEvent(core::execparams::defaultInstance(), &ev);
                }
            }
        }
        else
        {
            if (oldPedalState != NO_PEDAL)
            {
                if (oldPedalState == LEFT_PEDAL)
                {
                    sofa::core::objectmodel::KeyreleasedEvent ev(p_key1.getValue());
                    this->getContext()->propagateEvent(core::execparams::defaultInstance(), &ev);
                }
                else
                {
                    sofa::core::objectmodel::KeyreleasedEvent ev(p_key2.getValue());
                    this->getContext()->propagateEvent(core::execparams::defaultInstance(), &ev);
                }
            }
        }
    }
}

} //namespace sofa::component::playback
