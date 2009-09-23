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
#include <sofa/component/misc/InputEventReader.h>
#include <sofa/core/ObjectFactory.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifdef __linux__
#include <linux/input.h>
#include <poll.h>
#endif

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(InputEventReader)

// Register in the Factory
int InputEventReaderClass = core::RegisterObject("Read events from file")
        .add< InputEventReader >();

InputEventReader::InputEventReader()
    : filename( initData(&filename, std::string("/dev/input/mouse2"), "filename", "input events file name"))
    , inverseSense(initData(&inverseSense, false, "inverseSense", "inverse the sense of the mouvement"))
//, timeout( initData(&timeout, 0, "timeout", "time out to get an event from file" ))
    , fd(-1)
    , deplX(0), deplY(0)
{
}
void InputEventReader::init()
{
#ifdef __linux__
    if((fd = open(filename.getFullPath().c_str(), O_RDONLY)) < 0)
        sout << "ERROR: impossible to open the file: " << filename.getValue() << sendl;
#endif
}

InputEventReader::~InputEventReader()
{
#ifdef __linux__
    if (fd >= 0)
        close(fd);
#endif
}

void InputEventReader::getInputEvents()
{
#ifdef __linux__
    if (fd < 0) return;
    pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN;
    pfd.revents = 0;

    while (poll(&pfd, 1, 0 /*timeout.getValue()*/)>0 && (pfd.revents & POLLIN))
    {
        input_event ev;
        if (read(fd, &ev, sizeof(input_event)) == -1)
            serr << "Error: read function return an error." << sendl;

//		sout << "event type 0x" << std::hex << ev.type << std::dec << " code 0x" << std::hex << ev.code << std::dec << " value " << ev.value << sendl;
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
    }
#endif
}

void InputEventReader::handleEvent(core::objectmodel::Event *event)
{

    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {
        getInputEvents();
        if (deplX || deplY)
        {
            sofa::core::objectmodel::MouseEvent mouseEvent(sofa::core::objectmodel::MouseEvent::Move, deplX, deplY);
            deplX = 0;
            deplY = 0;
            getContext()->propagateEvent(&mouseEvent);
        }
    }
}

} // namespace misc

} // namespace component

} // namespace sofa
