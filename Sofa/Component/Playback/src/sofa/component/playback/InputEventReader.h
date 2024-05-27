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
#include <sofa/component/playback/config.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/core/objectmodel/DataFileName.h>

#ifdef __linux__
#include <linux/input.h>
#include <poll.h>
#endif

namespace sofa::component::playback
{

#ifndef __linux__
struct input_event {};
#endif

/**
 * @brief InputEventReader Class
 *
 * Reads mouse Linux events from file /dev/input/eventX and propagate them as SOFA MouseEvents.
 */
class InputEventReader : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(InputEventReader,core::objectmodel::BaseObject);
protected:
    /**
     * @brief Default Constructor.
     */
    InputEventReader();

    /**
     * @brief Default Destructor.
     */
    ~InputEventReader() override;
public:
    /**
     * @brief SceneGraph callback initialization method.
     */
    void init() override;

    /**
     * @brief handle an event.
     *
     * At every simulation step transforms the mouse Linux events in SOFA mouse events and propagates them
     */
    void handleEvent(core::objectmodel::Event *event) override;

private:
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_PLAYBACK()
    sofa::core::objectmodel::DataFileName filename;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_PLAYBACK()
    Data<bool> inverseSense;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_PLAYBACK()
    Data<bool> p_printEvent;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_PLAYBACK()
    Data<char> p_key1;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_PLAYBACK()
    Data<char> p_key2;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_PLAYBACK()
    Data<bool> p_writeEvents;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_PLAYBACK()
    sofa::core::objectmodel::DataFileName p_outputFilename;

    sofa::core::objectmodel::DataFileName d_filename; ///< file in which the events are read.
    Data<bool> d_inverseSense; ///< inverse the sense of the mouvement
    Data<bool> d_printEvent; ///< Print event informations
    Data<char> d_key1; ///< Key event generated when the left pedal is pressed
    Data<char> d_key2; ///< Key event generated when the right pedal is pressed
    Data<bool> d_writeEvents; ///< If true, write incoming events ; if false, read events from that file (if an output filename is provided)
    sofa::core::objectmodel::DataFileName d_outputFilename;
    std::ifstream* inFile;
    std::ofstream* outFile;

//	Data<double> timeout;
    int fd; ///< desciptor to open and read the file.

    int deplX, deplY; ///< mouse relative deplacements.

    enum PedalState { LEFT_PEDAL, RIGHT_PEDAL, NO_PEDAL };
    int pedalValue;
    PedalState currentPedalState, oldPedalState;


    /**
     * @brief getInputEvents gets the mouse relative deplacements.
     *
     * This method reads from file /dev/input/eventX and gets the mouse relative deplacements.
     */
    void getInputEvents();
    void manageEvent(const input_event &ev);
};

} // namespace sofa::component::playback
