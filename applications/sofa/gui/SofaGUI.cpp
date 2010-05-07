/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "SofaGUI.h"
#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/helper/vector.h>

#include <sofa/component/configurationsetting/SofaDefaultPathSetting.h>
#include <sofa/component/configurationsetting/BackgroundSetting.h>
#include <sofa/component/configurationsetting/StatsSetting.h>
#include <sofa/component/configurationsetting/ViewerDimensionSetting.h>

#include <algorithm>
#include <string.h>

namespace sofa
{

namespace gui
{
const char* SofaGUI::programName = NULL;
std::string SofaGUI::guiName = "";

SofaGUI::SofaGUI()
{

}

SofaGUI::~SofaGUI()
{

}

void SofaGUI::configureGUI(sofa::simulation::Node *groot)
{

    sofa::component::configurationsetting::SofaDefaultPathSetting *defaultPath;
    groot->get(defaultPath, sofa::core::objectmodel::BaseContext::SearchRoot);
    if (defaultPath)
    {
        if (!defaultPath->getRecordPath().empty())
            setRecordPath(defaultPath->getRecordPath());

        if (!defaultPath->getGnuplotPath().empty())
            setGnuplotPath(defaultPath->getGnuplotPath());
    }


    //Background
    sofa::component::configurationsetting::BackgroundSetting *background;
    groot->get(background, sofa::core::objectmodel::BaseContext::SearchRoot);
    if (background)
    {
        if (background->getImage().empty())
            setBackgroundColor(background->getColor());
        else
            setBackgroundImage(background->getImage());
    }

    //Stats
    sofa::component::configurationsetting::StatsSetting *stats;
    groot->get(stats, sofa::core::objectmodel::BaseContext::SearchRoot);
    if (stats)
    {
        setDumpState(stats->getDumpState());
        setLogTime(stats->getLogTime());
        setExportState(stats->getExportState());
#ifdef SOFA_DUMP_VISITOR_INFO
        setTraceVisitors(stats->getTraceVisitors());
#endif
    }

    //Viewer Dimension
    sofa::component::configurationsetting::ViewerDimensionSetting *dimension;
    groot->get(dimension, sofa::core::objectmodel::BaseContext::SearchRoot);
    if (dimension)
    {
        const defaulttype::Vec<2,int> &res=dimension->getDimension();
        if (dimension->getFullscreen()) setFullScreen();
        else setDimension(res[0], res[1]);
    }

    //TODO: Video Recorder Configuration

    //Mouse Manager using ConfigurationSetting component...
    sofa::helper::vector< sofa::component::configurationsetting::MouseButtonSetting*> mouseConfiguration;
    groot->get<sofa::component::configurationsetting::MouseButtonSetting>(&mouseConfiguration, sofa::core::objectmodel::BaseContext::SearchRoot);

    for (unsigned int i=0; i<mouseConfiguration.size(); ++i)
    {
        setMouseButtonConfiguration(mouseConfiguration[i]);
    }

}


} // namespace gui

} // namespace sofa
