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

#include <algorithm>
#include <string.h>

#include <sofa/core/ExecParams.h>
#include <sofa/simulation/common/ExportGnuplotVisitor.h>


using namespace sofa::simulation;
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
        if (!defaultPath->recordPath.getValue().empty())
        {
            setRecordPath(defaultPath->recordPath.getValue());
        }

        if (!defaultPath->gnuplotPath.getValue().empty())
            setGnuplotPath(defaultPath->gnuplotPath.getValue());
    }


    //Background
    sofa::component::configurationsetting::BackgroundSetting *background;
    groot->get(background, sofa::core::objectmodel::BaseContext::SearchRoot);
    if (background)
    {
        if (background->image.getValue().empty())
            setBackgroundColor(background->color.getValue());
        else
            setBackgroundImage(background->image.getFullPath());
    }

    //Stats
    sofa::component::configurationsetting::StatsSetting *stats;
    groot->get(stats, sofa::core::objectmodel::BaseContext::SearchRoot);
    if (stats)
    {
        setDumpState(stats->dumpState.getValue());
        setLogTime(stats->logTime.getValue());
        setExportState(stats->exportState.getValue());
#ifdef SOFA_DUMP_VISITOR_INFO
        setTraceVisitors(stats->traceVisitors.getValue());
#endif
    }

    //Viewer Dimension
    sofa::component::configurationsetting::ViewerSetting *viewerConf;
    groot->get(viewerConf, sofa::core::objectmodel::BaseContext::SearchRoot);
    if (viewerConf) setViewerConfiguration(viewerConf);

    //TODO: Video Recorder Configuration

    //Mouse Manager using ConfigurationSetting component...
    sofa::helper::vector< sofa::component::configurationsetting::MouseButtonSetting*> mouseConfiguration;
    groot->get<sofa::component::configurationsetting::MouseButtonSetting>(&mouseConfiguration, sofa::core::objectmodel::BaseContext::SearchRoot);

    for (unsigned int i=0; i<mouseConfiguration.size(); ++i)  setMouseButtonConfiguration(mouseConfiguration[i]);

}

void SofaGUI::exportGnuplot(sofa::simulation::Node* node, std::string gnuplot_directory )
{

    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    InitGnuplotVisitor v(params , gnuplot_directory);
    node->execute( v );
    ExportGnuplotVisitor expg ( params /* PARAMS FIRST */, node->getTime());
    node->execute ( expg );

}


} // namespace gui

} // namespace sofa
