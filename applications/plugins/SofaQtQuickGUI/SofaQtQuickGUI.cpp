/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "SofaQtQuickGUI.h"
#include "Tools.h"
#include "Camera.h"
#include "PickingInteractor.h"
#include "PythonInteractor.h"
#include "Scene.h"
#include "SceneListModel.h"
#include "DataListModel.h"
#include "Viewer.h"

using namespace sofa::qtquick;

const int versionMajor = 1;
const int versionMinor = 0;

SofaQtQuickGUI::SofaQtQuickGUI(QObject *parent) : QQmlExtensionPlugin(parent)
{
    init();
}

void SofaQtQuickGUI::init()
{
    Q_INIT_RESOURCE(sofaQML);
    Q_INIT_RESOURCE(sofaQtQuickResources);

    registerTypes("SofaQtQuickGUI");
}

void SofaQtQuickGUI::registerTypes(const char* /*uri*/)
{
	//Q_ASSERT(QLatin1String(uri) == QLatin1String("SofaQtQuickGUI"));

    qmlRegisterType<Tools>                      ("Tools"			, versionMajor, versionMinor, "Tools");
    qmlRegisterType<Camera>                     ("Camera"			, versionMajor, versionMinor, "Camera");
    qmlRegisterType<PickingInteractor>          ("PickingInteractor", versionMajor, versionMinor, "PickingInteractor");
    qmlRegisterType<PythonInteractor>           ("PythonInteractor"	, versionMajor, versionMinor, "PythonInteractor");
    qmlRegisterType<Scene>                      ("Scene"			, versionMajor, versionMinor, "Scene");
    qmlRegisterUncreatableType<SceneComponent> 	("SceneComponent"	, versionMajor, versionMinor, "SceneComponent", "SceneComponent is not instantiable");
    qmlRegisterUncreatableType<SceneData>       ("SceneData"        , versionMajor, versionMinor, "SceneData", "SceneData is not instantiable");
    qmlRegisterType<SceneListModel>             ("SceneListModel"   , versionMajor, versionMinor, "SceneListModel");
    qmlRegisterType<DataListModel>              ("DataListModel"    , versionMajor, versionMinor, "DataListModel");
    qmlRegisterType<Viewer>                     ("Viewer"			, versionMajor, versionMinor, "Viewer");
}
