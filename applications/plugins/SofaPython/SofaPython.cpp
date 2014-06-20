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
#include "SofaPython.h"
#include <sofa/core/Plugin.h>

#include "SceneLoaderPY.h"
#include "PythonScriptController.h"

using sofa::component::controller::PythonScriptController;
using sofa::simulation::SceneLoaderFactory;


class SofaPythonPlugin: public sofa::core::Plugin {
public:
    SofaPythonPlugin(): Plugin("SofaPython") {
        setDescription("Imbeds Python scripts in Sofa.");
        setVersion(SOFAPYTHON_VERSION);
        setLicense("LGPL");
        setAuthors("Bruno Carrez");

        addComponent<PythonScriptController>("A Sofa controller scripted in python.");
    }

    virtual bool init() {
        // register the scene loader
        SceneLoaderFactory::getInstance()->addEntry(new sofa::simulation::SceneLoaderPY());
        std::cout << "SofaPython: registered scene loader" << std::endl;
        return true;
    }

    virtual bool canBeUnloaded() {
        return false;
    }

    virtual bool exit() {
        // TODO: remove the scene loader registered at init()
        return false;
    }
};

SOFA_PLUGIN(SofaPythonPlugin);
