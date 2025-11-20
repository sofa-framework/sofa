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
#include <image/config.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/logging/Messaging.h>

#if IMAGE_HAVE_SOFAPYTHON
    #include <SofaPython/PythonFactory.h>
    #include "python/Binding_ImageData.h"
#endif

namespace sofa::defaulttype
{
    extern void registerDataExchange(sofa::core::ObjectFactory* factory);
}

namespace sofa::component
{

namespace misc
{
    extern void registerImageExporter(sofa::core::ObjectFactory* factory);
}

namespace engine
{
    extern void registerVoronoiToMeshEngine(sofa::core::ObjectFactory* factory);
    extern void registerTransferFunction(sofa::core::ObjectFactory* factory);
    extern void registerMeshToImageEngine(sofa::core::ObjectFactory* factory);
    extern void registerMergeImages(sofa::core::ObjectFactory* factory);
    extern void registerMarchingCubesEngine(sofa::core::ObjectFactory* factory);
    extern void registerImageValuesFromPositions(sofa::core::ObjectFactory* factory);
    extern void registerImageTransformEngine(sofa::core::ObjectFactory* factory);
    extern void registerImageTransform(sofa::core::ObjectFactory* factory);
    extern void registerImageToRigidMassEngine(sofa::core::ObjectFactory* factory);
    extern void registerImageSampler(sofa::core::ObjectFactory* factory);
    extern void registerImageOperation(sofa::core::ObjectFactory* factory);
    extern void registerImageFilter(sofa::core::ObjectFactory* factory);
    extern void registerImageDataDisplay(sofa::core::ObjectFactory* factory);
    extern void registerImageCoordValuesFromPositions(sofa::core::ObjectFactory* factory);
    extern void registerGenerateImage(sofa::core::ObjectFactory* factory);
    extern void registerDepthMapToMeshEngine(sofa::core::ObjectFactory* factory);
    extern void registerCollisionToCarvingEngine(sofa::core::ObjectFactory* factory);

#ifdef PLUGIN_IMAGE_COMPILE_GUI
    extern void registerContourImageToolBox(sofa::core::ObjectFactory* factory);
    extern void registerAverageCatchAllVector(sofa::core::ObjectFactory* factory);
    extern void registerCatchAllVector(sofa::core::ObjectFactory* factory);
    extern void registerDepthImageToolBox(sofa::core::ObjectFactory* factory);
    extern void registerMergedCatchAllVector(sofa::core::ObjectFactory* factory);
    extern void registerLabelBoxImageToolBox(sofa::core::ObjectFactory* factory);
    extern void registerLabelGridImageToolBox(sofa::core::ObjectFactory* factory);
    extern void registerLabelPointImageToolBox(sofa::core::ObjectFactory* factory);
    extern void registerLabelPointsBySectionImageToolBox(sofa::core::ObjectFactory* factory);
    extern void registerDistanceZoneImageToolBox(sofa::core::ObjectFactory* factory);
    extern void registerZoneGeneratorImageToolBox(sofa::core::ObjectFactory* factory);
#endif
}

//Here are just several convenient functions to help user to know what contains the plugin

extern "C" {
    SOFA_IMAGE_API void initExternalModule();
    SOFA_IMAGE_API const char* getModuleName();
    SOFA_IMAGE_API const char* getModuleVersion();
    SOFA_IMAGE_API const char* getModuleLicense();
    SOFA_IMAGE_API const char* getModuleDescription();
    SOFA_IMAGE_API void registerObjects(sofa::core::ObjectFactory* factory);
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        // make sure that this plugin is registered into the PluginManager
        sofa::helper::system::PluginManager::getInstance().registerPlugin(image::MODULE_NAME);

        first = false;

#if IMAGE_HAVE_SOFAPYTHON
        if( PythonFactory::s_sofaPythonModule ) // add the module only if the Sofa module exists (SofaPython is loaded)
        {
            simulation::PythonEnvironment::gil lock(__func__);
            
            // adding new bindings for Data<Image<T>>
            SP_ADD_CLASS_IN_FACTORY(ImageCData,sofa::core::objectmodel::Data<sofa::defaulttype::ImageC>)
            SP_ADD_CLASS_IN_FACTORY(ImageUCData,sofa::core::objectmodel::Data<sofa::defaulttype::ImageUC>)
            SP_ADD_CLASS_IN_FACTORY(ImageIData,sofa::core::objectmodel::Data<sofa::defaulttype::ImageI>)
            SP_ADD_CLASS_IN_FACTORY(ImageUIData,sofa::core::objectmodel::Data<sofa::defaulttype::ImageUI>)
            SP_ADD_CLASS_IN_FACTORY(ImageSData,sofa::core::objectmodel::Data<sofa::defaulttype::ImageS>)
            SP_ADD_CLASS_IN_FACTORY(ImageUSData,sofa::core::objectmodel::Data<sofa::defaulttype::ImageUS>)
            SP_ADD_CLASS_IN_FACTORY(ImageLData,sofa::core::objectmodel::Data<sofa::defaulttype::ImageL>)
            SP_ADD_CLASS_IN_FACTORY(ImageULData,sofa::core::objectmodel::Data<sofa::defaulttype::ImageUL>)
            SP_ADD_CLASS_IN_FACTORY(ImageFData,sofa::core::objectmodel::Data<sofa::defaulttype::ImageF>)
            SP_ADD_CLASS_IN_FACTORY(ImageDData,sofa::core::objectmodel::Data<sofa::defaulttype::ImageD>)
            SP_ADD_CLASS_IN_FACTORY(ImageBData,sofa::core::objectmodel::Data<sofa::defaulttype::ImageB>)
        }
#endif

        std::string pluginPath = sofa::helper::system::PluginManager::getInstance().findPlugin("image_gui");
        if (!pluginPath.empty())
        {
            sofa::helper::system::PluginManager::getInstance().loadPluginByPath(pluginPath);
        }
        else
        {
            msg_warning("initImage") << "the sub-plugin image_gui was not successfully loaded";
        }
    }
}

const char* getModuleName()
{
    return image::MODULE_NAME;
}

const char* getModuleVersion()
{
    return image::MODULE_VERSION;
}

const char* getModuleLicense()
{
    return "LGPL";
}


const char* getModuleDescription()
{
    return "Image support in SOFA";
}

void registerObjects(sofa::core::ObjectFactory* factory)
{
    sofa::defaulttype::registerDataExchange(factory);
    sofa::component::misc::registerImageExporter(factory);
    sofa::component::engine::registerVoronoiToMeshEngine(factory);
    sofa::component::engine::registerTransferFunction(factory);
    sofa::component::engine::registerMeshToImageEngine(factory);
    sofa::component::engine::registerMergeImages(factory);
    sofa::component::engine::registerMarchingCubesEngine(factory);
    sofa::component::engine::registerImageValuesFromPositions(factory);
    sofa::component::engine::registerImageTransformEngine(factory);
    sofa::component::engine::registerImageTransform(factory);
    sofa::component::engine::registerImageToRigidMassEngine(factory);
    sofa::component::engine::registerImageSampler(factory);
    sofa::component::engine::registerImageOperation(factory);
    sofa::component::engine::registerImageFilter(factory);
    sofa::component::engine::registerImageDataDisplay(factory);
    sofa::component::engine::registerImageCoordValuesFromPositions(factory);
    sofa::component::engine::registerGenerateImage(factory);
    sofa::component::engine::registerDepthMapToMeshEngine(factory);
    sofa::component::engine::registerCollisionToCarvingEngine(factory);
}

} // namespace sofa::component

