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
#include <image/config.h>
#include <sofa/helper/system/config.h>

#ifdef SOFA_HAVE_SOFAPYTHON
    #include <SofaPython/PythonFactory.h>
    #include "python/Binding_ImageData.h"
#endif


namespace sofa
{

namespace component
{

//Here are just several convenient functions to help user to know what contains the plugin

extern "C" {
    SOFA_IMAGE_API void initExternalModule();
    SOFA_IMAGE_API const char* getModuleName();
    SOFA_IMAGE_API const char* getModuleVersion();
    SOFA_IMAGE_API const char* getModuleLicense();
    SOFA_IMAGE_API const char* getModuleDescription();
    SOFA_IMAGE_API const char* getModuleComponentList();
}

void initExternalModule()
{
    static bool first = true;
    if (first)
    {
        first = false;

#ifdef SOFA_HAVE_SOFAPYTHON
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
    }
}

const char* getModuleName()
{
    return "Image Plugin";
}

const char* getModuleVersion()
{
    return "0.1";
}

const char* getModuleLicense()
{
    return "LGPL";
}


const char* getModuleDescription()
{
    return "Image support in SOFA";
}

const char* getModuleComponentList()
{
    return "ImageContainer,ImageExporter,ImageViewer,ImageFilter,ImageToMeshEngine";
}

} // namespace image

} // namespace sofa

////////// BEGIN CLASS LIST //////////
SOFA_LINK_CLASS(DepthMapToMeshEngine)
SOFA_LINK_CLASS(ImageAccumulator)
SOFA_LINK_CLASS(ImageContainer)
SOFA_LINK_CLASS(ImageDataDisplay)
SOFA_LINK_CLASS(ImageExporter)
SOFA_LINK_CLASS(ImageFilter)
SOFA_LINK_CLASS(ImageOperation)
SOFA_LINK_CLASS(ImageSampler)
SOFA_LINK_CLASS(ImageTransform)
SOFA_LINK_CLASS(ImageTransformEngine)
SOFA_LINK_CLASS(ImageValuesFromPositions)
SOFA_LINK_CLASS(ImageToRigidMassEngine)
#ifndef SOFA_NO_OPENGL
SOFA_LINK_CLASS(ImageViewer)
#endif /* SOFA_NO_OPENGL */
SOFA_LINK_CLASS(MarchingCubesEngine)
SOFA_LINK_CLASS(VoronoiToMeshEngine)
SOFA_LINK_CLASS(MergeImages)
SOFA_LINK_CLASS(MeshToImageEngine)
SOFA_LINK_CLASS(TransferFunction)
#ifdef SOFA_HAVE_LIBFREENECT
SOFA_LINK_CLASS(Kinect)
#endif
