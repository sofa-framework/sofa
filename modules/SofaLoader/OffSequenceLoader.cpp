/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaLoader/OffSequenceLoader.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/core/ObjectFactory.h>
#include <sstream>


namespace sofa
{

namespace component
{

namespace loader
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(OffSequenceLoader)

int OffSequenceLoaderClass = core::RegisterObject("Read and load an .off file at each timestep")
        .add< OffSequenceLoader >();


OffSequenceLoader::OffSequenceLoader():sofa::component::loader::MeshOffLoader()
    , nbFiles( initData(&nbFiles,(int)1,"nbOfFiles","number of files in the sequence") )
    , stepDuration( initData(&stepDuration,0.04,"stepDuration","how long each file is loaded") )
    , firstIndex(0) , currentIndex(0)
{
    this->f_listening.setValue(true);

    edges.setDisplayed(false);
    triangles.setDisplayed(false);
    quads.setDisplayed(false);
    polygons.setDisplayed(false);
    tetrahedra.setDisplayed(false);
    hexahedra.setDisplayed(false);
    edgesGroups.setDisplayed(false);
    trianglesGroups.setDisplayed(false);
    quadsGroups.setDisplayed(false);
    polygonsGroups.setDisplayed(false);
    tetrahedraGroups.setDisplayed(false);
    hexahedraGroups.setDisplayed(false);
}


void OffSequenceLoader::reset()
{
    currentIndex = firstIndex;
    clear();
    MeshOffLoader::load();
}


void OffSequenceLoader::init()
{
    MeshOffLoader::init();

    //parse the file name to get the index part
    std::string file = this->m_filename.getFullPath();
    m_filenameAndNb = file.substr(0, file.find("."));
    size_t indCar = m_filenameAndNb.size();
    std::string fileNb;

    while ( m_filenameAndNb[--indCar] >= '0' && m_filenameAndNb[indCar] <= '9')
        fileNb.insert(0, 1, m_filenameAndNb[indCar]);

    currentIndex = firstIndex = atoi(fileNb.c_str());
}


void OffSequenceLoader::handleEvent(sofa::core::objectmodel::Event* event)
{
    //load the next file at the beginning of animation step and if the current file duration is over
    if (simulation::AnimateBeginEvent::checkEventType(event))
    {
        if ( (currentIndex-firstIndex)*stepDuration.getValue() <= this->getContext()->getTime())
        {
            currentIndex++;
            if (currentIndex < firstIndex+nbFiles.getValue())
            {
                std::ostringstream os;
                os << currentIndex;
                std::string indexStr = os.str();
                std::string filetmp = m_filenameAndNb.substr(0, m_filenameAndNb.size()-indexStr.size());

                std::string newFile = filetmp + indexStr + std::string(".off");

                load(newFile.c_str());
            }
        }
    }
}


void OffSequenceLoader::clear()
{
    positions.beginWriteOnly()->clear();
    positions.endEdit();
    edges.beginWriteOnly()->clear();
    edges.endEdit();
    triangles.beginWriteOnly()->clear();
    triangles.endEdit();
    quads.beginWriteOnly()->clear();
    quads.endEdit();
    polygons.beginWriteOnly()->clear();
    polygons.endEdit();
    tetrahedra.beginWriteOnly()->clear();
    tetrahedra.endEdit();
    hexahedra.beginWriteOnly()->clear();
    hexahedra.endEdit();
    edgesGroups.beginWriteOnly()->clear();
    edgesGroups.endEdit();
    trianglesGroups.beginWriteOnly()->clear();
    trianglesGroups.endEdit();
    quadsGroups.beginWriteOnly()->clear();
    quadsGroups.endEdit();
    polygonsGroups.beginWriteOnly()->clear();
    polygonsGroups.endEdit();
    tetrahedraGroups.beginWriteOnly()->clear();
    tetrahedraGroups.endEdit();
    hexahedraGroups.beginWriteOnly()->clear();
    hexahedraGroups.endEdit();
}


bool OffSequenceLoader::load(const char * filename)
{
    bool fileRead = false;

    std::string cmd;
    std::ifstream file(filename);

    if (!file.good())
    {
        serr << "Cannot read file '" << m_filename << "'." << sendl;
        return false;
    }

    file >> cmd;
    if (cmd != "OFF")
    {
        serr << "Not a OFF file (header problem) '" << m_filename << "'." << sendl;
        return false;
    }

    clear();

    // -- Reading file
    fileRead = this->readOFF (file,filename);
    file.close();

    return fileRead;
}


}

}

}
