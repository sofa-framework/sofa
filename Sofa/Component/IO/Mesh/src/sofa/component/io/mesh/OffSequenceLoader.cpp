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

#include <sofa/component/io/mesh/OffSequenceLoader.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/core/ObjectFactory.h>
#include <sstream>
#include <fstream>

namespace sofa::component::io::mesh
{

void registerOffSequenceLoader(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Read and load an .off file at each timestep.")
        .add< OffSequenceLoader >());
}

OffSequenceLoader::OffSequenceLoader()
    : MeshOffLoader()
    , d_nbFiles(initData(&d_nbFiles, (int)1, "nbOfFiles", "number of files in the sequence") )
    , d_stepDuration(initData(&d_stepDuration, 0.04, "stepDuration", "how long each file is loaded") )
    , firstIndex(0) , currentIndex(0)
{
    this->f_listening.setValue(true);

    d_edges.setDisplayed(false);
    d_triangles.setDisplayed(false);
    d_quads.setDisplayed(false);
    d_polygons.setDisplayed(false);
    d_tetrahedra.setDisplayed(false);
    d_hexahedra.setDisplayed(false);
    d_edgesGroups.setDisplayed(false);
    d_trianglesGroups.setDisplayed(false);
    d_quadsGroups.setDisplayed(false);
    d_polygonsGroups.setDisplayed(false);
    d_tetrahedraGroups.setDisplayed(false);
    d_hexahedraGroups.setDisplayed(false);

    nbFiles.setOriginalData(&d_nbFiles);
    stepDuration.setOriginalData(&d_stepDuration);

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

    if (this->d_filename.getValue().empty())
    {
        msg_error() << "Attribute '" << this->d_filename.getName() << "' is empty.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    //parse the file name to get the index part
    std::string file = this->d_filename.getFullPath();
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
        if ((currentIndex-firstIndex) * d_stepDuration.getValue() <= this->getContext()->getTime())
        {
            currentIndex++;
            if (currentIndex < firstIndex + d_nbFiles.getValue())
            {
                std::ostringstream os;
                os << currentIndex;
                const std::string indexStr = os.str();
                const std::string filetmp = m_filenameAndNb.substr(0, m_filenameAndNb.size()-indexStr.size());

                const std::string newFile = filetmp + indexStr + std::string(".off");

                load(newFile.c_str());
            }
        }
    }
}


void OffSequenceLoader::clear()
{
    d_positions.beginWriteOnly()->clear();
    d_positions.endEdit();
    d_edges.beginWriteOnly()->clear();
    d_edges.endEdit();
    d_triangles.beginWriteOnly()->clear();
    d_triangles.endEdit();
    d_quads.beginWriteOnly()->clear();
    d_quads.endEdit();
    d_polygons.beginWriteOnly()->clear();
    d_polygons.endEdit();
    d_tetrahedra.beginWriteOnly()->clear();
    d_tetrahedra.endEdit();
    d_hexahedra.beginWriteOnly()->clear();
    d_hexahedra.endEdit();
    d_edgesGroups.beginWriteOnly()->clear();
    d_edgesGroups.endEdit();
    d_trianglesGroups.beginWriteOnly()->clear();
    d_trianglesGroups.endEdit();
    d_quadsGroups.beginWriteOnly()->clear();
    d_quadsGroups.endEdit();
    d_polygonsGroups.beginWriteOnly()->clear();
    d_polygonsGroups.endEdit();
    d_tetrahedraGroups.beginWriteOnly()->clear();
    d_tetrahedraGroups.endEdit();
    d_hexahedraGroups.beginWriteOnly()->clear();
    d_hexahedraGroups.endEdit();
}


bool OffSequenceLoader::load(const char * filename)
{
    bool fileRead = false;

    std::string cmd;
    std::ifstream file(filename);

    if (!file.good())
    {
        msg_error() << "Cannot read file '" << d_filename << "'.";
        return false;
    }

    file >> cmd;
    if (cmd != "OFF")
    {
        msg_error() << "Not a OFF file (header problem) '" << d_filename << "'.";
        return false;
    }

    clear();

    // -- Reading file
    fileRead = this->readOFF (file,filename);
    file.close();

    return fileRead;
}


} // namespace sofa::component::io::mesh
