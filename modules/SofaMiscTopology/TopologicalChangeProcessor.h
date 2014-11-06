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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_MISC_TOPOLOGICALCHANGEPROCESSOR_H
#define SOFA_COMPONENT_MISC_TOPOLOGICALCHANGEPROCESSOR_H

#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/topology/BaseTopology.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/Event.h>

#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>

#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/simulation/common/Visitor.h>
#include <sofa/SofaMisc.h>

#include <SofaBaseTopology/TriangleSetGeometryAlgorithms.h>
#include <sofa/defaulttype/Vec.h>

#ifdef SOFA_HAVE_ZLIB
#include <zlib.h>
#endif

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

#ifdef SOFA_FLOAT
typedef float Real; ///< alias
#else
typedef double Real; ///< alias
#endif

class TriangleIncisionInformation;

/** Read file containing topological modification. Or apply input modifications
 * A timestep has to be established for each modification.
 *
 * SIMPLE METHODE FOR THE MOMENT. DON'T HANDLE MULTIPLE TOPOLOGIES
*/
class SOFA_MISC_TOPOLOGY_API TopologicalChangeProcessor: public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(TopologicalChangeProcessor,core::objectmodel::BaseObject);


    sofa::core::objectmodel::DataFileName m_filename;
    Data < helper::vector< helper::vector <unsigned int> > > m_listChanges;

    // Parameters for time
    Data < double > m_interval;
    Data < double > m_shift;
    Data < bool > m_loop;

    // Inputs for operations on Data
    Data <bool> m_useDataInputs;
    Data <double> m_timeToRemove;
    Data <sofa::helper::vector <unsigned int> > m_edgesToRemove;
    Data <sofa::helper::vector <unsigned int> > m_trianglesToRemove;
    Data <sofa::helper::vector <unsigned int> > m_quadsToRemove;
    Data <sofa::helper::vector <unsigned int> > m_tetrahedraToRemove;
    Data <sofa::helper::vector <unsigned int> > m_hexahedraToRemove;

    Data <bool> m_saveIndicesAtInit;

    Data<Real>  m_epsilonSnapPath;
    Data<Real>  m_epsilonSnapBorder;

    Data<bool>  m_draw;


protected:
    TopologicalChangeProcessor();

    virtual ~TopologicalChangeProcessor();

    core::topology::BaseMeshTopology* m_topology;

    std::ifstream* infile;
#ifdef SOFA_HAVE_ZLIB
    gzFile gzfile;
#endif
    double nextTime;
    double lastTime;
    double loopTime;

    std::vector< TriangleIncisionInformation> triangleIncisionInformation;
    std::vector<std::string>    linesAboutIncision;


    std::vector<unsigned int>    errorTrianglesIndices;

public:
    virtual void init();

    virtual void reinit();

    virtual void readDataFile();

    virtual void handleEvent(sofa::core::objectmodel::Event* event);

    void setTime(double time);

    void processTopologicalChanges();
    void processTopologicalChanges(double time);

    bool readNext(double time, std::vector<std::string>& lines);

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MeshTopology.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<core::topology::BaseMeshTopology*>(context->getMeshTopology()) == NULL)
            return false;

        return BaseObject::canCreate(obj, context, arg);
    }

    void draw(const core::visual::VisualParams* vparams);

    void updateTriangleIncisionInformation();

protected:

    std::vector<Real> getValuesInLine(std::string line, unsigned int nbElements);

    void findElementIndex(defaulttype::Vector3 coord, int& triangleIndex, int oldTriangleIndex);
    void saveIndices();//only for incision
    void inciseWithSavedIndices();

    int findIndexInListOfTime(Real time);
};


class TriangleIncisionInformation
{
public:
    std::vector<unsigned int>      triangleIndices;
    std::vector<defaulttype::Vector3>                barycentricCoordinates;
    Real                                           timeToIncise;

    std::vector<defaulttype::Vector3>                coordinates;

    void display()
    {
        std::cout << "***(TriangleIncisionInformation)***" << std::endl;
        std::cout << "Time to incise: " << timeToIncise << std::endl;
        std::cout << "Triangle indices : ";
        for (unsigned int i = 0 ; i < triangleIndices.size() ; i++)
            std::cout << triangleIndices[i] << " ";
        std::cout <<  std::endl;
        std::cout << "Barycentric coordinates : ";
        for (unsigned int i = 0 ; i < barycentricCoordinates.size() ; i++)
            std::cout << barycentricCoordinates[i] << " | " ;
        std::cout <<  std::endl;
        std::cout << "Coordinates : ";
        for (unsigned int i = 0 ; i < coordinates.size() ; i++)
            std::cout << coordinates[i] << " | " ;
        std::cout <<  std::endl;
    }


    std::vector<defaulttype::Vector3> computeCoordinates(core::topology::BaseMeshTopology *topology)
    {
        sofa::component::topology::TriangleSetGeometryAlgorithms<defaulttype::Vec3Types>* triangleGeo;
        topology->getContext()->get(triangleGeo);

        coordinates.clear();


        if (coordinates.size() != triangleIndices.size())
        {
//                std::cout << "computeCoordinates:: about to resize coordinates with  " <<  triangleIndices.size() << std::endl;
            coordinates.resize(triangleIndices.size());
//                std::cout << "computeCoordinates:: is now resized  " <<  coordinates.size() << std::endl;
        }

        for (unsigned int i = 0 ; i < coordinates.size() ; i++)
        {
            defaulttype::Vec3Types::Coord coord[3];
            unsigned int triIndex = triangleIndices[i];

            if ( (int)triIndex >= topology->getNbTriangles())
            {
                std::cout << "ERROR(TriangleIncisionInformation::computeCoordinates) bad index to access triangles  " <<  triIndex << std::endl;
            }

            triangleGeo->getTriangleVertexCoordinates(triIndex, coord);

            coordinates[i].clear();
            for (unsigned k = 0 ; k < 3 ; k++)
            {
                coordinates[i] += coord[k] * barycentricCoordinates[i][k];
            }
        }

//            std::cout << "computeCoordinates:: size " <<  coordinates.size() << std::endl;

        return coordinates;
    }

};


} // namespace misc

} // namespace component

} // namespace sofa

#endif
