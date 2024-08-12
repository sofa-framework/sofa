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

#include <sofa/component/topology/utility/config.h>


#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include <sofa/component/topology/container/dynamic/TriangleSetGeometryAlgorithms.h>

#if SOFAMISCTOPOLOGY_HAVE_ZLIB
#include <zlib.h>
#endif // SOFAMISCTOPOLOGY_HAVE_ZLIB

#include <fstream>

namespace sofa::component::topology::utility
{

class TriangleIncisionInformation;

/** Read file containing topological modification. Or apply input modifications
 * A timestep has to be established for each modification.
 *
 * SIMPLE METHOD FOR THE MOMENT. DON'T HANDLE MULTIPLE TOPOLOGIES
*/
class SOFA_COMPONENT_TOPOLOGY_UTILITY_API TopologicalChangeProcessor: public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(TopologicalChangeProcessor,core::objectmodel::BaseObject);

    using Index = sofa::Index;
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_UTILITY()
    Data<std::string> m_filename;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_UTILITY()
    Data < type::vector< type::vector<Index> > > m_listChanges;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_UTILITY()
    Data<double> m_interval;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_UTILITY()
    Data<double> m_shift;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_UTILITY()
    Data<bool> m_loop;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_UTILITY()
    Data<bool> m_useDataInputs;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_UTILITY()
    Data<double> m_timeToRemove;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_UTILITY()
    Data<sofa::type::vector<Index> >m_pointsToRemove;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_UTILITY()
    Data<sofa::type::vector<Index> > m_edgesToRemove;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_UTILITY()
    Data<sofa::type::vector<Index> > m_trianglesToRemove;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_UTILITY()
    Data<sofa::type::vector<Index> > m_quadsToRemove;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_UTILITY()
    Data<sofa::type::vector<Index> > m_tetrahedraToRemove;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_UTILITY()
    Data <sofa::type::vector<Index> > m_hexahedraToRemove;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_UTILITY()
    Data<bool> m_saveIndicesAtInit;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_UTILITY()
    Data<SReal> m_epsilonSnapPath;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_UTILITY()
    Data<SReal> m_epsilonSnapBorder;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_TOPOLOGY_UTILITY()
    Data<bool> m_draw;

    sofa::core::objectmodel::DataFileName d_filename;
    Data < type::vector< type::vector<Index> > > d_listChanges; ///< 0 for adding, 1 for removing, 2 for cutting and associated indices.

    // Parameters for time
    Data < double > d_interval; ///< time duration between 2 actions
    Data < double > d_shift; ///< shift between times in the file and times when they will be read
    Data < bool > d_loop; ///< set to 'true' to re-read the file when reaching the end

    // Inputs for operations on Data
    Data <bool> d_useDataInputs; ///< If true, will perform operation using Data input lists rather than text file.
    Data <double> d_timeToRemove; ///< If using option useDataInputs, time at which will be done the operations. Possibility to use the interval Data also.
    Data <sofa::type::vector<Index> > d_pointsToRemove; ///< List of point IDs to be removed.
    Data <sofa::type::vector<Index> > d_edgesToRemove; ///< List of edge IDs to be removed.
    Data <sofa::type::vector<Index> > d_trianglesToRemove; ///< List of triangle IDs to be removed.
    Data <sofa::type::vector<Index> > d_quadsToRemove; ///< List of quad IDs to be removed.
    Data <sofa::type::vector<Index> > d_tetrahedraToRemove; ///< List of tetrahedron IDs to be removed.
    Data <sofa::type::vector<Index> > d_hexahedraToRemove; ///< List of hexahedron IDs to be removed.

    Data <bool> d_saveIndicesAtInit; ///< set to 'true' to save the incision to do in the init to incise even after a movement

    Data<SReal>  d_epsilonSnapPath; ///< epsilon snap path
    Data<SReal>  d_epsilonSnapBorder; ///< epsilon snap path

    Data<bool>  d_draw; ///< draw information

    /// Link to be set to the topology container in the component graph.
    SingleLink<TopologicalChangeProcessor, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected:
    TopologicalChangeProcessor();

    ~TopologicalChangeProcessor() override;

    core::topology::BaseMeshTopology* m_topology;

    std::ifstream* infile;
#if SOFAMISCTOPOLOGY_HAVE_ZLIB
    gzFile gzfile;
#endif
    double nextTime;
    double lastTime;
    double loopTime;

    std::vector< TriangleIncisionInformation> triangleIncisionInformation;
    std::vector<std::string>    linesAboutIncision;


    std::vector<Index>    errorTrianglesIndices;

public:
    void init() override;

    void reinit() override;

    virtual void readDataFile();

    void handleEvent(sofa::core::objectmodel::Event* event) override;

    void setTime(double time);

    void processTopologicalChanges();
    void processTopologicalChanges(double time);

    bool readNext(double time, std::vector<std::string>& lines);

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MeshTopology.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (context->getMeshTopology() == nullptr) {
            arg->logError("No mesh topology found in the context node.");
            return false;
        }

        return BaseObject::canCreate(obj, context, arg);
    }

    void draw(const core::visual::VisualParams* vparams) override;

    void updateTriangleIncisionInformation();

protected:

    std::vector<SReal> getValuesInLine(std::string line, size_t nbElements);

    void findElementIndex(type::Vec3 coord, Index& triangleIndex, Index oldTriangleIndex);
    void saveIndices();//only for incision
    void inciseWithSavedIndices();

    Index findIndexInListOfTime(SReal time);
};


class TriangleIncisionInformation
{
public:
    using Index = sofa::Index;

    std::vector<Index>      triangleIndices;
    std::vector<type::Vec3>                barycentricCoordinates;
    SReal                                           timeToIncise;

    std::vector<type::Vec3>                coordinates;

    void display()
    {
        std::stringstream tmp ;
        tmp<< "Time to incise: " << timeToIncise << msgendl;
        tmp<< "Triangle indices : ";
        for (Index i = 0 ; i < triangleIndices.size() ; i++)
            tmp<< triangleIndices[i] << " ";
        tmp<<  msgendl;
        tmp<< "Barycentric coordinates : ";
        for (Index i = 0 ; i < barycentricCoordinates.size() ; i++)
            tmp<< barycentricCoordinates[i] << " | " ;
        tmp<<  msgendl;
        tmp<< "Coordinates : ";
        for (Index i = 0 ; i < coordinates.size() ; i++)
            tmp<< coordinates[i] << " | " ;
        msg_info("TriangleIncisionInformation") << tmp.str() ;
    }


    std::vector<type::Vec3> computeCoordinates(core::topology::BaseMeshTopology *topology)
    {
        sofa::component::topology::container::dynamic::TriangleSetGeometryAlgorithms<defaulttype::Vec3Types>* triangleGeo;
        topology->getContext()->get(triangleGeo);

        coordinates.clear();


        if (coordinates.size() != triangleIndices.size())
        {
            coordinates.resize(triangleIndices.size());
        }

        for (unsigned int i = 0 ; i < coordinates.size() ; i++)
        {
            defaulttype::Vec3Types::Coord coord[3];
            Index triIndex = triangleIndices[i];

            if ( triIndex >= topology->getNbTriangles())
            {
                msg_error("TriangleIncisionInformation") << " Bad index to access triangles  " <<  triIndex ;
            }

            triangleGeo->getTriangleVertexCoordinates(triIndex, coord);

            coordinates[i].clear();
            for (unsigned k = 0 ; k < 3 ; k++)
            {
                coordinates[i] += coord[k] * barycentricCoordinates[i][k];
            }
        }

        return coordinates;
    }
};

} // namespace sofa::component::topology::utility
