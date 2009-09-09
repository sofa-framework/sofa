/*
 * VTKExporter.h
 *
 *  Created on: 9 sept. 2009
 *      Author: froy
 */

#ifndef VTKEXPORTER_H_
#define VTKEXPORTER_H_

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/component/component.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

class SOFA_COMPONENT_MISC_API VTKExporter : public core::objectmodel::BaseObject
{
private:
    sofa::core::componentmodel::topology::BaseMeshTopology* topology;

    std::ofstream* outfile;

    void writeVTK();

public:
    sofa::core::objectmodel::DataFileName vtkFilename;
    Data<bool> writeEdges;
    Data<bool> writeTriangles;
    Data<bool> writeQuads;
    Data<bool> writeTetras;
    Data<bool> writeHexas;

    VTKExporter();
    virtual ~VTKExporter();

    void init();

    void handleEvent(sofa::core::objectmodel::Event *);
};

}

}

}

#endif /* VTKEXPORTER_H_ */
