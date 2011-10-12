/*
 * VTKExporter.h
 *
 *  Created on: 9 sept. 2009
 *      Author: froy
 */

#ifndef VTKEXPORTER_H_
#define VTKEXPORTER_H_

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/component.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/BaseMechanicalState.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

class SOFA_EXPORTER_API VTKExporter : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(VTKExporter,core::objectmodel::BaseObject);

private:
    sofa::core::topology::BaseMeshTopology* topology;
    sofa::core::behavior::BaseMechanicalState* mstate;
    unsigned int stepCounter;

    std::ofstream* outfile;

    void fetchDataFields(const helper::vector<std::string>& strData, helper::vector<std::string>& objects, helper::vector<std::string>& fields, helper::vector<std::string>& names);
    void writeVTKSimple();
    void writeVTKXML();
    void writeParallelFile();
    void writeData(const helper::vector<std::string>& objects, const helper::vector<std::string>& fields, const helper::vector<std::string>& names);
    void writeDataArray(const helper::vector<std::string>& objects, const helper::vector<std::string>& fields, const helper::vector<std::string>& names);
    std::string segmentString(std::string str, unsigned int n);

public:
    sofa::core::objectmodel::DataFileName vtkFilename;
    Data<bool> fileFormat;	//0 for Simple Legacy Formats, 1 for XML File Format
    Data<defaulttype::Vec3Types::VecCoord> position;
    Data<bool> writeEdges;
    Data<bool> writeTriangles;
    Data<bool> writeQuads;
    Data<bool> writeTetras;
    Data<bool> writeHexas;
    Data<helper::vector<std::string> > dPointsDataFields;
    Data<helper::vector<std::string> > dCellsDataFields;
    Data<unsigned int> exportEveryNbSteps;
    Data<bool> exportAtBegin;
    Data<bool> exportAtEnd;

    int nbFiles;

    helper::vector<std::string> pointsDataObject;
    helper::vector<std::string> pointsDataField;
    helper::vector<std::string> pointsDataName;

    helper::vector<std::string> cellsDataObject;
    helper::vector<std::string> cellsDataField;
    helper::vector<std::string> cellsDataName;
protected:
    VTKExporter();
    virtual ~VTKExporter();
public:
    void init();
    void cleanup();
    void bwdInit();

    void handleEvent(sofa::core::objectmodel::Event *);
};

}

}

}

#endif /* VTKEXPORTER_H_ */
