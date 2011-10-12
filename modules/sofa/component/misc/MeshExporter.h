#ifndef SOFA_COMPONENT_MISC_MESHEXPORTER_H
#define SOFA_COMPONENT_MISC_MESHEXPORTER_H

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/component.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/helper/OptionsGroup.h>

#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

class SOFA_EXPORTER_API MeshExporter : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(MeshExporter,core::objectmodel::BaseObject);

protected:
    sofa::core::topology::BaseMeshTopology* topology;
    sofa::core::behavior::BaseMechanicalState* mstate;
    unsigned int stepCounter;

    int nbFiles;

    std::string getMeshFilename(const char* ext);

public:
    sofa::core::objectmodel::DataFileName meshFilename;
    Data<sofa::helper::OptionsGroup> fileFormat;
    Data<defaulttype::Vec3Types::VecCoord> position;
    Data<bool> writeEdges;
    Data<bool> writeTriangles;
    Data<bool> writeQuads;
    Data<bool> writeTetras;
    Data<bool> writeHexas;
    //Data<helper::vector<std::string> > dPointsDataFields;
    //Data<helper::vector<std::string> > dCellsDataFields;
    Data<unsigned int> exportEveryNbSteps;
    Data<bool> exportAtBegin;
    Data<bool> exportAtEnd;

    helper::vector<std::string> pointsDataObject;
    helper::vector<std::string> pointsDataField;
    helper::vector<std::string> pointsDataName;

    helper::vector<std::string> cellsDataObject;
    helper::vector<std::string> cellsDataField;
    helper::vector<std::string> cellsDataName;
protected:
    MeshExporter();
    virtual ~MeshExporter();
public:
    void writeMesh();
    void writeMeshVTKXML();
    void writeMeshVTK();
    void writeMeshGmsh();
    void writeMeshNetgen();
    void writeMeshTetgen();

    void init();
    void cleanup();
    void bwdInit();

    void handleEvent(sofa::core::objectmodel::Event *);
};

} // namespace misc

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MISC_MESHEXPORTER_H
