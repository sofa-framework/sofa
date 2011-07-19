/*
 * OBJExporter.h
 *
 *  Created on: 9 sept. 2009
 *      Author: froy
 */

#ifndef OBJEXPORTER_H_
#define OBJEXPORTER_H_

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

class SOFA_COMPONENT_MISC_API OBJExporter : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(OBJExporter,core::objectmodel::BaseObject);

private:
    unsigned int stepCounter;
    std::ofstream* outfile;
    std::ofstream* mtlfile;
    void writeOBJ();
    sofa::core::objectmodel::BaseContext* context;
    unsigned int maxStep;

public:
    sofa::core::objectmodel::DataFileName objFilename;
    Data<unsigned int> exportEveryNbSteps;
    Data<bool> exportAtBegin;
    Data<bool> exportAtEnd;
    bool  activateExport;
    OBJExporter();
    virtual ~OBJExporter();
    void init();
    void cleanup();
    void bwdInit();
    void handleEvent(sofa::core::objectmodel::Event *);
};

}

}

}

#endif /* OBJEXPORTER_H_ */
