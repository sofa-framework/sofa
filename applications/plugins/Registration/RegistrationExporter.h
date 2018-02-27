#ifndef SOFA_COMPONENT_MISC_REGISTRATIONEXPORTER_H
#define SOFA_COMPONENT_MISC_REGISTRATIONEXPORTER_H

#include <Registration/config.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/behavior/BaseMechanicalState.h>


#include <fstream>

namespace sofa
{

namespace component
{

namespace misc
{

class SOFA_REGISTRATION_API RegistrationExporter : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(RegistrationExporter,core::objectmodel::BaseObject);
	typedef defaulttype::Mat<4,4,SReal> Mat4x4;

protected:
	sofa::core::behavior::BaseMechanicalState* mstate;
    unsigned int stepCounter;

	std::vector<std::string> inFileNames;
	std::vector<std::string> outFileNames;
	std::vector<Mat4x4> inverseTransforms;
    std::string getMeshFilename(const char* ext);

public:
    sofa::core::objectmodel::DataFileName outPath;
    Data<defaulttype::Vec3Types::VecCoord> position; ///< points position (will use mechanical state if this is empty)
    Data<bool> applyInverseTransform; ///< apply inverse transform specified in loaders
    Data<unsigned int> exportEveryNbSteps; ///< export file only at specified number of steps (0=disable)
    Data<bool> exportAtBegin; ///< export file at the initialization
    Data<bool> exportAtEnd; ///< export file when the simulation is finished

	RegistrationExporter();
	virtual ~RegistrationExporter();

    void writeMesh();

	void init();
	void cleanup();
	void bwdInit();

	void handleEvent(sofa::core::objectmodel::Event *);
};

} // namespace misc

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MISC_REGISTRATIONEXPORTER_H
