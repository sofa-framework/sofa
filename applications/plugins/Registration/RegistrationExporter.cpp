#include "RegistrationExporter.h"

#include <sstream>
#include <iomanip>

#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <SofaLoader/MeshObjLoader.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/Quat.h>

#include <sofa/helper/system/FileRepository.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(RegistrationExporter)

int RegistrationExporterClass = core::RegisterObject("Replicate loaded obj files into path, with current positions")
.add< RegistrationExporter >();

RegistrationExporter::RegistrationExporter()
: stepCounter(0)
, outPath( initData(&outPath, "path", "output path"))
, position( initData(&position, "position", "points position (will use mechanical state if this is empty)"))
, applyInverseTransform( initData(&applyInverseTransform, false, "applyInverseTransform", "apply inverse transform specified in loaders"))
, exportEveryNbSteps( initData(&exportEveryNbSteps, (unsigned int)0, "exportEveryNumberOfSteps", "export file only at specified number of steps (0=disable)"))
, exportAtBegin( initData(&exportAtBegin, false, "exportAtBegin", "export file at the initialization"))
, exportAtEnd( initData(&exportAtEnd, false, "exportAtEnd", "export file when the simulation is finished"))
{
    f_listening.setValue(true);
}

RegistrationExporter::~RegistrationExporter()
{
}

void RegistrationExporter::init()
{
    sofa::core::objectmodel::BaseContext* context = this->getContext();
    context->get(mstate);

    if (!position.isSet() && mstate)
    {
        sofa::core::objectmodel::BaseData* parent = mstate->findData("position");
        if (parent)
        {
            position.setParent(parent);
            position.setReadOnly(true);
        }
    }

    std::vector<sofa::component::loader::MeshObjLoader*> loaders;
    this->getContext()->get<sofa::component::loader::MeshObjLoader,std::vector<sofa::component::loader::MeshObjLoader*> >(&loaders);

    if (!loaders.size()) 		serr << "Can not find MeshObjLoaders in context" << sendl;
    else for(unsigned int l=0;l<loaders.size();l++)
    {
        std::string strIn=loaders[l]->getFilename();
        if (sofa::helper::system::DataRepository.findFile(strIn))
        {
            this->inFileNames.push_back(sofa::helper::system::DataRepository.getFile(strIn));
            this->outFileNames.push_back(outPath.getFullPath().c_str() + this->inFileNames.back().substr(this->inFileNames[l].find_last_of('/')));
            if (this->f_printLog.getValue()) std::cout<<"RegistrationExporter: "<<this->inFileNames.back()<<"  ->  "<<this->outFileNames.back()<<std::endl;

            // get inverse transforms applied in loader
                        defaulttype::Vector3 scale=loaders[l]->getScale();
                        Mat4x4 m_scale; m_scale.fill(0);   for(unsigned int i=0;i<3;i++)	 m_scale[i][i]=1./scale[i]; m_scale[3][3]=1.;
                        defaulttype::Quaternion q = helper::Quater< SReal >::createQuaterFromEuler(defaulttype::Vec< 3, SReal >(loaders[l]->getRotation()) * M_PI / 180.0);
                        defaulttype::Mat<4,4,SReal> m33; q.inverse().toMatrix(m33);
                        Mat4x4 m_rot; m_rot.fill(0);   for(unsigned int i=0;i<3;i++)  for(unsigned int j=0;j<3;j++)	 m_rot[i][j]=m33[i][j];  m_rot[3][3]=1.;
                        defaulttype::Vector3 translation=loaders[l]->getTranslation();
                        Mat4x4 m_translation; m_translation.fill(0);   for(unsigned int i=0;i<3;i++)	 m_translation[i][3]=-translation[i];  for(unsigned int i=0;i<4;i++)	 m_translation[i][i]=1;
                        this->inverseTransforms.push_back(m_scale*m_rot*m_translation);
                        if (this->f_printLog.getValue()) std::cout<<"RegistrationExporter: transform = "<<this->inverseTransforms.back()<<std::endl;
               }
    }
}

void RegistrationExporter::writeMesh()
{
    // replicate loaded obj files with updated positions
    helper::ReadAccessor<Data< defaulttype::Vec3Types::VecCoord > > raPositions = position;
    unsigned int count=0;
    std::string line;

    for(unsigned int l=0;l<this->inFileNames.size();l++)
    {
            std::ifstream fileIn(this->inFileNames[l].c_str(), std::ifstream::in);
            if (fileIn.is_open())
            {
                std::ofstream fileOut (this->outFileNames[l].c_str(), std::ofstream::out);
                if (fileOut.is_open() )
                {
                    while( std::getline(fileIn,line) )
                    {
                        if (line.empty()) continue;
                        std::istringstream values(line);
                        std::string token;

                        values >> token;
                        if (token == "v") {
                            defaulttype::Vec< 4, SReal > p(raPositions[count][0],raPositions[count][1],raPositions[count][2],1);
                                                         if(applyInverseTransform.getValue()) p=inverseTransforms[l]*p;
                            if(count<raPositions.size()) fileOut<<"v "<<p[0]<<" "<<p[1]<<" "<<p[2]<<std::endl;
                            count++;
                            }
                        else fileOut<<line<<std::endl;
                    }
                    sout << "Written " << this->outFileNames[l].c_str() << sendl;
                    if (this->f_printLog.getValue()) std::cout<<"Written " << this->outFileNames[l].c_str() << std::endl;
                    fileOut.close();
                }
                fileIn.close();
            }
       }
}



void RegistrationExporter::handleEvent(sofa::core::objectmodel::Event *event)
{
    if (sofa::core::objectmodel::KeypressedEvent::checkEventType(event))
    {
        sofa::core::objectmodel::KeypressedEvent* ev = static_cast<sofa::core::objectmodel::KeypressedEvent*>(event);
        //std::cout << "key pressed " << ev->getKey() << std::endl;
        switch(ev->getKey())
        {

            case 'E':
            case 'e':
                writeMesh();
            break;
        }
    }
    else if ( /*simulation::AnimateEndEvent* ev =*/ simulation::AnimateEndEvent::checkEventType(event))
    {
        unsigned int maxStep = exportEveryNbSteps.getValue();
        if (maxStep == 0) return;

        stepCounter++;
        if(stepCounter >= maxStep)
        {
            stepCounter = 0;
            writeMesh();
        }
    }
}

void RegistrationExporter::cleanup()
{
    if (exportAtEnd.getValue())
        writeMesh();
}

void RegistrationExporter::bwdInit()
{
    if (exportAtBegin.getValue())
        writeMesh();
}

} // namespace misc

} // namespace component

} // namespace sofa
