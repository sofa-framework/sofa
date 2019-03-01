

#ifndef LABELPOINTSBYSECTIONIMAGETOOLBOX_H
#define LABELPOINTSBYSECTIONIMAGETOOLBOX_H

#include <QDataStream>
#include "../labelimagetoolbox.h"

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseObjectDescription.h>
#include <sofa/core/objectmodel/DataFileName.h>

#include "labelpointsbysectionimagetoolboxaction.h"
#include <sofa/helper/system/FileRepository.h>

#include <string>
#include <cstring>

#include <QDockWidget>

#include <image/image_gui/config.h>




namespace sofa
{

namespace component
{

namespace engine
{

class SOFA_IMAGE_GUI_API LabelPointsBySectionImageToolBox: public LabelImageToolBox
{
public:
    SOFA_CLASS(LabelPointsBySectionImageToolBox,LabelImageToolBox);
    
    typedef sofa::gui::qt::LabelPointsBySectionImageToolBoxAction::Point Point;
    typedef sofa::gui::qt::LabelPointsBySectionImageToolBoxAction::VecPointSection VecPointSection;
    typedef sofa::gui::qt::LabelPointsBySectionImageToolBoxAction::MapSection MapSection;
    typedef sofa::core::objectmodel::DataFileName DataFileName;

    LabelPointsBySectionImageToolBox():LabelImageToolBox()
        , d_ip(initData(&d_ip, "imagepositions",""))
        , d_p(initData(&d_p, "3Dpositions",""))
        , d_axis(initData(&d_axis, (unsigned int)4,"axis",""))
        , d_filename(initData(&d_filename,"filename",""))
    {
    
    }
    
    void init() override
    {
        addOutput(&d_ip);
        addOutput(&d_p);
        addOutput(&d_axis);

        loadFile();
        
    }
    
    sofa::gui::qt::LabelImageToolBoxAction* createTBAction(QWidget*parent=NULL) override
    {
        sofa::gui::qt::LabelPointsBySectionImageToolBoxAction * t = new sofa::gui::qt::LabelPointsBySectionImageToolBoxAction(this,parent);

        return t;
    }


    bool saveFile()
    {
        std::cout << "save" << std::endl;
        const char* filename = d_filename.getFullPath().c_str();
        std::ofstream file(filename);

        if(!file.good())
        {
            std::cerr << "Error: LabelPointsBySectionImageToolBox: Cannot write file '"<<d_filename<<"'."<<std::endl;
            return false;
        }

        bool fileWrite = this->writePointList (file,filename);
        file.close();

        return fileWrite;
    }

    bool writePointList(std::ofstream &file, const char* /*filename*/)
    {
        helper::vector<sofa::defaulttype::Vec3d>& vip = *(d_ip.beginEdit());
        helper::vector<sofa::defaulttype::Vec3d>& vp = *(d_p.beginEdit());

        int size = vip.size();
        if(vip.size() != vp.size())
        {
            std::cerr << "Warning: the imagepositions vector size is different of the 3Dpositions vector size.";
            if(vip.size()>vp.size())size=vp.size();
        }

        std::ostringstream values;

        values << d_axis.getValue();
        file << values.str() << "\n";

        for(int i=0;i<size;i++)
        {

            std::ostringstream values;


            sofa::defaulttype::Vec3d &ip = vip[i];
            sofa::defaulttype::Vec3d &p = vp[i];

            values << ip[0] << " "<< ip[1] <<" "<< ip[2] <<" "<< p[0] <<" "<< p[1] <<" "<< p[2];

            file << values.str() << "\n";
        }

        d_ip.endEdit();
        d_p.endEdit();

        return true;
    }


    bool loadFile()
    {
        if(!canLoad())return false;

        const char* filename = d_filename.getFullPath().c_str();
        std::ifstream file(filename);

        if(!file.good())
        {
            std::cerr << "Error: LabelPointsBySectionImageToolBox: Cannot read file '"<<d_filename<<"'."<<std::endl;
            return false;
        }

        bool fileRead = this->readPointList (file,filename);
        file.close();

        return fileRead;
    }

    bool readPointList(std::ifstream &file, const char* /*filename*/)
    {
        helper::vector<sofa::defaulttype::Vec3d>& vip = *(d_ip.beginEdit());
        helper::vector<sofa::defaulttype::Vec3d>& vp = *(d_p.beginEdit());

        bool firstline=true;
        std::string line;
        while( std::getline(file,line) )
        {
            if(line.empty()) continue;

            std::istringstream values(line);

            if(firstline)
            {
                int axis;

                values >> axis;

                d_axis.setValue(axis);

                firstline=false;
            }
            else
            {
                sofa::defaulttype::Vec3d ip;
                sofa::defaulttype::Vec3d p;

                values >> ip[0] >> ip[1] >> ip[2] >> p[0] >> p[1] >> p[2];

                vip.push_back(ip);
                vp.push_back(p);
            }
        }


        d_ip.endEdit();
        d_p.endEdit();

        return true;
    }

    bool canLoad()
    {
        if(d_filename.getValue() == "")
        {
            std::cerr << "Error: LabelPointsBySectionImageToolBox: No file name given"<<std::endl;
            return false;
        }

        const char* filename = d_filename.getFullPath().c_str();

        std::string sfilename(filename);

        if(!sofa::helper::system::DataRepository.findFile(sfilename))
        {
            std::cerr << "Error: LabelPointsBySectionImageToolBox: File '"<<d_filename<<"' not found."<<std::endl;
            return false;
        }

        std::ifstream file(filename);
        if(!file.good())
        {
            std::cerr << "Error: LabelPointsBySectionImageToolBox: Cannot read file '"<<d_filename<<"'."<<std::endl;
            return false;
        }

        std::string cmd;

        file >> cmd;
        if(cmd.empty())
        {
            std::cerr << "Error: LabelPointsBySectionImageToolBox: Cannot read first line in file '"<<d_filename<<"'."<<std::endl;
            return false;
        }


        std::cout << "load from " <<filename << std::endl;
        file.close();
        return true;
    }
    
public:
    Data< helper::vector<sofa::defaulttype::Vec3d> > d_ip;
    Data< helper::vector<sofa::defaulttype::Vec3d> >d_p;
    Data<unsigned int> d_axis;
    DataFileName d_filename;
};


}}}

#endif // LabelPointsBySectionImageToolBox_H


