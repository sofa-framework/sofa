

#ifndef LABELBOXIMAGETOOLBOX_H
#define LABELBOXIMAGETOOLBOX_H

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/helper/system/FileRepository.h>

#include "labelboximagetoolboxaction.h"
#include "../labelimagetoolbox.h"


#include <image/image_gui/config.h>




namespace sofa
{

namespace component
{

namespace engine
{

class SOFA_IMAGE_GUI_API LabelBoxImageToolBox: public LabelImageToolBox
{
public:
    SOFA_CLASS(LabelBoxImageToolBox,LabelImageToolBox);

    typedef sofa::core::objectmodel::DataFileName DataFileName;

    
    LabelBoxImageToolBox():LabelImageToolBox()
        , d_ip(initData(&d_ip, "imagepositions",""))
        , d_p(initData(&d_p, "3Dpositions",""))
        , d_ipbox(initData(&d_ipbox,"imagepositionbox",""))
        , d_pbox(initData(&d_pbox,"positionbox",""))
        , d_filename(initData(&d_filename,"filename",""))
    {
    
    }
    
    void init() override
    {
        addOutput(&d_ip);
        addOutput(&d_p);
        addOutput(&d_ipbox);
        addOutput(&d_ipbox);


        loadFile();
    }
    
    sofa::gui::qt::LabelImageToolBoxAction* createTBAction(QWidget*parent=NULL) override
    {
        return new sofa::gui::qt::LabelBoxImageToolBoxAction(this,parent);
    }


    sofa::defaulttype::Vec6d calculatebox(helper::vector<sofa::defaulttype::Vec3d>& vip)
    {
        sofa::defaulttype::Vec6d box;

        for(unsigned int i=0;i<vip.size();i++)
        {
            sofa::defaulttype::Vec3d& v = vip[i];

            if(i==0)
            {
                box[0]=box[3]=v.x();
                box[1]=box[4]=v.y();
                box[2]=box[5]=v.z();
            }

            if(v.x()<box[0])box[0]=v.x();
            if(v.y()<box[1])box[1]=v.y();
            if(v.z()<box[2])box[2]=v.z();
            if(v.x()>box[3])box[3]=v.x();
            if(v.y()>box[4])box[4]=v.y();
            if(v.z()>box[5])box[5]=v.z();
        }

        return box;
    }


    void calculatebox()
    {
        helper::vector<sofa::defaulttype::Vec3d>& vip = *(d_ip.beginEdit());
        helper::vector<sofa::defaulttype::Vec3d>& vp = *(d_p.beginEdit());

        d_ipbox.setValue(calculatebox(vip));
        d_pbox.setValue(calculatebox(vp));

        d_ip.endEdit();
        d_p.endEdit();
    }

    bool saveFile()
    {
        const char* filename = d_filename.getFullPath().c_str();
        std::ofstream file(filename);

        if(!file.good())
        {
            std::cerr << "Error: LabelPointsBySectionImageToolBox: Cannot write file '"<<d_filename<<"'."<<std::endl;
            return false;
        }

        bool fileWrite = this->writeData (file,filename);
        file.close();

        return fileWrite;
    }

    bool writeData(std::ofstream &file, const char* /*filename*/)
    {

        const sofa::defaulttype::Vec6d &bip = d_ipbox.getValue();
        const sofa::defaulttype::Vec6d &bp = d_pbox.getValue();

        std::ostringstream values1, values2;

        values1 << bip[0] << " " << bip[1] << " " << bip[2] << " " << bip[3] << " " << bip[4] << " " << bip[5];
        values2 << bp[0] << " " << bp[1] << " " << bp[2] << " " << bp[3] << " " << bp[4] << " " << bp[5];

        file << values1.str() << "\n";
        file << values2.str() << "\n";

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

        bool fileRead = this->readData (file,filename);
        file.close();

        return fileRead;
    }

    bool readData(std::ifstream &file, const char* /*filename*/)
    {
        helper::vector<sofa::defaulttype::Vec3d>& vip = *(d_ip.beginEdit());
        helper::vector<sofa::defaulttype::Vec3d>& vp = *(d_p.beginEdit());

        vip.clear();
        vp.clear();

        int numline=0;
        std::string line;
        while( std::getline(file,line) )
        {
            if(line.empty() || numline>2) continue;

            std::istringstream values(line);

            if(numline==0)
            {
                sofa::defaulttype::Vec3d ip1;
                sofa::defaulttype::Vec3d ip2;

                values >> ip1[0] >> ip1[1] >> ip1[2] >> ip2[0] >> ip2[1] >> ip2[2];

                vip.push_back(ip1);
                vip.push_back(ip2);
            }
            else if(numline==1)
            {
                sofa::defaulttype::Vec3d ip1;
                sofa::defaulttype::Vec3d ip2;

                values >> ip1[0] >> ip1[1] >> ip1[2] >> ip2[0] >> ip2[1] >> ip2[2];

                vp.push_back(ip1);
                vp.push_back(ip2);
            }

            numline++;
        }

        d_ip.endEdit();
        d_p.endEdit();

        this->calculatebox();

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
    Data<helper::vector <sofa::defaulttype::Vec3d> > d_ip;
    Data<helper::vector <sofa::defaulttype::Vec3d> > d_p;
    Data<sofa::defaulttype::Vec6d> d_ipbox;
    Data<sofa::defaulttype::Vec6d> d_pbox;
    DataFileName d_filename;
};


}}}

#endif // LabelBoxIMAGETOOLBOX_H

