#ifndef LABELGRIDIMAGETOOLBOX_H
#define LABELGRIDIMAGETOOLBOX_H

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/topology/BaseMeshTopology.h>


#include "labelgridimagetoolboxaction.h"
#include "../labelimagetoolbox.h"

#include "../labelpointsbysection/labelpointsbysectionimagetoolbox.h"
#include "../labelbox/labelboximagetoolbox.h"
#include <image/ImageTypes.h>


#include <image/image_gui/config.h>
#include <sofa/helper/rmath.h>




namespace sofa
{

namespace component
{

namespace engine
{

class SOFA_IMAGE_GUI_API LabelGridImageToolBoxNoTemplated: public LabelImageToolBox
{
public:
    SOFA_CLASS(LabelGridImageToolBoxNoTemplated,LabelImageToolBox);

    typedef sofa::core::objectmodel::DataFileName DataFileName;

    typedef sofa::defaulttype::Vec3d Coord3;
    typedef sofa::defaulttype::Vec3d Deriv3;
    typedef sofa::defaulttype::Vec6d Vec6d;
    typedef helper::vector< Coord3 > VecCoord3;
    typedef helper::vector< double > VecReal;
    typedef std::map<int, VecCoord3> MapSection;

    typedef Vec<5,unsigned int> imCoord;
    typedef sofa::defaulttype::ImageLPTransform<SReal> TransformType;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;

    typedef sofa::core::topology::BaseMeshTopology::QuadsAroundEdge QuadsAroundEdge;
    typedef sofa::core::topology::BaseMeshTopology::SeqQuads Quads;
    typedef sofa::core::topology::BaseMeshTopology::SeqEdges Edges;
    typedef sofa::core::topology::BaseMeshTopology::EdgesInQuad EdgesInQuad;
    typedef sofa::core::topology::BaseMeshTopology::Quad Quad;
    typedef sofa::core::topology::BaseMeshTopology::Edge Edge;
    typedef sofa::core::topology::BaseMeshTopology::QuadID QuadID;
    typedef sofa::core::topology::BaseMeshTopology::EdgeID EdgeID;
    typedef sofa::core::topology::BaseMeshTopology::EdgesAroundVertex EdgesAroundVertex;
    typedef sofa::core::topology::BaseMeshTopology::QuadsAroundVertex QuadsAoundVertex;
    typedef sofa::core::objectmodel::Tag Tag;



    //typedef typename TransformType::Coord Coord;

    typedef sofa::defaulttype::Vec<2, unsigned int> Vec2ui;

    struct InterpolationItem
    {
        Vec2ui sections;
        double ratio;

        InterpolationItem() : ratio(0.0) {}
    };

    typedef helper::vector< InterpolationItem > VecII;
    
    LabelGridImageToolBoxNoTemplated():LabelImageToolBox()
        //, d_ip(initData(&d_ip, "imageposition",""))
        //, d_p(initData(&d_p, "3Dposition",""))
        , d_reso(initData(&d_reso, Vec2ui(10,10),"resolution",""))
        , d_filename(initData(&d_filename,"filename",""))
        , d_transform(initData(&d_transform,TransformType(),"transform","Transform"))
        , d_outQuads(initData(&d_outQuads,"outQuads",""))
        , d_outEdges(initData(&d_outEdges,"outEdges",""))
        , d_outImagePosition(initData(&d_outImagePosition,"outImagePosition",""))
        //, d_outNormalImagePosition(initData(&d_outNormalImagePosition,"outNormalImagePosition",""))
        , d_outNormalImagePositionBySection(initData(&d_outNormalImagePositionBySection,"outNormalImagePositionBySection",""))
        , d_tagFilter(initData(&d_tagFilter,"tagfilter",""))
    {
    
    }
    
    void init() override
    {
        //addOutput(&d_ip);
        //addOutput(&d_p);
        addOutput(&d_outImagePosition);
        addOutput(&d_outNormalImagePositionBySection);
        addOutput(&d_outQuads);
        addOutput(&d_outEdges);

        addInput(&d_reso);
        addInput(&d_transform);
        addInput(&d_filename);


        sofa::core::objectmodel::BaseContext* context = this->getContext();

        if(d_tagFilter.getValue() != "")
        {
            Tag tag(d_tagFilter.getValue());
            labelpoints = context->get<LabelPointsBySectionImageToolBox>(tag,core::objectmodel::BaseContext::Local);
            labelbox = context->get<LabelBoxImageToolBox>(tag,core::objectmodel::BaseContext::Local);
        }

        if(!labelpoints)
        {
            labelpoints = context->get<LabelPointsBySectionImageToolBox>(core::objectmodel::BaseContext::Local);
        }

        if(!labelbox)
        {
            labelbox = context->get<LabelBoxImageToolBox>(core::objectmodel::BaseContext::Local);
        }

        if(!labelpoints)
        {
            std::cerr << "Warning: LabelGridImageToolBox no found LabelPointsBySectionImageToolBox"<<std::endl;
        }

        if(!labelbox)
        {
            std::cerr << "Warning: LabelGridImageToolBox no found LabelBoxImageToolBox"<<std::endl;
        }

        executeAction();

    }
    
    sofa::gui::qt::LabelImageToolBoxAction* createTBAction(QWidget*parent=NULL) override
    {
        return new sofa::gui::qt::LabelGridImageToolBoxAction(this,parent);
    }

    helper::vector<int> parseVector()
    {
        if(!labelpoints)return helper::vector<int>();

        unsigned int axis = labelpoints->d_axis.getValue();
        helper::vector<int> list;

        helper::vector<sofa::defaulttype::Vec3d>& vip = *(labelpoints->d_ip.beginEdit());

        list.push_back((int) (vip[vip.size()-1])[axis]);
        for(int i=vip.size()-2;i>=0;i--)
        {
            int v = (int) (vip[i])[axis];
            if(list.back() != v)list.push_back(v);
        }

        labelpoints->d_ip.endEdit();

        unsigned int i=0;
        while(i<list.size()-1)
        {
            if(list[i]==list[i+1])
            {
                list[i+1]=-1;
                i++;
            }
            else if(list[i]<list[i+1])
            {
                //std::cout << "swap " << list[i] << " " << list[i+1] << std::endl;

                std::swap(list[i],list[i+1]);
                if(i!=0) i--;
                else i++;
            }
            else i++;
        }

        while(list.back()==-1)list.pop_back();

        /////////////////////////////////////////
        //std::cout << "list sorted = [ ";
        /*for(unsigned int i=0;i<list.size();i++)
        {
          //  std::cout << list[i] << " ";
        }*/
        //std::cout << "]" << std::endl;
        /////////////////////////////////////////

        return list;
    }

    bool selectionSection(helper::vector<int> &used)
    {
        if(!labelpoints)return false;

//        std::cout << "hahahaha" << labelpoints->d_ip.getValue().size()<<std::endl;
        if(labelpoints->d_ip.getValue().size()==0)return false;
//        std::cout << "hahahaha" << labelpoints->d_ip.getValue().size()<<std::endl;

        Vec2ui reso = d_reso.getValue();
        if(reso.x()==0 || reso.y()==0)return false;

        unsigned int axis = labelpoints->d_axis.getValue();
        helper::vector<int> list = parseVector();
        used.clear();

        // verify the boundary of surface
        // include in labelbox (if it exists)

        bool maxIsBox;
        bool minIsBox;

        double max = list.front();
        double min = list.back();

        if(labelbox)
        {
            double minBox = labelbox->d_ipbox.getValue()[axis];
            double maxBox = labelbox->d_ipbox.getValue()[3+axis];

            //std::cout << minBox << " "<<maxBox << std::endl;

            if(maxBox<=max)
            {
                max=maxBox;
                maxIsBox=true;
            }
            else
            {
                maxIsBox=false;
            }

            if(minBox>=min)
            {
                min=minBox;
                minIsBox=true;
            }
            else
            {
                minIsBox=false;
            }
        }
        else
        {
            maxIsBox = false;
            minIsBox = false;
        }

        //std::cout << "min-max " << min << " " <<max << " "<<minIsBox<< " "<<maxIsBox<<std::endl;

        // create first item of section interpolation
        if(minIsBox)
        {
          //  std::cout << "ooo"<<std::endl;
            interpolationItemList.push_back( calculteII(min,list));
        }
        else
        {
            InterpolationItem it;
            it.sections = Vec2ui(min,min);
            it.ratio=1;

            interpolationItemList.push_back(it);
        }


        // create intermediates items of section interpolation
        double dt = (max-min)/(double)reso.x();
        double current = min+dt;

        for(unsigned int i=0;i<reso.x()-1;i++)
        {
            interpolationItemList.push_back( calculteII(current,list));
            current+=dt;
        }


        // create last item of section interpolation
        if(maxIsBox)
        {
            //std::cout << "lo"<<std::endl;
            interpolationItemList.push_back( calculteII(max,list));
        }
        else
        {
            InterpolationItem it;
            it.sections = Vec2ui(max,max);
            it.ratio=1;

            interpolationItemList.push_back(it);
        }

        //search used sections and avoid double
        for(unsigned int i=0;i<interpolationItemList.size();i++)
        {
            InterpolationItem &it = interpolationItemList[i];
            if(i==0)
            {
                used.push_back(it.sections.x());
            }

            if(it.sections.y() != (unsigned int)used.back())
            {
                if(it.sections.x() != (unsigned int)used.back())
                {
                    used.push_back(it.sections.x());
                }
                used.push_back(it.sections.y());
            }
        }

        //display used
        /*std::cout << "used = [ ";
        for(unsigned int i=0;i<used.size();i++)
        {
            std::cout << used[i]<< " ";
        }
        std::cout << "]"<< std::endl;
*/
        return true;
    }

    InterpolationItem calculteII(double current,helper::vector<int>& list)
    {
        //std::cout << "--------------" <<std::endl << "current "<<current<<std::endl;

        InterpolationItem it;
        int c = current;


        for(unsigned int i=0;i<list.size()-1;i++)
        {
          //  std::cout << i << std::endl;
           // std::cout << list[i] << "/"<<list[i+1] << " "<< (list[i]>= c && c>=list[i+1]) << std::endl;
            if(list[i]>= c && c>=list[i+1])
            {

                it.sections = Vec2ui(list[i+1],list[i]);
                it.ratio = (current- (double)list[i+1])/((double)list[i]-(double)list[i+1]);

             //   std::cout << "sec-rat  "<< list[i+1] << " " << list[i]<<" - "<<it.ratio<<std::endl;

                return it;
            }
            /*else if(i==0 && list[0]<c)
            {
                it.sections = Vec2ui(list[0],list[0]);
                it.ratio = 1;

                std::cout << "sec-rat2 "<< list[0] << " " << list[0]<<" - "<<it.ratio<<std::endl;

                return it;
            }*/
        }

        return it;
    }

    bool copySection(helper::vector<int> &used)
    {
        const VecCoord3& vec = labelpoints->d_ip.getValue();
        unsigned int axis = labelpoints->d_axis.getValue();
        currentMap.clear();

        int currentSection = -1;
        bool currentSectionNeeded = false;
        for(unsigned int i=0;i<vec.size();i++)
        {
            
            if(currentSection != helper::round((vec[i])[axis]))
            {
                currentSection = helper::round((vec[i])[axis]);

                currentSectionNeeded=false;
                for(unsigned int j=0;j<used.size();j++)
                {
                    if (used[j]==currentSection)
                        currentSectionNeeded=true;
                }
            }
            if(currentSectionNeeded)
            {

                (currentMap[currentSection]).push_back(vec[i]);
            }
        }

        MapSection::iterator it = currentMap.begin();
        //std::cout << "---------------"<<std::endl<< "copysection"<< std::endl;
        while(it != currentMap.end())
        {
          //  std::cout << it->first << " " << it->second.size()<<std::endl;
            ++it;
        }
        return true;
    }

    inline bool isInBox(const Coord3& c,const Vec6d& box, unsigned int axis)const
    {
        return (   ((c[0]> box[0] && c[0]<box[3])||axis==0)
                && ((c[1]> box[1] && c[1]<box[4])||axis==1)
                && ((c[2]> box[2] && c[2]<box[5])||axis==2));
    }

    Coord3 calculateIntersection(const Coord3 pIsOut, const Coord3 pIsIn, const Vec6d box, unsigned int axis)const
    {
        double ratio=0;
        double tmp=1;
        for(unsigned int i=0;i<3;i++)
        {
            if(i!=axis)
            {
                if(pIsOut[i] < box[i])
                {
                    tmp = (box[i]-pIsOut[i])/(pIsIn[i]-pIsOut[i]);
                    if(ratio<tmp)ratio=tmp;
                }

                if(pIsOut[i] > box[3+i])
                {
                    tmp = (box[3+i]-pIsOut[i])/(pIsIn[i]-pIsOut[i]);
                    if(ratio<tmp)ratio=tmp;
                }
            }
        }

        return pIsOut + (pIsIn-pIsOut)*ratio;
    }

    /**
     * @brief cutSection
     * @return
     *
     * @todo fix error when the vector is composed of 2 points (out of the box, the lines pass throuth the box )
     */

    bool cutSection()
    {
        if(!labelbox)return true;
        unsigned int axis = labelpoints->d_axis.getValue();

        const Vec6d &box = labelbox->d_ipbox.getValue();

        //std::cout << "==============================================="<<std::endl<< "Cut Section"<<std::endl;
        MapSection::iterator it = currentMap.begin();
        while(it != currentMap.end())
        {
          //  std::cout << it->first << " " << it->second.size()<< " "<< axis <<std::endl;
            VecCoord3 &vecIn = it->second;
            VecCoord3 vecOut;



            //std::cout << "box " << box << std::endl;
            //std::cout << "oldcurrentMap["<<it->first<<"] = | ";

            for(unsigned int i=0;i<vecIn.size()-1;i++)
            {
                bool v0 = isInBox(vecIn[i],box,axis);
                bool v1 = isInBox(vecIn[i+1],box,axis);
              //  std::cout << vecIn[i] <<"#"<< v0<<"#"<<v1<<" | ";
                if( v0 )
                {
                    vecOut.push_back(vecIn[i]);
                    if(!v1)
                    {
                        vecOut.push_back(calculateIntersection(vecIn[i+1],vecIn[i],box,axis));
                    }
                }
                else if (v1)
                {
                    vecOut.push_back(calculateIntersection(vecIn[i],vecIn[i+1],box,axis));
                }

                if(i==vecIn.size()-2 && v0 && !v1)
                {
                    vecOut.push_back(calculateIntersection(vecIn[i+1],vecIn[i],box,axis));
                }

                if(i==vecIn.size()-2 && v1)
                {
                    vecOut.push_back(vecIn[i+1]);
                }

            }
            /*std::cout <<vecIn.back()<< " |" <<std::endl;


            std::cout << vecOut.size()<<std::endl;
            std::cout << "newcurrentMap[] = | ";
            for(unsigned int i=0;i<vecOut.size();i++)
            {
                std::cout << vecOut[i] << " | ";
            }
            std::cout<<std::endl;
*/

            vecIn.swap(vecOut);

            ++it;
        }
        std::cout << "~Cut Section"<<std::endl<< "==============================================="<<std::endl;
//
        return true;
    }

    double calculateDistance(VecCoord3 & vec)
    {
        double distance = 0;
  //      std::cout << "distance = ";
        for(unsigned int i=1;i<vec.size();i++)
        {
            Coord3 dx = vec[i]-vec[i-1];

            distance += dx.norm();

    //        std::cout << dx.norm() << " ";
        }
      //  std::cout << " = "<<distance<<std::endl;
        return distance;
    }

    void calculateVecReso(VecCoord3 &vin, VecCoord3 &vout, VecReal &vecDistance)
    {
        double distance = 0;
        double distanceGoal = 0;

        unsigned int j=0;

        vout.push_back(vin.front());
        Coord3 dx, v1, v2;
        for(unsigned int i=0;i<vecDistance.size()-1;i++)
        {
            distanceGoal += vecDistance[i];
            //std::cout << "next distance: "<<i<< " " <<distanceGoal << std::endl;


            while( (distance < distanceGoal) && (j != vin.size()-1) )
            {
                v1 = vin[j]; v2 = vin[j+1];
                dx = v2-v1;
                distance += dx.norm();
                //std::cout << "while " << v1 << " " << v2 << " " << dx << distance<<std::endl;
                ++j;
            }

            if(distance >= distanceGoal)
            {

                double distance_min = distance - dx.norm();
                double distance_max = distance;

                double ratio = (distanceGoal-distance_min)/(distance_max-distance_min);
                //std::cout << "goal " << ratio << std::endl;

                Coord3 result = v1 + dx*ratio;

                vout.push_back(result);
            }
            else
            {
                vout.push_back(vin.back());
            }
        }
        vout.push_back(vin.back());



/*        std::cout << "oldcalculateVecReso("<< vin.size() <<") | ";
        for(unsigned int i=0;i<vin.size();i++)
            std::cout << vin[i]<< " | ";
        std::cout << std::endl;
*//*
        std::cout << "calculateVecReso("<< vout.size() <<") | ";
        for(unsigned int i=0;i<vout.size();i++)
            std::cout << vout[i]<< " | ";
        std::cout << std::endl;*/
    }

    bool changeResolutionSection()
    {
        unsigned int resolution = d_reso.getValue().y();

        MapSection::iterator it = currentMap.begin();
        //std::cout << std::endl<<"========================="<<std::endl;
        while(it != currentMap.end())
        {
            VecCoord3 &vecIn = it->second;
            VecCoord3 vecOut;
            VecReal vecDistance; vecDistance.assign(resolution,0);

          //  std::cout << resolution << "###"<< vecIn.size() << std::endl;


            //calculate distance total;
            double distance = calculateDistance(vecIn);
            VecReal vecDistError; vecDistError.assign(resolution,distance/(double)resolution);

            //std::cout << resolution << "##"<< vecIn.size() << std::endl;


            calculateVecReso(vecIn,vecOut,vecDistError);

            //std::cout << resolution << "##"<< vecIn.size() << std::endl;

            if(vecOut.size() != (resolution+1) )
                std::cerr << "BUG: "<<__FILE__ <<":l" <<__LINE__<< " ->  resolution problem (" << vecOut.size() << "instead of" << resolution+1 << ")" << std::endl;

            calculateDistance(vecOut);

            vecIn.swap(vecOut);

            ++it;
        }
        return true;
    }

    void createGrid()
    {
        helper::vector<sofa::defaulttype::Vec3d>& out = *(d_outImagePosition.beginEdit());

        //std::cout << "jhj"<<std::endl;
        out.clear();
        for(unsigned int i=0;i<interpolationItemList.size();i++)
        {
            InterpolationItem &item = interpolationItemList[i];

            VecCoord3 v1 = currentMap[item.sections.x()];
            VecCoord3 v2 = currentMap[item.sections.y()];
            double ratio = item.ratio;


            for(unsigned int j=0;j<v1.size();j++)
            {
                Coord3 v = v1[j] + (v2[j]-v1[j])*ratio;

                out.push_back(v);
            }
        }
        //std::cout << "jhj"<<std::endl;

        d_outImagePosition.endEdit();
    }

    void calculateMesh()
    {
        Edges& outEdge = *(d_outEdges.beginEdit());
        Quads& outQuad = *(d_outQuads.beginEdit());
        const Vec2ui &reso = d_reso.getValue();

        const unsigned int &resox = reso.x()+1;
        const unsigned int &resoy = reso.y()+1;

        outEdge.clear();
        outQuad.clear();
        for(unsigned int i=0;i<resox;i++)
        {
            for(unsigned int j=0;j<resoy;j++)
            {
                int id = i*resoy + j;

                if(j != resoy-1)
                {
                    outEdge.push_back(Edge(id,id+1));
                }
                if(i != resox-1)
                {
                    outEdge.push_back(Edge(id,id+resoy));
                }

                if(j != resoy-1 && i != resox-1)
                {
                    outQuad.push_back(Quad(id,id+1,id+resoy+1,id+resoy));
                }
            }
        }

        d_outEdges.endEdit();
        d_outQuads.endEdit();

        //calculate quadsAroundPoints for normal computation (point)
    }

    void calculateNormals()
    {
        //helper::vector<sofa::defaulttype::Vec3d>& out = *(d_outNormalImagePosition.beginEdit());
        helper::vector<sofa::defaulttype::Vec3d>& out2 = *(d_outNormalImagePositionBySection.beginEdit());
        helper::vector<sofa::defaulttype::Vec3d>& pos = *(d_outImagePosition.beginEdit());


        //helper::vector<sofa::defaulttype::Vec3d> out2tmp;
        //Quads& outQuad = *(d_outQuads.beginEdit());

        /*

        out.clear();
        out.resize(pos.size());

        //normals
        for(unsigned int i=0;i<outQuad.size();i++)
        {
            for(int i=0;i<4;i++)
            {
                int i0 = i;
                int i1 = (i+1)%4;
                int i2 = (i1+1)%4;


            }
        }


*/


        //normals along section
        const Vec2ui &reso = d_reso.getValue();
        const unsigned int &resox = reso.x()+1;
        const unsigned int &resoy = reso.y()+1;

        unsigned int axis1=0,axis2=1,axis=labelpoints->d_axis.getValue();

        switch(axis)
        {
            case 0:
                axis1=1;
                axis2=2;
                break;
            case 1:
                axis1=0;
                axis2=2;
                break;
            case 2:
                axis1=0;
                axis2=1;
                break;
        }

        out2.clear();
        out2.resize(pos.size());

        std::cout << out2.size() << std::endl;
        for(unsigned int i=0;i<resox;i++)
        {
            for(unsigned int j=0;j<resoy-1;j++)
            {
                unsigned int id = i*resoy + j;
                unsigned int id1 = id+1;
                //std::cout << id << " "<<id1<<std::endl;
                Deriv3 d = pos[id1]-pos[id];

                Deriv3 normal;

                normal[axis1] = -d[axis2];
                normal[axis2] = d[axis1];
                normal[axis] = 0;
                normal.normalize();
                out2[id] += normal;
                out2[id1] += normal;
            }
        }
        for(unsigned int i=0;i<out2.size();i++)
        {
            Deriv3 d = out2[i];
            d.normalize();
            out2[i]=d;
        }


        //d_outNormalImagePosition.endEdit();
        d_outNormalImagePositionBySection.endEdit();
        d_outImagePosition.endEdit();
        //d_outQuads.endEdit();
    }

    void clearTmpData()
    {
        interpolationItemList.clear();
        currentMap.clear();
    }


    void executeAction()
    {
        helper::vector<int> used;

        std::cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1"<<std::endl;

        bool b1=selectionSection(used),b2=false,b3=false,b4=false;
        std::cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa2"<<std::endl;
        if(b1)
            b2 = copySection(used);
        std::cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa3"<<std::endl;
        if(b2)
            b3 = cutSection();
        std::cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa4"<<std::endl;
        if(b3)
            b4 = changeResolutionSection();
        std::cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa5"<<std::endl;
        if(b4)
        {   
            createGrid();
            std::cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa6"<<std::endl;
            calculateMesh();
            std::cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa7"<<std::endl;
            calculateNormals();
            std::cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa8"<<std::endl;

            //calculate3DPositionOfGrid();
        }

        clearTmpData();

        std::cout << "ouuu"<<std::endl;
    }


    void calculate3DPositionOfGrid()
    {
        std::cout << "TODO: 3D position calculation"<<std::endl;
    }
    
    inline Coord3 toImage(const Coord3& p)const
    {
        return d_transform.getValue().toImage(p);
    }

    inline Coord3 fromImage(const Coord3& p)const
    {
        return d_transform.getValue().fromImage(p);
    }

public:
//    Data< helper::vector<sofa::defaulttype::Vec3d> > d_ip;
//    Data< helper::vector<sofa::defaulttype::Vec3d> > d_p;
    Data< Vec2ui > d_reso;
    DataFileName d_filename;
    Data< TransformType> d_transform; ///< Transform
    Data< Quads > d_outQuads;
    Data< Edges > d_outEdges;
    Data< helper::vector<sofa::defaulttype::Vec3d> > d_outImagePosition;
//    Data< helper::vector<sofa::defaulttype::Vec3d> > d_outNormalImagePosition;
    Data< helper::vector<sofa::defaulttype::Vec3d> > d_outNormalImagePositionBySection;
    Data< std::string > d_tagFilter;




    LabelBoxImageToolBox *labelbox;
    LabelPointsBySectionImageToolBox *labelpoints;
    imCoord resolution;


    VecII interpolationItemList;
    MapSection currentMap;


};

template<class _ImageTypes>
class SOFA_IMAGE_GUI_API LabelGridImageToolBox: public LabelGridImageToolBoxNoTemplated
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(LabelGridImageToolBox,_ImageTypes),LabelGridImageToolBoxNoTemplated);
    typedef LabelGridImageToolBoxNoTemplated Inherited;

    typedef _ImageTypes ImageTypes;
    typedef typename ImageTypes::T T;

    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    typedef helper::WriteAccessor<Data< ImageTypes > > waImage;




    LabelGridImageToolBox():LabelGridImageToolBoxNoTemplated()
        , d_image(initData(&d_image,ImageTypes(),"image","Input image"))
    {

    }

    void init() override
    {
        Inherited::init();
        addInput(&d_image);

        resolution = getDimensions();
    }

    virtual Inherited::imCoord getDimensions()
    {
        return d_image.getValue().getDimensions();
    }



public:
    Data< ImageTypes >   d_image; ///< Input image
};




}}}

#endif // LabelGridIMAGETOOLBOX_H


