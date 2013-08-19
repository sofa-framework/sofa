#ifndef DEPTHIMAGETOOLBOX_H
#define DEPTHIMAGETOOLBOX_H

#include <sofa/component/component.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/topology/BaseMeshTopology.h>


#include "depthimagetoolboxaction.h"
#include "../labelimagetoolbox.h"

#include "../labelgrid/labelgridimagetoolbox.h"
#include "../depth/depthimagetoolbox.h"
#include "ImageTypes.h"


#include "initImage.h"




namespace sofa
{

namespace component
{

namespace engine
{

class SOFA_IMAGE_API DepthImageToolBox: public LabelImageToolBox
{
public:
    SOFA_CLASS(DepthImageToolBox,LabelImageToolBox);

    typedef sofa::core::objectmodel::DataFileName DataFileName;

    typedef sofa::defaulttype::Vec3d Coord3;
    typedef sofa::defaulttype::Vec3d Deriv3;
    typedef sofa::defaulttype::Vec6d Vec6d;
    typedef helper::vector< Coord3 > VecCoord3;
    typedef helper::vector< double > VecReal;
    typedef LabelGridImageToolBoxNoTemplated::Vec2ui Vec2i;

    //typedef Vec<5,unsigned int> imCoord;
    typedef helper::vector<double> VecDouble;
    typedef sofa::defaulttype::ImageLPTransform<SReal> TransformType;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;

    typedef typename sofa::defaulttype::ImageD ImageD;
    typedef typename ImageD::imCoord imCoord;
    typedef helper::WriteAccessor< Data< ImageD > > waImage;

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
    typedef typename sofa::core::objectmodel::Tag Tag;

    struct Layer
    {
        enum TypeOffset {Distance=0, Percent=1};
        int layer1,layer2,base;

        int typeOffset1, typeOffset2;
        double offset1, offset2;
        std::string name;

        Layer():layer1(-1),layer2(-1),base(-1),typeOffset1(Distance),typeOffset2(Distance),offset1(0),offset2(0){}
    };

    typedef helper::vector< Layer > VecLayer;
    
    DepthImageToolBox():LabelImageToolBox()
        , d_filename(initData(&d_filename,"filename",""))
        , d_transform(initData(&d_transform,TransformType(),"transform","Transform"))
        , d_tagFilter(initData(&d_tagFilter,"tagfilter",""))
        , d_image(initData(&d_image,ImageD(),"image",""))
        , wImage(this->d_image)
    {
    }
    
    virtual void init()
    {
        addInput(&d_transform);
        addInput(&d_filename);
        addOutput(&d_image);

        sofa::core::objectmodel::BaseContext* context = this->getContext();

        if(d_tagFilter.getValue() != "")
        {
            Tag tag(d_tagFilter.getValue());
            context->get<LabelGridImageToolBoxNoTemplated>(&labelsOfGrid,tag,core::objectmodel::BaseContext::Local);
        }

        if(labelsOfGrid.size()==0)
        {
            context->get<LabelGridImageToolBoxNoTemplated>(&labelsOfGrid,core::objectmodel::BaseContext::Local);
        }

        if(labelsOfGrid.size()==0)
        {
            std::cerr << "Warning: DepthImageToolBox no found LabelGridImageToolBoxNoTemplated"<<std::endl;
        }



        executeAction();

    }
    
    virtual sofa::gui::qt::LabelImageToolBoxAction* createTBAction(QWidget*parent=NULL)
    {
        return new sofa::gui::qt::DepthImageToolBoxAction(this,parent);
    }

    void createLayer()
    {
        layers.push_back(Layer());
    }

    float pointPlaneNormalDistance(Coord3 &point, Coord3 plane[3])
    {
        Coord3 AB = plane[1]-plane[0];
        Coord3 BC = plane[2]-plane[1];
        Coord3 BM = plane[1]-point;
        Coord3 N = AB.cross(BC);
        N.normalize();

        return (BM*N);
    }

    Coord3 linePlaneIntersection(Coord3 line[2], Coord3 plane[3])
    {
        //std::cout << "linePlaneIntersection" << std::endl;
        float dP1 = pointPlaneNormalDistance(line[0],plane);
        float dP2 = pointPlaneNormalDistance(line[1],plane);
        //std::cout << plane << "//"<<line[0]<<"//"<<line[1]<<std::endl;
        //std::cout << dP1 << "/"<< dP2<< "/"<< line[0]+(line[1]-line[0])*(dP1/(dP1-dP2))<<std::endl;

        return line[0]+(line[1]-line[0])*(dP1/(dP1-dP2));
    }

    bool pointIsInTriangle(Coord3 &point, Coord3 triangle[3])
    {
        Coord3 AB = triangle[1]-triangle[0];
        Coord3 AC = triangle[2]-triangle[0];
        Coord3 BC = triangle[2]-triangle[1];
        Coord3 AM = point - triangle[0];
        Coord3 BM = point - triangle[1];
        Coord3 CM = point - triangle[2];

        if((AB.cross(AM))*(AM.cross(AC))<0-0.001)return false;
        if(((-AB).cross(BM))*(BM.cross(BC))<0-0.001)return false;
        if(((-AC).cross(CM))*(CM.cross(-BC))<0-0.001)return false;
        return true;
    }

    Coord3 calculateQuadNormalVector(const Quad &q,const VecCoord3 &p)
    {
        Coord3 r = p[q[0]]+p[q[1]]+p[q[2]]+p[q[3]];
        r.normalize();
        return r;
    }

    Coord3 calculateQuadNormalVectorFromImage(const Quad &q,const VecCoord3 &p)
    {
        Coord3 r = fromImage(p[q[0]])+fromImage(p[q[1]])+fromImage(p[q[2]])+fromImage(p[q[3]]);
        r.normalize();
        return r;
    }

    Coord3 calculateQuadNormalPosition(const Quad &q,const VecCoord3 &p)
    {
        return (p[q[0]]+p[q[1]]+p[q[2]]+p[q[3]])/4;
    }



    bool searchLineQuadIntersection(LabelGridImageToolBoxNoTemplated *s, Coord3 line[2], Coord3 &out)
    {
        //std::cout << "searchLineQuadIntersection"<<std::endl;
        const Quads &quads = s->d_outQuads.getValue();
        const VecCoord3 &pos = s->d_outImagePosition.getValue();

        double distance = -1;
        double minDistanceProjPos = -1;
        Coord3 def;

        for(unsigned int i=0;i<quads.size();i++)
        {
            //std::cout << "quad" << i << std::endl;
            Quad q = quads[i];

            for(unsigned int j=0;j<4;j++)
            {
                //std::cout << "case" << j << std::endl;

                Coord3 c[3] = { pos[q[(j+1)%4]], pos[q[(j+2)%4]], pos[q[(j+3)%4]] };

                Coord3 projection = linePlaneIntersection(line,c);

                //std::cout << projection << "//" << line[0]<<"//"<<(line[0]-projection).norm() <<std::endl;

                if(pointIsInTriangle(projection,c))
                {
                    Coord3 vp = line[0]-projection;
                    double d = vp.norm();

                    if(distance==-1 || distance>d)
                    {
                        distance=d;
                        out=projection;
                    }
                }
                else if(distance==-1)
                {
                    if(minDistanceProjPos==-1)
                    {
                        minDistanceProjPos = (projection-line[0]).norm();
                        def=projection;
                    }

                    for(int i=0;i<3;i++)
                    {
                        double d = (projection-line[0]).norm();
                        if(d<minDistanceProjPos)
                        {
                            minDistanceProjPos = d;
                            def=projection;
                        }
                    }
                }
            }
        }

        if(distance==-1)
            out = def;


        return true;

    }

    VecReal calculateSurfaceDistance(LabelGridImageToolBoxNoTemplated *b,LabelGridImageToolBoxNoTemplated *s)
    {
        //std::cout << "calculateSurfaceDistance"<<b<<std::endl;

        const Quads &quads = b->d_outQuads.getValue();

        const VecCoord3 &norm = b->d_outNormalImagePositionBySection.getValue();

        const VecCoord3 &pos = b->d_outImagePosition.getValue();

        VecReal lout;

        lout.resize(quads.size());

        //double min,max,sum;

        for(unsigned int i=0;i<quads.size();i++)
        {
            Coord3 normalVec = calculateQuadNormalVector(quads[i],norm);
            Coord3 normalPos = calculateQuadNormalPosition(quads[i],pos);

            Coord3 line[2]={normalPos, normalVec+normalPos};
            Coord3 out;

            searchLineQuadIntersection(s,line,out);

            lout[i]=(out-line[0]).norm();

            //std::cout << "ccc"<<out<< "/"<<line[0] << "/"<<lout[i]<<std::endl;
            //exit(0);
           /* if(i==0)
            {
                min=max=sum=lout[i];
            }
            else
            {
                if(min>lout[i])min=lout[i];
                if(max<lout[i])max=lout[i];
                sum += lout[i];
            }*/
        }
        //std::cout << "mma " << min << " " << max << " " << sum/(double)quads.size()<<std::endl;


        return lout;
    }

    void transfertDistanceFromImagePositionTo3DPosition(LabelGridImageToolBoxNoTemplated *b,VecReal &v)
    {
        const Quads &quads = b->d_outQuads.getValue();

        const VecCoord3 &norm = b->d_outNormalImagePositionBySection.getValue();

        for(unsigned int i=0;i<quads.size();i++)
        {
            Coord3 normalVec = calculateQuadNormalVectorFromImage(quads[i],norm);

            v[i] *= normalVec.norm();
        }
    }

    void addConstantOffsetToDistance(VecReal &v,double offset)
    {
        for(unsigned int i=0;i<v.size();i++)
            v[i]+=offset;
    }

    void addPercentOffsetToDistance(VecReal &v1,VecReal &v2,double offset)
    {
        for(unsigned int i=0;i<v1.size();i++)
            v1[i]=v1[i]*(1-offset)+v2[i]*(offset);
    }

    bool createLayerInImage(VecReal &v1,VecReal &v2,Vec2i reso,int numLayer)
    {
        std::cout << "createLayerInImage" << std::endl;
        unsigned int k=0;

        for(unsigned int i=0;i<reso.x();i++)
            for(unsigned int j=0;j<reso.y();j++)
            {
                //wImage.;
                std::cout << i << " "<<j<<" "<<wImage->getDimensions()<< std::endl;

                ImageD::CImgT &img = wImage->getCImg();
                img.atXYZC(i,j,numLayer,0) = v1[k];
                img.atXYZC(i,j,numLayer,1) = v2[k];

                k++;
            }

        return true;
    }

    bool calculateDistanceMap(Layer&l,int numLayer)
    {
        std::cout << "calculateDistanceMap"<<l.layer1<<" "<<l.layer2<<" "<<l.base<<std::endl;
        if(l.layer1>=labelsOfGrid.size() || l.layer2>=labelsOfGrid.size()  || l.base>=labelsOfGrid.size())return false;

        LabelGridImageToolBoxNoTemplated *surf1 = labelsOfGrid[l.layer1];
        LabelGridImageToolBoxNoTemplated *surf2 = labelsOfGrid[l.layer2];
        LabelGridImageToolBoxNoTemplated *base = labelsOfGrid[l.base];

        //Quads& out = *(d_outImagePosition.beginEdit());

        VecReal v1 = calculateSurfaceDistance(base,surf1);
        VecReal v2 = calculateSurfaceDistance(base,surf2);

        transfertDistanceFromImagePositionTo3DPosition(base,v1);
        transfertDistanceFromImagePositionTo3DPosition(base,v2);

        switch(l.typeOffset1)
        {
            case Layer::Distance:
                addConstantOffsetToDistance(v1,l.offset1);
                break;
            case Layer::Percent:
                addPercentOffsetToDistance(v1,v2,l.offset1);
                break;
            default:
                break;
        }

        switch(l.typeOffset2)
        {
            case Layer::Distance:
                addConstantOffsetToDistance(v2,l.offset2);
                break;
            case Layer::Percent:
                addPercentOffsetToDistance(v2,v1,l.offset2);
                break;
            default:
                break;
        }

        createLayerInImage(v1,v2,base->d_reso.getValue(),numLayer);

        std::cout << toImage(1) << "/"<<fromImage(1)<<"/"<< toImage(Coord3(1.0,1.0,1.0)) << "/"<<fromImage(Coord3(435.934, 778.795, -70.2253)) <<std::endl;
        std::cout << d_transform.getValue() << std::endl;

        return true;
    }

    void initImage()
    {
        std::cout << "initImage()" << std::endl;

        Vec2i resomax(1,1);
        //int nbSpectrum = layers.size()*2;

        for(unsigned int i=0;i<layers.size();i++)
        {
            LabelGridImageToolBoxNoTemplated *base = labelsOfGrid[layers[i].base];

            Vec2i current = base->d_reso.getValue();

            std::cout << current << std::endl;

            if(current.x()>resomax.x())resomax.x() = (current.x());
            if(current.y()>resomax.y())resomax.y() = (current.y());
        }

        std::cout << "resomax " << resomax << std::endl;
        //std::cout << nbSpectrum << std::endl;

        int size = (layers.size())?layers.size():1;
        imCoord c(resomax.x(),resomax.y(),size,2,1);
        wImage->setDimensions(c);

        std::cout << "->" << c << "->"<<wImage->getDimensions() << std::endl;


    }


    void executeAction()
    {
        std::cout << "executeAction"<<std::endl;

        initImage();

        for(unsigned int i=0;i<layers.size();i++)
        {
            Layer &current = layers[i];
            calculateDistanceMap(current,i);
        }
    }
    
    inline Coord3 toImage(const Coord3& p)const
    {
        return d_transform.getValue().toImage(p);
    }

    inline Coord3 fromImage(const Coord3& p)const
    {
        return d_transform.getValue().fromImage(p);
    }

    inline double toImage(const double& p)const
    {
        return d_transform.getValue().toImage(p);
    }

    inline double fromImage(const double& p)const
    {
        return d_transform.getValue().fromImage(p);
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

        bool fileWrite = this->writeLayers (file,filename);
        file.close();

        return fileWrite;
    }

    void writeOffset(std::ostringstream& value, double& offset, int &type)
    {
        switch(type)
        {
            case Layer::Distance:
                value << offset << " d";
                break;
            case Layer::Percent:
                value << offset*100 <<" %";
                break;
        };
    }

    bool writeLayers(std::ofstream &file, const char* /*filename*/)
    {
        std::ostringstream values;
        values << layers.size();
        file << values.str() << "\n";

        for(unsigned int i=0;i<layers.size();i++)
        {
            Layer &l=layers[i];
            std::ostringstream values;

            values << "layer: " <<l.name << " # " << l.base << " # " << l.layer1<< " ";
            writeOffset(values, l.offset1,l.typeOffset1);
            values << " # " << l.layer2 << " ";
            writeOffset(values,l.offset2,l.typeOffset2);
            file << values.str() << "\n";
        }
    }

    void readOffset(std::string &in,double &offset ,int &typeLayer)
    {
        if(in == "%")
        {
            offset *= 0.01;
            typeLayer = Layer::Percent;
        }

        if(in == "d")
        {
            typeLayer = Layer::Distance;
        }

    }

    bool readLayers(std::ifstream &file, const char* /*filename*/)
    {


        bool firstline=true;
        std::string line;

        int i=0;
        while( std::getline(file,line) )
        {
            if(line.empty()) continue;

            std::istringstream values(line);

            if(firstline)
            {
                /*int axis;

                values >> axis;

                d_axis.setValue(axis);

                firstline=false;*/
            }
            else
            {
                /*
                sofa::defaulttype::Vec3d ip;
                sofa::defaulttype::Vec3d p;

                values >> ip[0] >> ip[1] >> ip[2] >> p[0] >> p[1] >> p[2];

                vip.push_back(ip);
                vp.push_back(p);*/

                Layer &l = layers[i];

                std::string dummy1,dummy2,dummy3;
                std::string offset1,offset2;

                values >> dummy1 >> l.name >> dummy2 >> l.layer1 >> l.offset1 >> offset1 >> dummy3 >> l.layer2 >> l.offset2 >> offset2;

                readOffset(offset1,l.offset1,l.typeOffset1);
                readOffset(offset2,l.offset2,l.typeOffset2);

                i++;
            }
        }

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

        bool fileRead = this->readLayers (file,filename);
        file.close();

        return fileRead;
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
    DataFileName d_filename;
    Data< TransformType> d_transform;

    Data< VecDouble > d_outImagePosition;

    Data< std::string > d_tagFilter;
    helper::vector< LabelGridImageToolBoxNoTemplated* > labelsOfGrid;

    Data< sofa::defaulttype::ImageD > d_image;
    waImage wImage;

    VecLayer layers;
};

}}}

#endif // DepthImageToolBox_H


