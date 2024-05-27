#ifndef DEPTHIMAGETOOLBOX_H
#define DEPTHIMAGETOOLBOX_H

#include <image_gui/config.h>

#include <QTextStream>
#include <QFile>

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/topology/BaseMeshTopology.h>


#include "depthimagetoolboxaction.h"
#include "../labelimagetoolbox.h"

#include "../labelgrid/labelgridimagetoolbox.h"
#include "../depth/depthimagetoolbox.h"
#include <image/ImageTypes.h>
#include "meshData.h"




namespace sofa
{

namespace component
{

namespace engine
{

class SOFA_IMAGE_GUI_API DepthImageToolBox: public LabelImageToolBox
{
public:
    SOFA_CLASS(DepthImageToolBox,LabelImageToolBox);

    typedef sofa::core::objectmodel::DataFileName DataFileName;

    typedef sofa::type::Vec3d Coord3;
    typedef sofa::type::Vec3d Deriv3;
    typedef sofa::type::Vec6d Vec6d;
    typedef type::vector< Coord3 > VecCoord3;
    typedef type::vector< double > VecReal;
    typedef LabelGridImageToolBoxNoTemplated::Vec2ui Vec2i;

    //typedef Vec<5,unsigned int> imCoord;
    typedef type::vector<double> VecDouble;
    typedef sofa::defaulttype::ImageLPTransform<SReal> TransformType;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;

    typedef sofa::defaulttype::ImageD ImageD;
    typedef ImageD::imCoord imCoord;
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
    typedef sofa::core::objectmodel::Tag Tag;

    struct Layer
    {
        enum TypeOffset {Distance=0, Percent=1};
        int layer1,layer2,base;

        int typeOffset1, typeOffset2;
        double offset1, offset2;
        std::string name;
        int nbSlice;

        Layer():layer1(-1),layer2(-1),base(-1),typeOffset1(Distance),typeOffset2(Distance),offset1(0),offset2(0),nbSlice(-1){}
    };

    typedef type::vector< Layer > VecLayer;
    
    DepthImageToolBox():LabelImageToolBox()
        , d_filename(initData(&d_filename,"filename",""))
        , d_scnfilename(initData(&d_scnfilename,"scnfilename",""))
        , d_transform(initData(&d_transform,TransformType(),"transform","Transform"))
        , d_tagFilter(initData(&d_tagFilter,"tagfilter",""))
        , d_image(initData(&d_image,ImageD(),"image",""))
        , wImage(this->d_image)
    {
    }
    
    void init() override
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
    
    sofa::gui::qt::LabelImageToolBoxAction* createTBAction(QWidget*parent=nullptr) override
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
        float dP1 = pointPlaneNormalDistance(line[0],plane);
        float dP2 = pointPlaneNormalDistance(line[1],plane);

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
        Coord3 r = p[q[0]]+p[q[1]]+p[q[2]]+p[q[3]];
        Coord3 g(0,0,0);

        r.normalize();

        r = fromImage(r);//r.normalize();
        g = fromImage(g);

        r -=g;

        return r;
    }

    Coord3 calculateQuadNormalPosition(const Quad &q,const VecCoord3 &p)
    {
        return (p[q[0]]+p[q[1]]+p[q[2]]+p[q[3]])/4;
    }



    bool searchLineQuadIntersection(LabelGridImageToolBoxNoTemplated *s, Coord3 line[2], Coord3 &out)
    {
        const Quads &quads = s->d_outQuads.getValue();
        const VecCoord3 &pos = s->d_outImagePosition.getValue();

        double distance = -1;
        double minDistanceProjPos = -1;
        Coord3 def;



        for(unsigned int i=0;i<quads.size();i++)
        {
            Quad q = quads[i];

            for(unsigned int j=0;j<4;j++)
            {
                Coord3 c[3] = { pos[q[(j+1)%4]], pos[q[(j+2)%4]], pos[q[(j+3)%4]] };

                Coord3 projection = linePlaneIntersection(line,c);

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
                        minDistanceProjPos = (projection-c[0]).norm();
                        def=projection;
                    }

                    for(unsigned int kk=0; kk < 3; kk++)
                    {
                        double d = (projection-c[kk]).norm();
                        if (d < minDistanceProjPos)
                        {
                            minDistanceProjPos = d;
                            def=projection;
                        }
                    }
                }
            }
        }

        if(distance==-1)
        {
            out = def;
        }

        return true;

    }

    VecReal calculateSurfaceDistance(LabelGridImageToolBoxNoTemplated *b,LabelGridImageToolBoxNoTemplated *s)
    {
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

        }

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
        {
            v[i]+=offset;


        }

    }

    void addPercentOffsetToDistance(VecReal &v1,VecReal &v2,double offset)
    {
        for(unsigned int i=0;i<v1.size();i++)
            v1[i]=v1[i]*(1-offset)+v2[i]*(offset);
    }

    bool createLayerInImage(VecReal &v1,VecReal &v2,Vec2i reso,int numLayer)
    {
        unsigned int k=0;

        for(unsigned int i=0;i<reso.x();i++)
            for(unsigned int j=0;j<reso.y();j++)
            {
                ImageD::CImgT &img = wImage->getCImg();
                img.atXYZC(i,j,numLayer,0) = v1[k];
                img.atXYZC(i,j,numLayer,1) = v2[k];

                k++;
            }

        return true;
    }

    bool createMeshFromLayer(LabelGridImageToolBoxNoTemplated *base, VecReal &v1,VecReal &v2,Vec2i reso,unsigned int numLayer, Layer& l)
    {
        if( meshs.veclayer.size() <= numLayer )
            meshs.veclayer.push_back(MeshDataImageToolBox::Layer());

        MeshDataImageToolBox::VecCoord3 &position = meshs.positions;
        MeshDataImageToolBox::Layer &layer = meshs.veclayer[numLayer];
        layer.clear();
        layer.name=l.name;

        unsigned int k0=position.size();

        int &slices = l.nbSlice;
        layer.setSlice(slices);

        type::vector<sofa::type::Vec3d>& baseIP = *(base->d_outImagePosition.beginEdit());
        type::vector<sofa::type::Vec3d>& baseN = *(base->d_outNormalImagePositionBySection.beginEdit());

        Quads& baseQuad = *(base->d_outQuads.beginEdit());

        //modify position vector
        unsigned int nbposition = (reso.x()+1)*(reso.y()+1)*(slices+1);
        for(unsigned int i=0;i<nbposition;i++)position.push_back(Coord3(0,0,0));

        VecReal sumByMesh;
        for(unsigned int i=0;i<nbposition;i++)sumByMesh.push_back(0);

        for(unsigned int k=0;k<baseQuad.size();k++)
        {
            Quad &quad = baseQuad[k];


            double dt = (double)1.0/(double)slices;

            for(int h=0;h<slices+1;h++)
            {
                double dt_h = dt*h;

                for(unsigned int q=0;q<4;q++)
                {
                    sofa::type::Vec3d baseIPk = fromImage(baseIP[quad[q]]);
                    sofa::type::Vec3d baseNk = fromImage(baseN[quad[q]]) - fromImage(sofa::type::Vec3d(0,0,0));
                    baseNk.normalize();

                    sofa::type::Vec3d min = baseIPk + baseNk*v1[k];
                    sofa::type::Vec3d max = baseIPk + baseNk*v2[k];

                    sofa::type::Vec3d point = min*dt_h+max*(1-dt_h);

                    unsigned int index =  (quad[q]*(slices+1)) + h;
                    position[k0 +index] += (/*fromImage(*/point/*)*/);
                    layer.positionIndexOfSlice[h].push_back(k0+index);
                    sumByMesh[index] += 1;
                }
            }
        }

        for(unsigned int i=k0;i<position.size();i++)
        {
            if(sumByMesh[i-k0]!=0)
                position[i]/=sumByMesh[i-k0];

        }

        //create edge Around the domain
        std::map<unsigned int,int> mapEdge;
        std::map<unsigned int,int>::iterator it;
        for(unsigned int i=0;i<baseQuad.size();i++)
        {
            Quad &quad = baseQuad[i];

            for(unsigned int j=0;j<4;j++)
            {
                auto &p = quad[j];

                it=mapEdge.find(p);
                if(it==mapEdge.end())
                {
                    std::cout << "map " << p << " true" << std::endl;
                    mapEdge[p]=0;
                }
                else
                {
                    std::cout << "map " << p << " false"<< std::endl;
                    mapEdge[p]++;
                }
            }
        }
        it = mapEdge.begin();
        while(it != mapEdge.end())
        {
            if(it->second < 2)
            {
                std::cout << "mmap"<< it->first << " "<<it->second<<std::endl;
                for(int i=0;i<slices+1;i++)
                {
                    std::cout << " ->"<<k0+it->first*(slices+1)+i<<std::endl;

                    layer.edgePositionIndexOfSlice[i].push_back(k0+it->first*(slices+1)+i);
                }
            }

            ++it;
        }



        MeshDataImageToolBox::VecIndex4& outG1 = layer.grid1;
        MeshDataImageToolBox::VecIndex4& outG2 = layer.grid2;

        //create grids
        //unsigned int k=0;
        for(unsigned int j=0;j<reso.y();j++)
        for(unsigned int i=0;i<reso.x();i++)
        {
            unsigned int p1 = j*(reso.x()+1)+i;
            unsigned int p2 = j*(reso.x()+1)+i+1;
            unsigned int p3 = (j+1)*(reso.x()+1)+i;
            unsigned int p4 = (j+1)*(reso.x()+1)+i+1;

            p1*=(slices+1);
            p2*=(slices+1);
            p3*=(slices+1);
            p4*=(slices+1);

            MeshDataImageToolBox::Index4 q1(p1+k0,p2+k0,p4+k0,p3+k0);
            outG1.push_back(q1);

            p1+=slices;
            p2+=slices;
            p3+=slices;
            p4+=slices;

            MeshDataImageToolBox::Index4 q2(p1+k0,p2+k0,p4+k0,p3+k0);

            outG2.push_back(q2);

        }

        //create extern mesh
        MeshDataImageToolBox::VecIndex3 &outM = layer.triangles;

        for(unsigned int i=0;i<layer.grid1.size();i++)
        {
            MeshDataImageToolBox::Index4 &q = layer.grid1[i];

            MeshDataImageToolBox::Index3 tmpv1(q[2],q[1],q[0]);
            MeshDataImageToolBox::Index3  tmpv2(q[3],q[2],q[0]);

            outM.push_back(tmpv1);
            outM.push_back(tmpv2);
        }

        for(unsigned int i=0;i<layer.grid2.size();i++)
        {
            MeshDataImageToolBox::Index4  &q = layer.grid2[i];

            MeshDataImageToolBox::Index3  tmpv1(q[0],q[1],q[2]);
            MeshDataImageToolBox::Index3  tmpv2(q[0],q[2],q[3]);

            outM.push_back(tmpv1);
            outM.push_back(tmpv2);
        }

        //create Hexa
        MeshDataImageToolBox::VecIndex8 &outH = layer.hexas;

        for(unsigned int i=0;i<layer.grid2.size();i++)
        {
            MeshDataImageToolBox::Index4  &q = layer.grid1[i];

            for(int j=0;j<slices;j++)
            {
                MeshDataImageToolBox::Index8  tmpv1(q[0]+j,q[1]+j,q[2]+j,q[3]+j,q[0]+j+1,q[1]+j+1,q[2]+j+1,q[3]+j+1);

                layer.hexaIndexOfSlice[j].push_back(outH.size());
                outH.push_back(tmpv1);
            }
        }

        //create Tetra
        MeshDataImageToolBox::VecIndex4 &outT = layer.tetras;

        for(unsigned int i=0;i<layer.hexas.size();i++)
        {
            MeshDataImageToolBox::Index8  &q = layer.hexas[i];

            MeshDataImageToolBox::Index4 tmpv1(q[0],q[1],q[3],q[4]);
            MeshDataImageToolBox::Index4 tmpv2(q[3],q[1],q[4],q[6]);
            MeshDataImageToolBox::Index4 tmpv3(q[4],q[5],q[6],q[1]);
            MeshDataImageToolBox::Index4 tmpv4(q[3],q[1],q[2],q[6]);
            MeshDataImageToolBox::Index4 tmpv5(q[7],q[3],q[6],q[4]);

            uint ss = outT.size();
            uint sl = i%slices;
            layer.tetraIndexOfSlice[sl].push_back(ss);
            layer.tetraIndexOfSlice[sl].push_back(ss+1);
            layer.tetraIndexOfSlice[sl].push_back(ss+2);
            layer.tetraIndexOfSlice[sl].push_back(ss+3);
            layer.tetraIndexOfSlice[sl].push_back(ss+4);
            outT.push_back(tmpv1);
            outT.push_back(tmpv2);
            outT.push_back(tmpv3);
            outT.push_back(tmpv4);
            outT.push_back(tmpv5);
        }

        return true;
    }

    bool calculateDistanceMap(Layer&l,unsigned int numLayer)
    {
        if((size_t)l.layer1>=labelsOfGrid.size() || (size_t)l.layer2>=labelsOfGrid.size()  || (size_t)l.base>=labelsOfGrid.size())return false;

        LabelGridImageToolBoxNoTemplated *surf1 = labelsOfGrid[l.layer1];
        LabelGridImageToolBoxNoTemplated *surf2 = labelsOfGrid[l.layer2];
        LabelGridImageToolBoxNoTemplated *base = labelsOfGrid[l.base];

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
        createMeshFromLayer(base,v1,v2,base->d_reso.getValue(),numLayer,l);

        return true;
    }

    void initImage()
    {
        Vec2i resomax(1,1);

        for(unsigned int i=0;i<layers.size();i++)
        {
            LabelGridImageToolBoxNoTemplated *base = labelsOfGrid[layers[i].base];

            Vec2i current = base->d_reso.getValue();

            if(current.x()>resomax.x())resomax.x() = (current.x());
            if(current.y()>resomax.y())resomax.y() = (current.y());
        }

        int size = (layers.size())?layers.size():1;
        imCoord c(resomax.x(),resomax.y(),size,2,1);
        wImage->setDimensions(c);
    }


    void executeAction()
    {
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
            std::ostringstream tmpvalues;

            tmpvalues << "layer: " <<l.name << " # " << l.base << " # " << l.layer1<< " ";
            writeOffset(tmpvalues, l.offset1,l.typeOffset1);
            tmpvalues << " # " << l.layer2 << " ";
            writeOffset(tmpvalues,l.offset2,l.typeOffset2);
            file << tmpvalues.str() << "\n";
        }
        return true;
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
                sofa::type::Vec3d ip;
                sofa::type::Vec3d p;

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

        file.close();
        return true;
    }

    bool saveSCN()
    {
        QFile file(QString::fromStdString(d_scnfilename.getValue()));

        if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
                 return false;

        QTextStream out(&file);
        //out << "The magic number is: " << 49 << "\n" <<

        //MeshDataImageToolBox &m = meshs;


        out << "  <Node name=\"MeshModel\">" << "\n";

        for(unsigned int i=0;i<meshs.veclayer.size();i++)
        {/*
            out << "    <VisualModel name=\"grid1_"<< i <<"\" position=\"";

            for(unsigned int j=0;j<meshs.positions.size();j++ )
            {
                out << meshs.positions[j].x() << " "  << meshs.positions[j].y() << " " << meshs.positions[j].z() << " ";
            }

            out << "\" quads=\"";
            for(unsigned int j=0;j<meshs.veclayer[i].grid1.size();j++ )
            {
                out << meshs.veclayer[i].grid1[j].x() << " " << meshs.veclayer[i].grid1[j].y() << " " << meshs.veclayer[i].grid1[j].z() << " " << meshs.veclayer[i].grid1[j].w() << " ";
            }

            out << "\" useNormals=\"f\" color=\"1 0 0 1\"/>" << "\n";

            out << "    <VisualModel name=\"grid2_"<< i <<"\" position=\"";

            for(unsigned int j=0;j<meshs.positions.size();j++ )
            {
                out << meshs.positions[j].x() << " "  << meshs.positions[j].y() << " " << meshs.positions[j].z() << " ";
            }

            out << "\" quads=\"";
            for(unsigned int j=0;j<meshs.veclayer[i].grid2.size();j++ )
            {
                out << meshs.veclayer[i].grid2[j].x() << " " << meshs.veclayer[i].grid2[j].y() << " " << meshs.veclayer[i].grid2[j].z() << " " << meshs.veclayer[i].grid2[j].w() << " ";
            }

            out << "\" useNormals=\"f\" color=\"1 0 0 1\"/>" << "\n";*/

            QString name =QString::fromStdString(meshs.veclayer[i].name);
            out << "<Node name=\"" << name << "\">\n";

            saveSCN_layer(out, i);
            saveSCN_grid1(out, i);
            saveSCN_grid2(out, i);
            saveSCN_indexLayer(out,i);

            std::list<unsigned int> vecG1, vecG2;
            saveSCN_indexGrid1(out,i,vecG1);
            saveSCN_indexGrid2(out,i,vecG2);

            saveSCN_simplifiedGrid1(out,i,vecG1);
            saveSCN_simplifiedGrid2(out,i,vecG2);

            saveSCN_positionIndexSlice(out,i);
            saveSCN_edgepositionIndexSlice(out,i);
            saveSCN_hexaIndexSlice(out,i);
            saveSCN_tetraIndexSlice(out,i);


            out << "</Node>\n";
        }

        saveSCN_mergedLayer(out);

        out << "  </Node> " << "\n";





        //<nVisualModel name="grid1_0" position="@Grid1.position" quads="@Grid1.quads" useNormals="f" color="1 0 0 1"/>
        //<nVisualModel name="grid2_0" position="@Grid2.position" quads="@Grid2.quads" useNormals="f" color="1 1 0 1"/>
        //<VisualModel name="layer_0" position="@Layer.position" quads="@Layer.quads" useNormals="f" color="1 0 1 1"/>

        file.close();

        return true;
    }

    void saveSCN_layer(QTextStream &out, int index)
    {
        QString name="Layer";

        out << " <Mesh name=\"" << name << "\"";
        saveSCN_position(out);
        saveSCN_indexTriangle(out, meshs.veclayer[index].triangles);
        saveSCN_indexTetra(out, meshs.veclayer[index].tetras);
        saveSCN_indexHexa(out, meshs.veclayer[index].hexas);
        out << "/>\n";
    }

    void saveSCN_positionIndexSlice(QTextStream &out, int index)
    {
        int nbSlice = meshs.veclayer[index].nbSlice;
        for(int i=0;i<nbSlice+1;i++)
        {
            out <<" <CatchAllVector name=\"SlicePositionIndex" << i << "\" data=\"";

            MeshDataImageToolBox::VecIndex & vecl = meshs.veclayer[index].positionIndexOfSlice[i];
            std::list<unsigned int> vec;

            for(size_t j=0;j<vecl.size();j++)vec.push_back(vecl[j]);


            vec.sort();
            vec.unique();

            for(std::list<unsigned int>::iterator it=vec.begin();it!= vec.end();++it)
            {
                out << *it << " ";
            }
            out << "\"/>\n";
        }
    }

    void saveSCN_edgepositionIndexSlice(QTextStream &out, int index)
    {
        int nbSlice = meshs.veclayer[index].nbSlice;
        for(int i=0;i<nbSlice+1;i++)
        {
            out <<" <CatchAllVector name=\"SliceEdgePositionIndex" << i << "\" data=\"";

            MeshDataImageToolBox::VecIndex & vecl = meshs.veclayer[index].edgePositionIndexOfSlice[i];
            std::list<unsigned int> vec;

            for(size_t j=0;j<vecl.size();j++)vec.push_back(vecl[j]);

            vec.sort();
            vec.unique();

            for(std::list<unsigned int>::iterator it=vec.begin();it!= vec.end();++it)
            {
                out << *it << " ";
            }
            out << "\"/>\n";
        }
    }

    void saveSCN_hexaIndexSlice(QTextStream &out, int index)
    {
        int nbSlice = meshs.veclayer[index].nbSlice;
        for(int i=0;i<nbSlice;i++)
        {
            out <<" <CatchAllVector name=\"SliceHexaIndex" << i << "\" data=\"";

            MeshDataImageToolBox::VecIndex & vecl = meshs.veclayer[index].hexaIndexOfSlice[i];
            std::list<unsigned int> vec;

            for(size_t j=0;j<vecl.size();j++)vec.push_back(vecl[j]);

            vec.sort();
            vec.unique();

            for(std::list<unsigned int>::iterator it=vec.begin();it!= vec.end();++it)
            {
                out << *it << " ";
            }
            out << "\"/>\n";
        }
    }

    void saveSCN_tetraIndexSlice(QTextStream &out, int index)
    {
        int nbSlice = meshs.veclayer[index].nbSlice;
        for(int i=0;i<nbSlice;i++)
        {
            out <<" <CatchAllVector name=\"SliceTetraIndex" << i << "\" data=\"";

            MeshDataImageToolBox::VecIndex & vecl = meshs.veclayer[index].tetraIndexOfSlice[i];
            std::list<unsigned int> vec;

            for(size_t j=0;j<vecl.size();j++)vec.push_back(vecl[j]);

            vec.sort();
            vec.unique();

            for(std::list<unsigned int>::iterator it=vec.begin();it!= vec.end();++it)
            {
                out << *it << " ";
            }
            out << "\"/>\n";
        }
    }

    void saveSCN_indexLayer(QTextStream &out, int index)
    {
        out << " <CatchAllVector name=\"TrianglesIndex\" data=\"";
        unsigned int offset1 = 0;
        for(int i=0;i<index;i++)
        {
            offset1 += meshs.veclayer[i].triangles.size();
        }

        for(unsigned int i=0;i<meshs.veclayer[index].triangles.size();i++)
        {
            out << i+offset1 << " ";
        }

        out << "\" />\n";

        out << " <CatchAllVector name=\"TetraIndex\" data=\"";
        unsigned int offset2 = 0;
        for(int i=0;i<index;i++)
        {
            offset2 += meshs.veclayer[i].tetras.size();
        }

        for(unsigned int i=0;i<meshs.veclayer[index].tetras.size();i++)
        {
            out << i+offset2 << " ";
        }

        out << "\" />\n";

        out << " <CatchAllVector name=\"HexaIndex\" data=\"";
        unsigned int offset3 = 0;
        for(int i=0;i<index;i++)
        {
            offset3 += meshs.veclayer[i].hexas.size();
        }

        for(unsigned int i=0;i<meshs.veclayer[index].hexas.size();i++)
        {
            out << i+offset3 << " ";
        }

        out << "\" />\n";

        out << " <CatchAllVector name=\"Grid1Index\" data=\"";
        unsigned int offset4 = 0;
        for(int i=0;i<index;i++)
        {
            offset4 += meshs.veclayer[i].grid1.size();
        }

        for(unsigned int i=0;i<meshs.veclayer[index].grid1.size();i++)
        {
            out << i+offset4 << " ";
        }

        out << "\" />\n";

        out << " <CatchAllVector name=\"Grid2Index\" data=\"";
        unsigned int offset5 = 0;
        for(int i=0;i<index;i++)
        {
            offset5 += meshs.veclayer[i].grid2.size();
        }

        for(unsigned int i=0;i<meshs.veclayer[index].grid2.size();i++)
        {
            out << i+offset5 << " ";
        }

        out << "\" />\n";
    }

    void saveSCN_simplifiedGrid1(QTextStream &out, int index, std::list<unsigned int> &vec)
    {
        out << " <Mesh name=\"SimplifiedGrid1\" position=\"";
        saveSCN_clearPositionGrid1(out,vec);
        out << "\" quads=\"";
        saveSCN_clearIndexGrid1(out,index,vec);
        out << "\" />\n";
    }

    void saveSCN_simplifiedGrid2(QTextStream &out, int index, std::list<unsigned int> &vec)
    {
        out << " <Mesh name=\"SimplifiedGrid2\" position=\"";
        saveSCN_clearPositionGrid2(out,vec);
        out << "\" quads=\"";
        saveSCN_clearIndexGrid2(out,index,vec);
        out << "\" />\n";
    }

    void saveSCN_clearPositionGrid1(QTextStream &out, /*int index,*/ std::list<unsigned int> &vec)
    {
        saveSCN_baseClearPositionGrid(out,vec);
    }

    void saveSCN_clearPositionGrid2(QTextStream &out, /*int index,*/ std::list<unsigned int> &vec)
    {
        saveSCN_baseClearPositionGrid(out,vec);
    }

    void saveSCN_baseClearPositionGrid(QTextStream &out, std::list<unsigned int> &vec)
    {
        VecCoord3 &pos = meshs.positions;

        for(std::list<unsigned int>::iterator it=vec.begin();it!= vec.end();++it)
        {
            Coord3 &c= pos[*it];

            out << c.x() << " " << c.y() << " " << c.z() << " ";
        }
    }

    void saveSCN_clearIndexGrid1(QTextStream &out, int index, std::list<unsigned int> &vec)
    {
        saveSCN_baseClearIndexGrid(out,meshs.veclayer[index].grid1,vec);
    }

    void saveSCN_clearIndexGrid2(QTextStream &out, int index, std::list<unsigned int> &vec)
    {
        saveSCN_baseClearIndexGrid(out,meshs.veclayer[index].grid2,vec);
    }

    void saveSCN_baseClearIndexGrid(QTextStream &out,MeshDataImageToolBox::VecIndex4 &grid, std::list<unsigned int> &vec)
    {
        for(unsigned int i=0;i<grid.size();i++)
        {
            MeshDataImageToolBox::Index4 &quad= grid[i];

            for(unsigned int j=0;j<4;j++)
            {
                unsigned int &point = quad[j];

                unsigned int k=0;
                for(std::list<unsigned int>::iterator it=vec.begin();it!= vec.end();++it)
                {
                    if(*it==point)
                    {
                        out << k << " ";
                        k=vec.size();
                    }

                    k++;
                }
            }
        }
    }

    void saveSCN_indexGrid1(QTextStream &out, int index, std::list<unsigned int> &vec)
    {
        out << " <CatchAllVector name=\"Grid1UsedPositionIndex\" data=\"";
        saveSCN_baseIndexGrid(out,meshs.veclayer[index].grid1,vec);
        out << "\" />\n";
    }

    void saveSCN_indexGrid2(QTextStream &out, int index, std::list<unsigned int> &vec)
    {
        out << " <CatchAllVector name=\"Grid2UsedPositionIndex\" data=\"";
        saveSCN_baseIndexGrid(out,meshs.veclayer[index].grid2,vec);
        out << "\" />\n";
    }

    void saveSCN_baseIndexGrid(QTextStream &out,MeshDataImageToolBox::VecIndex4 &grid, std::list<unsigned int> &vec)
    {
        for(unsigned int i=0;i<grid.size();i++)
        {
            MeshDataImageToolBox::Index4 & q = grid[i];
            for(int j=0;j<4;j++)
            {
                vec.push_back(q[j]);
            }
        }

        vec.sort();
        vec.unique();
        for(std::list<unsigned int>::iterator it=vec.begin();it!= vec.end();++it)
        {
            out << *it << " ";
        }
    }

    void saveSCN_mergedLayer(QTextStream &out)
    {
        out << "<Node name=\"Merged\">\n";

        out << " <Mesh name=\"Merged\" ";
        saveSCN_position(out);
        saveSCN_indexTriangle(out);
        saveSCN_indexTetra(out);
        saveSCN_indexHexa(out);
        out << "/>\n";


        out << "</Node>\n";
    }


    void saveSCN_grid1(QTextStream &out, int index)
    {
        QString name="Grid1";
        saveSCN_grid(out, meshs.veclayer[index].grid1,name);
    }

    void saveSCN_grid2(QTextStream &out, int index)
    {
        QString name="Grid2";
        saveSCN_grid(out, meshs.veclayer[index].grid2,name);
    }

    void saveSCN_grid(QTextStream &out, MeshDataImageToolBox::VecIndex4 &grid, QString &meshName)
    {
        out << " <Mesh name=\"" << meshName << "\"";
        saveSCN_position(out);
        saveSCN_indexQuad(out,grid);
        out << "/> \n";
    }

    void saveSCN_position(QTextStream &out)
    {
        out << " position=\"";
        for(unsigned int j=0;j<meshs.positions.size();j++ )
        {
            out << meshs.positions[j].x() << " "  << meshs.positions[j].y() << " " << meshs.positions[j].z() << " ";
        }
        out << "\"";
    }

    void saveSCN_indexQuad(QTextStream &out,MeshDataImageToolBox::VecIndex4 &grid)
    {
        out << " quads=\"";
        saveSCN_index4(out,grid);
        out << "\"";
    }

    void saveSCN_indexTetra(QTextStream &out,MeshDataImageToolBox::VecIndex4 &tetra)
    {
        if(tetra.size()==0)return;
        out << " tetrahedra=\"";
        saveSCN_index4(out,tetra);
        out << "\"";
    }

    void saveSCN_indexTetra(QTextStream &out)
    {
        out << " tetrahedra=\"";
        for(unsigned int j=0;j<meshs.veclayer.size();j++ )
        {
            saveSCN_index4(out,meshs.veclayer[j].tetras);
        }
        out << "\"";
    }



    void saveSCN_indexHexa(QTextStream &out,MeshDataImageToolBox::VecIndex8 &hexa)
    {
        if(hexa.size()==0)return;
        out << " hexahedra=\"";
        saveSCN_index8(out,hexa);
        out << "\"";
    }

    void saveSCN_indexHexa(QTextStream &out)
    {
        out << " hexahedra=\"";
        for(unsigned int j=0;j<meshs.veclayer.size();j++ )
        {
            saveSCN_index8(out,meshs.veclayer[j].hexas);
        }
        out << "\"";
    }

    void saveSCN_indexTriangle(QTextStream &out,MeshDataImageToolBox::VecIndex3 &triangles)
    {
        if(triangles.size()==0)return;
        out << " triangles=\"";
        saveSCN_index3(out,triangles);
        out << "\"";
    }

    void saveSCN_indexTriangle(QTextStream &out)
    {
        out << " triangles=\"";
        for(unsigned int j=0;j<meshs.veclayer.size();j++ )
        {
            saveSCN_index3(out,meshs.veclayer[j].triangles);
        }
        out << "\"";
    }

    void saveSCN_index4(QTextStream &out,MeshDataImageToolBox::VecIndex4 &index4)
    {
        for(unsigned int j=0;j<index4.size();j++ )
        {
            out << index4[j].x() << " " << index4[j].y() << " " << index4[j].z() << " " << index4[j].w() << " ";
        }
    }

    void saveSCN_index3(QTextStream &out,MeshDataImageToolBox::VecIndex3 &index3)
    {
        for(unsigned int j=0;j<index3.size();j++ )
        {
            out << index3[j].x() << " " << index3[j].y() << " " << index3[j].z() << " ";
        }
    }

    void saveSCN_index8(QTextStream &out,MeshDataImageToolBox::VecIndex8 &index8)
    {
        for(unsigned int j=0;j<index8.size();j++ )
        {
            out << index8[j][0] << " "<< index8[j][1] << " "<<index8[j][2] << " "<<index8[j][3] << " "<<index8[j][4] << " "<<index8[j][5] << " "<<index8[j][6] << " "<<index8[j][7] << " ";
        }
    }

public:
    DataFileName d_filename;
    DataFileName d_scnfilename;

    Data< TransformType> d_transform; ///< Transform

    Data< VecDouble > d_outImagePosition;

    Data< std::string > d_tagFilter;
    type::vector< LabelGridImageToolBoxNoTemplated* > labelsOfGrid;

    Data< sofa::defaulttype::ImageD > d_image;
    waImage wImage;

    VecLayer layers;

    MeshDataImageToolBox meshs;
};

}}}

#endif // DepthImageToolBox_H


