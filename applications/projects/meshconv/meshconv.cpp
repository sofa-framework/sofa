/******* COPYRIGHT ************************************************
*                                                                 *
*                         FlowVR Render                           *
*                   Parallel Rendering Modules                    *
*                                                                 *
*-----------------------------------------------------------------*
* COPYRIGHT (C) 2005 by                                           *
* Laboratoire Informatique et Distribution (UMR5132) and          *
* INRIA Project MOVI. ALL RIGHTS RESERVED.                        *
*                                                                 *
* This source is covered by the GNU GPL, please refer to the      *
* COPYING file for further information.                           *
*                                                                 *
*-----------------------------------------------------------------*
*                                                                 *
*  Original Contributors:                                         *
*    Jeremie Allard,                                              *
*    Clement Menier.                                              *
*                                                                 *
*******************************************************************
*                                                                 *
* File: ./src/utils/meshconv.cpp                                  *
*                                                                 *
* Contacts:                                                       *
*                                                                 *
******************************************************************/
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include <ftl/cmdline.h>
#include <flowvr/render/mesh.h>

using namespace flowvr::render;

extern void tesselateMesh(Mesh& obj, int rec=1, bool onSphere=false);

int main(int argc, char** argv)
{
    bool normalize = false;
    bool flip = false;
    bool wnormals = false;
    bool rnormals = false;
    bool rtexcoords = false;
    bool closemesh = false;
    Vec3f translation;
    Vec3f rotation;
    Vec3f scale(1,1,1);
    float dilate = 0.0f;
    int tesselate = 0;
    bool sphere = false;
    bool dist = false;
    int res = 16;
    int rx=0,ry=0,rz=0;
    float border = 0.25f;
    ftl::CmdLine cmd("Usage: meshconv [options] mesh.input [mesh.output]");
    cmd.opt("normalize",'n',"transform points so that the center is at <0,0,0> and the max coodinate is 1",&normalize);
    cmd.opt("flip",'f',"flip normals",&flip);
    cmd.opt("wnormals",'N',"force writing of normals",&wnormals);
    cmd.opt("rnormals",'O',"remove normals",&rnormals);
    cmd.opt("rtexcoords",'T',"remove texcoords",&rtexcoords);
    cmd.opt("close",'c',"close mesh",&closemesh);
    cmd.opt("translate",'t',"translate the mesh",&translation);
    cmd.opt("rotate",'r',"rotate the mesh using euler angles in degree",&rotation);
    cmd.opt("scale",'s',"scale the mesh using 3 coefficients",&scale);
    cmd.opt("dilate",'d',"dilate (i.e. displace vertices of the given distance along their normals)",&dilate);
    cmd.opt("tesselate",'a',"tesselate (split each edge in 2 resursivly n times)",&tesselate);
    cmd.opt("sphere",'S',"consider the mesh as a sphere for tesselation",&sphere);
    cmd.opt("dist",'D',"compute distance field",&dist);
    cmd.opt("res",'R',"resolution of distance field",&res);
    cmd.opt("rx",'X',"X resolution of distance field",&rx);
    cmd.opt("ry",'Y',"Y resolution of distance field",&ry);
    cmd.opt("rz",'Z',"Z resolution of distance field",&rz);
    cmd.opt("border",'B',"distance field border size relative to the object's BBox size (or negative for exact size)",&border);
    bool error=false;
    if (!cmd.parse(argc,argv,&error))
        return error?1:0;

    if (cmd.args.size()<1 || cmd.args.size()>2)
    {
        std::cerr << cmd.help() << std::endl;
        return 1;
    }

    std::string file_in = cmd.args[0];
    std::string file_out;
    if (cmd.args.size()>=2) file_out = cmd.args[1];

    Mesh obj;

    if (!obj.load(file_in.c_str()))
    {
        std::cerr << "Failed to read "<<file_in<<std::endl;
        return 1;
    }

    if (normalize)
    {
        BBox bb = obj.calcBBox();
        std::cout << "Mesh bbox="<<bb<<std::endl;
        std::cout << "Normalizing mesh..."<<std::endl;
        float size = 0;
        for (int i=0; i<3; i++)
        {
            float d = bb.b[i]-bb.a[i];
            if (d>size) size=d;
        }
        Vec3f center = (bb.a+bb.b)*0.5;
        float sc = 2/size;
        Mat4x4f m; m.identity();
        m(0,0)=sc; m(0,3) = -center[0]*sc;
        m(1,1)=sc; m(1,3) = -center[1]*sc;
        m(2,2)=sc; m(2,3) = -center[2]*sc;
        for (int i=0; i<obj.nbp(); i++)
            obj.PP(i) = transform(m,obj.getPP(i));
    }

    if (scale != Vec3f(1,1,1))
    {
        std::cout << "Scaling mesh..."<<std::endl;
        for (int i=0; i<obj.nbp(); i++)
        {
            Vec3f p = obj.getPP(i);
            p[0] *= scale[0];
            p[1] *= scale[1];
            p[2] *= scale[2];
            obj.PP(i) = p;
        }
    }

    if (rotation != Vec3f(0,0,0))
    {
        std::cout << "Rotating mesh..."<<std::endl;
        Mat3x3f mat;
        Quat qx,qy,qz;
        qx.fromDegreeAngAxis(rotation[0],Vec3f(1,0,0));
        qy.fromDegreeAngAxis(rotation[1],Vec3f(0,1,0));
        qz.fromDegreeAngAxis(rotation[2],Vec3f(0,0,1));
        Quat q = qx*qy*qz;
        q.toMatrix(&mat);
        std::cout << "mat = "<<mat<<std::endl;
        for (int i=0; i<obj.nbp(); i++)
        {
            obj.PP(i) = mat*obj.getPP(i);
            obj.PN(i) = mat*obj.getPN(i);
        }
    }

    if (translation != Vec3f(0,0,0))
    {
        std::cout << "Translating mesh..."<<std::endl;
        for (int i=0; i<obj.nbp(); i++)
        {
            obj.PP(i) = obj.getPP(i) + translation;
        }
    }

    std::cout << "Mesh bbox="<<obj.calcBBox()<<std::endl;
    if (flip)
    {
        std::cout << "Flipping mesh..."<<std::endl;
        obj.flipAll();
        //obj.calcFlip();
    }
    if (closemesh)
    {
        bool closed = obj.isClosed();
        std::cout << "Mesh is "<<(closed?"":"NOT ")<<"closed."<<std::endl;
        if (closemesh && !closed)
        {
            obj.calcFlip();
            std::cout << "Closing mesh..."<<std::endl;
            obj.close();
            std::cout << "Mesh is "<<(obj.isClosed()?"":"NOT ")<<"closed."<<std::endl;
        }
    }

    if (dilate != 0.0f)
        obj.dilate(dilate);

    if (tesselate > 0)
    {
        std::cout << "Tesselating mesh..."<<std::endl;
        tesselateMesh(obj, tesselate, sphere);
    }

    if (dist)
    {
        if (!rx) rx = res;
        if (!ry) ry = res;
        if (!rz) rz = res;
        std::cout << "Flipping mesh..."<<std::endl;
        obj.calcFlip();
        obj.calcEdges();
        bool closed = obj.isClosed();
        std::cout << "Mesh is "<<(closed?"":"NOT ")<<"closed."<<std::endl;
        if (!closed)
        {
            std::cout << "Closing mesh..."<<std::endl;
            obj.close();
            std::cout << "Mesh is "<<(obj.isClosed()?"":"NOT ")<<"closed."<<std::endl;
        }
        std::cout << "Computing "<<rx<<'x'<<ry<<'x'<<rz<<" DistMap..."<<std::endl;
        obj.calcDistMap(rx,ry,rz,(border<0 ? -border : obj.calcBBox().size()*border));
    }


    if (wnormals)
        obj.setAttrib(Mesh::MESH_POINTS_NORMAL,true);
    if (rnormals)
        obj.setAttrib(Mesh::MESH_POINTS_NORMAL,false);
    if (rtexcoords)
        obj.setAttrib(Mesh::MESH_POINTS_TEXCOORD,false);
    if (!file_out.empty())
    {
        std::cout << "Saving result..."<<std::endl;
        obj.save(file_out.c_str());
    }
    return 0;
}
