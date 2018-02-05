/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
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
    float closedist = 0.0f;
    float mergedist = -1.0f;
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
    float vsize = 0.0f;
    ftl::CmdLine cmd("Usage: meshconv [options] mesh.input [mesh.output]");
    cmd.opt("normalize",'n',"transform points so that the center is at <0,0,0> and the max coodinate is 1",&normalize);
    cmd.opt("flip",'f',"flip normals",&flip);
    cmd.opt("wnormals",'N',"force writing of normals",&wnormals);
    cmd.opt("rnormals",'O',"remove normals",&rnormals);
    cmd.opt("rtexcoords",'T',"remove texcoords",&rtexcoords);
    cmd.opt("close",'c',"close mesh",&closemesh);
    cmd.opt("close2",'C',"close mesh creating intermediate vertices no further appart than given dist",&closedist);
    cmd.opt("merge",'m',"merge vertices closer than the given distance",&mergedist);
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
    cmd.opt("vsize",'V',"size of each voxel in distance field",&vsize);
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

    Mat4x4f xform;
    bool hasXForm = false;
    xform.identity();

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
        hasXForm = true;
        xform = mat;
    }

    if (translation != Vec3f(0,0,0))
    {
        std::cout << "Translating mesh..."<<std::endl;
        for (int i=0; i<obj.nbp(); i++)
        {
            obj.PP(i) = obj.getPP(i) + translation;
        }
        hasXForm = true;
        xform[0][3] = translation[0];
        xform[1][3] = translation[1];
        xform[2][3] = translation[2];
    }

    if (obj.distmap && hasXForm)
    {
        //Mat4x4f m; m.invert(xform);
        std::cout << "distmap mat = "<<xform <<" * " << obj.distmap->mat<<" = ";
        obj.distmap->mat = xform * obj.distmap->mat;
        std::cout << obj.distmap->mat<<std::endl;
    }
    {
        BBox bb = obj.calcBBox();
        std::cout << "Mesh bbox = "<<bb<<std::endl;
        std::cout << "Mesh center and radius = "<<(bb.a+bb.b)*0.5<<"  "<<(bb.b-bb.a)*0.5 << std::endl;
    }
    if (mergedist >= 0.0f)
    {
        std::cout << "Merging vertices closer than " << mergedist <<std::endl;
        obj.mergeVertices(mergedist);
    }
    if (flip)
    {
        std::cout << "Flipping mesh..."<<std::endl;
        obj.flipAll();
        //obj.calcFlip();
    }
    if (closemesh || closedist != 0.0f)
    {
        bool closed = obj.isClosed();
        std::cout << "Mesh is "<<(closed?"":"NOT ")<<"closed."<<std::endl;
        if (!closed)
        {
            obj.calcFlip();
            std::cout << "Closing mesh..."<<std::endl;
            if (closedist != 0.0f)
            {
                obj.closeDist(closedist);
            }
            else //if (closemesh)
            {
                obj.close();
            }
            std::cout << "Mesh is "<<(obj.isClosed()?"":"NOT ")<<"closed."<<std::endl;

        }
    }

    if (dilate != 0.0f)
        obj.dilate(dilate);

    if (tesselate > 0 || sphere)
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
        BBox bb = obj.calcBBox();
        float bsize = (border<0 ? -border : bb.size()*border);
        Vec3f bbsize = bb.b-bb.a;
        bbsize[0] += 2*bsize;
        bbsize[1] += 2*bsize;
        bbsize[2] += 2*bsize;
        if (vsize > 0)
        {
            rx = (int)ceilf(bbsize[0]/vsize);
            ry = (int)ceilf(bbsize[1]/vsize);
            rz = (int)ceilf(bbsize[2]/vsize);
        }
        std::cout << "Computing "<<rx<<'x'<<ry<<'x'<<rz<<" DistMap..."<<std::endl;
        obj.calcDistMap(rx,ry,rz,bsize);
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
