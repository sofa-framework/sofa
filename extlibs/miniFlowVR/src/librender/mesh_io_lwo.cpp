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
* File: ./src/librender/mesh.cpp                                  *
*                                                                 *
* Contacts:                                                       *
*                                                                 *
******************************************************************/
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <set>
#include <stack>

#include <flowvr/render/mesh.h>


namespace flowvr
{

namespace render
{

class MeshLwoWriter
{
public:
    std::vector<char> data;
    struct ChunkHeader
    {
        unsigned int id;
        unsigned int lengthPos;
        unsigned int lengthSize;
    };
    std::stack<ChunkHeader> chunkStack;

    template<class T>
    unsigned int writeSwap(T v, int pos = -1)
    {
        union {
            T t;
            char b[sizeof(T)];
        } tmp;
        tmp.t = v;
        unsigned int wpos = (pos < 0) ? (unsigned int)data.size() : (unsigned int)pos;
        if (wpos + sizeof(T) > data.size())
            data.resize(wpos+sizeof(T));
        for (unsigned int i=0;i<sizeof(T);++i)
            data[wpos+i] = tmp.b[sizeof(T)-1-i];
        return wpos;
    }

    unsigned int writeU1(unsigned char v, int pos = -1) { return writeSwap(v,pos); }
    unsigned int writeI1(char v, int pos = -1) { return writeSwap(v,pos); }
    unsigned int writeU2(unsigned short v, int pos = -1) { return writeSwap(v,pos); }
    unsigned int writeI2(short v, int pos = -1) { return writeSwap(v,pos); }
    unsigned int writeU4(unsigned int v, int pos = -1) { return writeSwap(v,pos); }
    unsigned int writeI4(int v, int pos = -1) { return writeSwap(v,pos); }
    unsigned int writeIndex(unsigned int v, int pos = -1)
    {
        if (v < 0xff00)
            return writeU2((unsigned short)v,pos);
        else
            return writeU4(v|0xff000000,pos);
    }
    unsigned int writeF4(float v, int pos = -1) { return writeSwap(v,pos); }
    unsigned int writeF4(double v, int pos = -1) { return writeSwap((float)v,pos); }
    unsigned int writeStr(const char* s, int pos = -1)
    {
        unsigned int l = 0;
        while (s && s[l]) ++l;
        unsigned int size = l;
        ++size; // add a least one zero
        if(size & 1) ++size; // and another one to align to words
        unsigned int wpos = (pos < 0) ? (unsigned int)data.size() : (unsigned int)pos;
        if (wpos + size > data.size())
            data.resize(wpos+size);
        for (unsigned int i=0;i<l;++i)
            data[wpos+i] = s[i];
        for (unsigned int i=l;i<size;++i)
            data[wpos+i] = (char)0;
        return wpos;
    }

    void openChunk(unsigned int id)
    {
        ChunkHeader chunk;
        chunk.id = id;
        writeU4(id);
        chunk.lengthPos = writeU4(0);
        chunk.lengthSize = 4;
        chunkStack.push(chunk);
    }

    void openSubChunk(unsigned int id)
    {
        ChunkHeader chunk;
        chunk.id = id;
        writeU4(id);
        chunk.lengthPos = writeU2(0);
        chunk.lengthSize = 2;
        chunkStack.push(chunk);
    }

    std::string id2str(unsigned int id)
    {
        union
        {
            unsigned int tmp;
            char str[5];
        };
        str[4] = '\0';
        tmp = id;
        return std::string(str,str+4);
    }

    void closeChunk(unsigned int id = 0)
    {
        if (chunkStack.empty())
        {
            std::cerr << "LWO: extra closeChunk ignored" << std::endl;
            return;
        }
        ChunkHeader chunk = chunkStack.top();
        chunkStack.pop();
        if (id && id != chunk.id)
        {
            std::cerr << "LWO: closeChunk(" << id2str(id) << ") while current chunk is " << id2str(chunk.id) << std::endl;
        }
        unsigned int length = data.size() - (chunk.lengthPos + chunk.lengthSize);
        switch(chunk.lengthSize)
        {
        case 2: writeU2((unsigned short)length, chunk.lengthPos); break;
        case 4: writeU4(length, chunk.lengthPos); break;
        default: std::cerr << "LWO: unknown chunk length size " << chunk.lengthSize << std::endl;
        }
        if (length&1)
            writeU1(0);
    }
    void closeSubChunk(unsigned int id = 0)
    {
        closeChunk(id);
    }
};

bool Mesh::loadLwo(const char* /*filename*/)
{
  return false;
}

static inline unsigned int ID(const char* str)
{
    return (((unsigned int)str[0])<<24)|(((unsigned int)str[1])<<16)|(((unsigned int)str[2])<<8)|(((unsigned int)str[3]));
}

bool Mesh::saveLwo(const char* filename) const
{
  FILE* fp = fopen(filename,"w+");
  if (fp==NULL) return false;
  std::cout<<"Writing LWO file "<<filename<<std::endl;


  //bool normal   = getAttrib(MESH_POINTS_NORMAL);
  bool texcoord = getAttrib(MESH_POINTS_TEXCOORD);
  bool group    = getAttrib(MESH_POINTS_GROUP);

  MeshLwoWriter out;

  std::vector<int> groups_pos;
  int nbpos = 0;

  out.openChunk(ID("FORM"));
  {
      out.writeU4(ID("LWO2"));
      
      out.openChunk(ID("TAGS"));
      {
          out.writeStr("Default");
      }
      out.closeChunk(ID("TAGS"));
      /*
      out.openChunk(ID("LAYR"));
      {
          out.writeU2(0); // number
          out.writeU2(0); // flags
          out.writeF4(0.0f); // pivot
          out.writeF4(0.0f);
          out.writeF4(0.0f);
          out.writeStr("Default"); // name
          //out.writeI2(-1); // parent
      }
      out.closeChunk(ID("LAYR"));
      */
      out.openChunk(ID("PNTS"));
      {
          //std::cout<<nbp()<<" points, "<<nbf()<<" faces"<<std::endl;
          //fprintf(fp,"#Face Count %d\n",nbf());
          if (group)
          {
              groups_pos.resize(nbg());
              for (int i=0;i<nbg();i++)
              {
                  int p = getGP0(i);
                  if (p >= 0)
                  { // this is not a subgroup
                      ++nbpos;
                  }
                  groups_pos[i] = nbpos-1;
              }
              fprintf(fp,"#Vertex Count %d\n",nbpos);
              for (int i=0;i<nbg();i++)
              {
                  int p = getGP0(i);
                  if (p >= 0)
                  {
                      out.writeF4(getPP(p)[0]);
                      out.writeF4(getPP(p)[1]);
                      out.writeF4(getPP(p)[2]);
                  }
              }
          }
          else
          {
              nbpos = nbp();
              for (int i=0;i<nbp();i++)
              {
                  out.writeF4(getPP(i)[0]);
                  out.writeF4(getPP(i)[1]);
                  out.writeF4(getPP(i)[2]);
              }
          }
      }
      out.closeChunk(ID("PNTS"));

      out.openChunk(ID("POLS"));
      {
          out.writeU4(ID("FACE"));

          for (int i=0;i<nbf();i++)
          {
              out.writeU2(3);
              for (int j=0;j<3;j++)
              {
                  int p = getFP(i)[j];
                  int pp=p; //, pn=p, pt=p;
                  if (group)
                  {
                      pp = groups_pos[getPG(p)];
                      //pn = getPG(p);
                  }
                  out.writeIndex(pp);
/*
                  if (normal && texcoord)
                      fprintf(fp," %d/%d/%d",pp+v0,pt+vt0,pn+vn0);
                  else if (normal)
                      fprintf(fp," %d//%d",pp+v0,pn+vn0);
                  else if (texcoord)
                      fprintf(fp," %d/%d",pp+v0,pt+vt0);
                  else
                      fprintf(fp," %d",pp+v0);
*/
              }
          }
      }
      out.closeChunk(ID("POLS"));

      if (texcoord)
      {
          out.openChunk(ID("VMAP"));
          {
              out.writeU4(ID("TXUV"));
              out.writeU2(2);
              out.writeStr("UV");
              for (int i=0;i<nbpos;i++)
              {
                  int p = i;
                  if (group) p = groups_pos[i];
                  out.writeIndex(i);
                  out.writeF4(getPT(p)[0]);
                  out.writeF4(getPT(p)[1]);
              }
          }
          out.closeChunk(ID("VMAP"));
          if (group)
          {
              out.openChunk(ID("VMAD"));
              {
                  out.writeU4(ID("TXUV"));
                  out.writeU2(2);
                  out.writeStr("UV");
                  for (int i=0;i<nbf();i++)
                  {
                      for (int j=0;j<3;j++)
                      {
                          int p = getFP(i)[j];
                          int pp=p,pt=p;
                          if (group)
                          {
                              pp = groups_pos[getPG(p)];
                              //pn = getPG(p);
                          }
                          if(pp != pt)
                          {
                              out.writeIndex(j);
                              out.writeIndex(i);
                              out.writeF4(getPT(pt)[0]);
                              out.writeF4(getPT(pt)[1]);
                          }
                      }
                  }
              }
              out.closeChunk(ID("VMAD"));
          }
      }

      out.openChunk(ID("PTAG"));
      {
          out.writeU4(ID("SURF"));

          for (int i=0;i<nbf();i++)
          {
              out.writeIndex(i);
              out.writeU2(0);
          }
      }
      out.closeChunk(ID("PTAG"));

      out.openChunk(ID("SURF"));
      {
          out.writeStr("Default"); //  name
          out.writeStr(""); // parent
          /*
          out.openSubChunk(ID("COLR"));
          {
              out.writeF4(1.0f);
              out.writeF4(1.0f);
              out.writeF4(1.0f);
              out.writeIndex(0);
          }
          out.closeSubChunk(ID("COLR"));
          out.openSubChunk(ID("DIFF"));
          {
              out.writeF4(1.0f);
              out.writeIndex(0);
          }
          out.closeSubChunk(ID("DIFF"));
          */
          out.openSubChunk(ID("SMAN"));
          {
              out.writeF4((float)M_PI);
          }
          out.closeSubChunk(ID("SMAN"));
      }
      out.closeChunk(ID("SURF"));
  }
  out.closeChunk(ID("FORM"));
/*

  if (normal)
  {
    if (group)
    {
      for (int i=0;i<nbg();i++)
      {
        int p = getGP0(i);
	if (p<0) p = -p; // subgroup
        fprintf(fp,"vn %f %f %f\n",getPN(p)[0],getPN(p)[1],getPN(p)[2]);
      }
    }
    else
    {
      for (int i=0;i<nbp();i++)
      {
	fprintf(fp,"vn %f %f %f\n",getPN(i)[0],getPN(i)[1],getPN(i)[2]);
      }
    }
  }
  if (texcoord)
  {
    for (int i=0;i<nbp();i++)
    {
      fprintf(fp,"vt %f %f\n",getPT(i)[0],getPT(i)[1]);
    }
  }
*/

  size_t n = fwrite(&(out.data[0]),out.data.size(),1,fp);

  fclose(fp);
  return (n == 1);
}

} // namespace render

} // namespace flowvr
