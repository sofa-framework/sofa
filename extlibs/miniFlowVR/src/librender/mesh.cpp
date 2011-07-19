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
#include <flowvr/render/mesh.h>
#ifndef MINI_FLOWVR
#include <flowvr/render/chunkwriter.h>
#endif
#include <ftl/type.h>

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <map>
#include <set>

namespace flowvr
{

namespace render
{

Mesh::Mesh()
: attrib(0), distmap(NULL), voxel(NULL)
{
}

Mesh::~Mesh()
{
  if(distmap!=NULL) free(distmap);
  if(voxel!=NULL) free(voxel);
}

void Mesh::operator=(const Mesh& mesh)
{
  if (&mesh == this) return;
  clear();
  attrib = mesh.attrib;
  points_p = mesh.points_p;
  points_t = mesh.points_t;
  points_n = mesh.points_n;
  points_c = mesh.points_c;
  points_v = mesh.points_v;
  points_g = mesh.points_g;
  groups_p0 = mesh.groups_p0;
  faces_p = mesh.faces_p;
  faces_n = mesh.faces_n;
  faces_u = mesh.faces_u;
  faces_v = mesh.faces_v;
  edges = mesh.edges;
  if (mesh.distmap!=NULL)
  {
    distmap = (DistMap*) malloc(mesh.distmap->size());
    memcpy(distmap,mesh.distmap,mesh.distmap->size());
  }
  if (mesh.voxel!=NULL)
  {
    voxel = (Voxel*) malloc(mesh.voxel->size());
    memcpy(voxel,mesh.voxel,mesh.voxel->size());
  }
}

void Mesh::clear()
{
  attrib=0;
  points_p.clear();
  points_t.clear();
  points_n.clear();
  points_c.clear();
  points_v.clear();
  points_g.clear();
  groups_p0.clear();
  faces_p.clear();
  faces_n.clear();
  faces_u.clear();
  faces_v.clear();
  edges.clear();
  if(distmap!=NULL) { free(distmap); distmap = NULL; }
  if(voxel!=NULL) { free(voxel); voxel = NULL; }
}

void Mesh::init(int nbp, int nbf, int attribs)
{
  clear();
  attrib=attribs;
  if (getAttrib(MESH_POINTS_POSITION))
  {
    points_p.resize(nbp);
    points_n.resize(nbp);
    if (getAttrib(MESH_POINTS_TEXCOORD))
      points_t.resize(nbp);
    if (getAttrib(MESH_POINTS_COLOR))
      points_c.resize(nbp);
    if (getAttrib(MESH_POINTS_VALUE))
      points_v.resize(nbp);
    if (getAttrib(MESH_POINTS_GROUP))
    {
      points_g.resize(nbp);
      groups_p0.resize(1);
    }
  }
  if (getAttrib(MESH_FACES))
  {
    faces_p.resize(nbf);
    faces_n.resize(nbf);
    faces_u.resize(nbf);
    faces_v.resize(nbf);
  }
}

#ifdef WIN32
#define strcasecmp stricmp
#endif

bool Mesh::load(const char* filename)
{
  const char* anchor=strrchr(filename,'#');
  std::string fname(filename, (anchor?anchor : filename+strlen(filename)));
#ifndef MINI_FLOWVR
  flowvr::render::ChunkRenderWriter::path.findFile(fname);
#endif
  filename = fname.c_str();
  if (anchor) ++anchor; // skip '#'

  const char* ext=filename + strlen(filename); //strrchr(filename,'.');
  while (ext > filename+1 && *ext!='.' && *ext != '/' && *ext != '\\')
      --ext;
  if (*ext=='.')
  {
    if (!strcasecmp(ext,".obj"))
      return loadObj(filename,anchor);
    if (!strcasecmp(ext,".off"))
      return loadOff(filename);
    if (!strcasecmp(ext,".lwo"))
      return loadLwo(filename);
    if (!strcasecmp(ext,".vtk"))
      return loadVtk(filename);
  }
  return loadMesh(filename);
}

bool Mesh::save(const char* filename) const
{
  const char* ext=strrchr(filename,'.');
  if (ext!=NULL)
  {
    if (!strcasecmp(ext,".obj"))
      return saveObj(filename);
    if (!strcasecmp(ext,".off"))
      return saveOff(filename);
    if (!strcasecmp(ext,".stl"))
      return saveStl(filename);
    if (!strcasecmp(ext,".lwo"))
      return saveLwo(filename);
    if (!strcasecmp(ext,".vtk"))
      return saveVtk(filename);
  }
  return saveMesh(filename);
}

void Mesh::calcNormals(bool force)
{
  for (int i=0;i<nbf();i++)
  {
    Vec3f a = PP(getFP(i)[1])-PP(getFP(i)[0]);
    Vec3f b = PP(getFP(i)[2])-PP(getFP(i)[0]);
    FN(i) = cross(a,b);
    FN(i).normalize();
    
    // also compute barycentric coordinate vectors
    Vec3f c = cross(getFN(i),b);
    FU(i) = c*(1/dot(a,c));
    c = cross(getFN(i),a);
    FV(i) = c*(1/dot(b,c));
  }
  if (!getAttrib(MESH_POINTS_NORMAL) || force)
  {
    for (int i=0;i<nbf();i++)
    {
      PN(getFP(i)[0]).clear();
      PN(getFP(i)[1]).clear();
      PN(getFP(i)[2]).clear();
    }
    for (int i=0;i<nbf();i++)
    {
      PN(getFP(i)[0]) += getFN(i);
      PN(getFP(i)[1]) += getFN(i);
      PN(getFP(i)[2]) += getFN(i);
    }
    
    if (!getAttrib(MESH_POINTS_GROUP))
    {
      for (int i=0;i<nbp();i++)
	PN(i).normalize();
    }
    else
    {
      for (int g=0;g<nbg();g++)
      {
	int p0 = getGP0(g); if (p0<0) p0 = -p0;
	int p1 = (g<nbg()?getGP0(g+1):nbp()); if (p1<0) p1 = -p1;
	Vec3f n;
	for (int p=p0; p<p1; p++)
	  n+=getPN(p);
	n.normalize();

	for (int p=p0; p<p1; p++)
	  PN(p)=n;
      }
    }
  }
  setAttrib(MESH_POINTS_NORMAL, true);
  std::cout << "Normals computed."<<std::endl;
}

void Mesh::clearEdges()
{
  edges.resize(nbp());
  for (int i=0;i<nbp();i++)
    edges[i].clear();
}

bool Mesh::addEdgeFace(int p0, int p1, int f)
{
  //std::cerr << "addEdgeFace("<<p0<<","<<p1<<","<<f<<")" << std::endl;
  if (getAttrib(MESH_POINTS_GROUP))
  {
    p0 = getGP0(getPG(p0));
    p1 = getGP0(getPG(p1));
  }
  if ((int)edges.size() <= p0 || (int)edges.size() < p1)
    edges.resize(nbp());
  //std::cerr << "getEdgeFace("<<p0<<","<<p1<<")" << std::endl;
  int f2 = getEdgeFace(p0,p1);
  if (f2>=0) { std::cerr << "Overlapping faces "<<f2<<" and "<<f<<std::endl; return false; }
  //std::cerr << "edges("<<p0<<","<<p1<<")" << std::endl;
  if (p0<p1)
    edges[p0][p1].f1=f;
  else
    edges[p1][p0].f2=f;
  //std::cerr << "<addEdgeFace("<<p0<<","<<p1<<","<<f<<")" << std::endl;
  return true;
}

int Mesh::getEdgeFace(int p0, int p1)
{
  //std::cerr << "getEdgeFace("<<p0<<","<<p1<<")" << std::endl;
  if (getAttrib(MESH_POINTS_GROUP))
  {
    p0 = getGP0(getPG(p0));
    p1 = getGP0(getPG(p1));
  }
  int r;
  if (p0<p1)
    r = edges[p0][p1].f1;
  else
    r = edges[p1][p0].f2;
  //std::cerr << "<getEdgeFace("<<p0<<","<<p1<<")="<< r << std::endl;
  return r;
}

void Mesh::calcEdges()
{
  clearEdges();
  for (int i=0;i<nbf();i++)
  {
    //std::cout << i << std::endl;
    addEdgeFace(getFP(i)[0],getFP(i)[1],i);
    addEdgeFace(getFP(i)[1],getFP(i)[2],i);
    addEdgeFace(getFP(i)[2],getFP(i)[0],i);
  }
}

void Mesh::flipAll()
{
  std::cerr << "Flipping ALL faces"<<std::endl;
  for (int i=0;i<nbf();i++)
  {
    int p = getFP(i)[1];
    FP(i)[1] = getFP(i)[2];
    FP(i)[2] = p;
    FN(i) = -getFN(i);
  }
  for (int i=0;i<nbp();i++)
  {
    PN(i) = -getPN(i);
  }
  calcNormals();
  calcEdges();
}

void Mesh::calcFlip()
{
  bool recalcNormals = false;
  clearEdges();
  std::vector<bool> fdone;
  int ntodo = nbf();
  fdone.resize(nbf());
  for (int i=0;i<nbf();i++)
    fdone[i] = false;
  // First we add only the first face
  addEdgeFace(getFP(0)[0],getFP(0)[1],0);
  addEdgeFace(getFP(0)[1],getFP(0)[2],0);
  addEdgeFace(getFP(0)[2],getFP(0)[0],0);
  fdone[0] = true; --ntodo;
  // Then we add faces touching at least one edge
  while (ntodo>0)
  {
    int ndone = 0;
    for (int i=0;i<nbf();i++)
    {
      if (fdone[i]) continue;
      bool pos = false;
      bool neg = false;
      if (getEdgeFace(getFP(i)[1],getFP(i)[0])>=0) pos = true;
      if (getEdgeFace(getFP(i)[2],getFP(i)[1])>=0) pos = true;
      if (getEdgeFace(getFP(i)[0],getFP(i)[2])>=0) pos = true;
      if (getEdgeFace(getFP(i)[0],getFP(i)[1])>=0) neg = true;
      if (getEdgeFace(getFP(i)[1],getFP(i)[2])>=0) neg = true;
      if (getEdgeFace(getFP(i)[2],getFP(i)[0])>=0) neg = true;
      if (!pos && !neg) continue; // untouching face
      if (pos && neg)
      {
	std::cerr << "ERROR: Invalid face "<<i<<std::endl;
      }
      else
      {
	if (neg)
	{
	  std::cerr << "Flipping face "<<i<<std::endl;
	  int p = getFP(i)[1];
	  FP(i)[1] = getFP(i)[2];
	  FP(i)[2] = p;
	  recalcNormals = true;
	}
	// adding face
	addEdgeFace(getFP(i)[0],getFP(i)[1],i);
	addEdgeFace(getFP(i)[1],getFP(i)[2],i);
	addEdgeFace(getFP(i)[2],getFP(i)[0],i);
      }
      fdone[i] = true;
      ++ndone;
    }
    if (!ndone) break;
    ntodo -= ndone;
  }
  if (recalcNormals)
    calcNormals(true);
  // Then we check the normal of the highest point
  int pmax = 0;
  for (int i=1;i<nbp();i++)
    if (getPP(i)[2] > getPP(pmax)[2])
      pmax = i;
  if (getPN(pmax)[2]<0)
  { // Flip all faces
    flipAll();
  }
}

bool Mesh::isClosed()
{
  if (edges.empty()) calcEdges();
  int nopen = 0;
  for (unsigned int i=0;i<edges.size();i++)
  {
    for (std::map<int,Edge>::iterator it = edges[i].begin(); it != edges[i].end(); it++)
    {
      if ((it->second.f1==-1) ^ (it->second.f2==-1))
      {
	if (it->second.f1==-1) std::cerr << "Open edge "<<it->first<<"-"<<i<<" from face "<<it->second.f2<<std::endl;
	else                   std::cerr << "Open edge "<<i<<"-"<<it->first<<" from face "<<it->second.f1<<std::endl;
	++nopen;
      }
    }
  }
  return nopen==0;
}

void Mesh::closeLoop(const std::vector<int>& loop, float dist)
{
  if (loop.size()<3) return;
  Vertex v;
  for (unsigned int j=0;j<loop.size();j++)
  {
    v+=getP(loop[j]);
  }
  v.mean(loop.size());
  int p = addP(v);

  float maxdist = 0.0f;
  if (dist > 0.0f)
  for (unsigned int j=0;j<loop.size();j++)
  {
      float d = (getPP(loop[j])-v.p).norm();
      if (d > maxdist) maxdist = d;
  }

  if (dist <= 0.0f || maxdist < dist)
  {
    if (dist > 0.0f)
      std::cout << "# open " << loop.size()<<"-sided loop : max radius " << maxdist <<std::endl;
    Vec3i f;
    f[0] = p;
    for (unsigned int j=0;j<loop.size();j++)
    {
      f[1] = loop[j];
      f[2] = loop[(j+1)%loop.size()];
      if (dist == 0.0f)
	  std::cout << "f "<<f[0]+1<<" "<<f[1]+1<<" "<<f[2]+1<<std::endl;
      addF(f);
    }
  }
  else
  {
    int nsteps = (int)ceilf(maxdist/dist);
    std::cout << "# open " << loop.size()<<"-sided loop : max radius " << maxdist << " -> using " << nsteps << " steps."<<std::endl;
    std::vector<int> loopOut, loopIn;
    Vec3i f;
    //loopOut = loop;
    loopOut.resize(loop.size());
    for (unsigned int j=0;j<loop.size();j++) loopOut[j] = loop[j];
    loopIn.resize(loop.size());
    for (int s=0;s<nsteps-1;++s)
    {
      float fact = ((float)(nsteps-1-s))/((float)nsteps);
      for (unsigned int j=0;j<loop.size();j++)
      {
	Vertex v2;
	if (s&1)
	    v2.lerp(v,1-fact,getP(loop[j]),fact);
	else
	    v2.lerp(v,1-fact,getP(loop[j]),0.5f*fact,getP(loop[(j+1)%loop.size()]),0.5f*fact);
	loopIn[j] = addP(v2);
      }
      for (unsigned int j=0;j<loop.size();j++)
      {
	f[0] = loopIn[j];
	f[1] = loopOut[j];
	f[2] = loopOut[(j+1)%loop.size()];
	//std::cout << "f "<<f[0]+1<<" "<<f[1]+1<<" "<<f[2]+1<<std::endl;
	addF(f);
	f[0] = loopIn[j];
	f[1] = loopOut[(j+1)%loop.size()];
	f[2] = loopIn[(j+1)%loop.size()];
	//std::cout << "f "<<f[0]+1<<" "<<f[1]+1<<" "<<f[2]+1<<std::endl;
	addF(f);
      }
      //loopOut = loopIn;
      for (unsigned int j=0;j<loop.size();j++) loopOut[j] = loopIn[j];
    }
    f[0] = p;
    for (unsigned int j=0;j<loop.size();j++)
    {
      f[1] = loopOut[j];
      f[2] = loopOut[(j+1)%loop.size()];
      //std::cout << "f "<<f[0]+1<<" "<<f[1]+1<<" "<<f[2]+1<<std::endl;
      addF(f);
    }
  }
}

void Mesh::close()
{
  closeDist(0.0f);
}

void Mesh::closeDist(float dist)
{
  // First detect flipped faces
  //calcFlip();
  //if (edges.empty())
  calcEdges();
  std::vector<int> next;
  next.resize(nbp());
  for (int i=0;i<nbp();i++)
    next[i] = -1;
  int nopen = 0;
  for (int i=0;i<nbp();i++)
  {
    for (std::map<int,Edge>::iterator it = edges[i].begin(); it != edges[i].end(); it++)
    {
      if ((it->second.f1==-1) ^ (it->second.f2==-1))
      {
	if (it->second.f2==-1) { next[it->first] = i; }
	else                   { next[i] = it->first; }
	++nopen;
      }
    }
  }
  int nloop = 0;
  if (nopen>0)
  { // close all loops
    std::vector<int> loop;
    for (unsigned int i=0;i<next.size();i++)
    {
      int n = next[i];
      if (n!=-1)
      {
	loop.push_back(i);
	while (n>=0 && n!=(int)i && loop.size()<(unsigned)nbp())
	{
	  loop.push_back(n);
	  n = next[n];
	}
	if (n==(int)i)
	{ // closed loop
	  closeLoop(loop, dist);
	  ++nloop;
	  for (unsigned int j=0;j<loop.size();j++)
	    next[loop[j]] = -1;
	}
	else
	{
	    std::cerr << "Open loop from "<<i<<" to "<<n<<std::endl;
	}
	loop.clear();
      }
    }
    if (nloop>0)
	std::cout << "Closed "<<nloop<<" loops."<<std::endl;
  }
}

BBox Mesh::calcBBox()
{
  bb.clear();
  for (int i=0;i<nbp();i++)
  {
    bb+=getPP(i);
  }
  return bb;
}

/// Compute bounding box of a submesh
BBox Mesh::calcBBox(int f0, int nbf)
{
  BBox bb;
  bb.clear();
  for (int i=0;i<nbf;i++)
  {
    bb+=getPP(getFP(f0+i)[0]);
    bb+=getPP(getFP(f0+i)[1]);
    bb+=getPP(getFP(f0+i)[2]);
  }
  return bb;
}


void Mesh::translate(Vec3f d)
{
  for (int i=0;i<nbp();i++)
    PP(i)+=d;
  bb.a+=d;
  bb.b+=d;
  if (distmap!=NULL)
  {
    distmap->mat[0][3]+=d[0];
    distmap->mat[1][3]+=d[1];
    distmap->mat[2][3]+=d[2];
  }
  if (voxel!=NULL)
  {
    voxel->mat[0][3]+=d[0];
    voxel->mat[1][3]+=d[1];
    voxel->mat[2][3]+=d[2];
  }
}

void Mesh::dilate(float dist)
{
  calcNormals();
  for (int i=0;i<nbp();i++)
    PP(i)+=getPN(i)*dist;
  if (distmap!=NULL)
  {
    int n = distmap->nval();
    for (int i=0;i<n;i++)
      distmap->data[i]-=dist;
  }
  calcNormals();
}

/// Return true if some materials requires tangent vectors
bool Mesh::needTangents() const
{
  if (getAttrib(MESH_POINTS_TEXCOORD))
  {
    // look for referenced materials with a bump texture
    for (unsigned int i=0;i<mat_groups.size();i++)
    {
      if (mat_groups[i].mat && !mat_groups[i].mat->map_bump.empty())
      {
	return true;
      }
    }
  }
  return false;
}

/// Compute the tangent and co-tangent at each point
void Mesh::calcTangents(std::vector<Vec3f>& tangent1, std::vector<Vec3f>& tangent2) const
{
  tangent1.resize(nbp());
  tangent2.resize(nbp());
    
  // see http://www.terathon.com/code/tangent.php
  
  for (int i=0;i<nbf();i++)
  {
    int i1 = getFP(i)[0];
    int i2 = getFP(i)[1];
    int i3 = getFP(i)[2];
    
    const Vec3f& v1 = getPP(i1);
    const Vec3f& v2 = getPP(i2);
    const Vec3f& v3 = getPP(i3);
    
    const Vec2f& w1 = getPT(i1);
    const Vec2f& w2 = getPT(i2);
    const Vec2f& w3 = getPT(i3);
    
    float x1 = v2[0] - v1[0];
    float x2 = v3[0] - v1[0];
    float y1 = v2[1] - v1[1];
    float y2 = v3[1] - v1[1];
    float z1 = v2[2] - v1[2];
    float z2 = v3[2] - v1[2];
    
    float s1 = w2[0] - w1[0];
    float s2 = w3[0] - w1[0];
    float t1 = w2[1] - w1[1];
    float t2 = w3[1] - w1[1];
    
    float r = 1.0f / (s1 * t2 - s2 * t1);
    Vec3f sdir((t2 * x1 - t1 * x2) * r, (t2 * y1 - t1 * y2) * r, (t2 * z1 - t1 * z2) * r);
    Vec3f tdir((s1 * x2 - s2 * x1) * r, (s1 * y2 - s2 * y1) * r, (s1 * z2 - s2 * z1) * r);
    
    tangent1[i1] += sdir;
    tangent1[i2] += sdir;
    tangent1[i3] += sdir;
    
    tangent2[i1] += tdir;
    tangent2[i2] += tdir;
    tangent2[i3] += tdir;
  }
  
  for (unsigned int i=0;i<tangent1.size();i++)
  {
    const Vec3f& normal = getPN(i);
    Vec3f& t1 = tangent1[i];
    Vec3f& t2 = tangent2[i];
    // Gram-Schmidt orthogonalize
    t1 -= normal * ( normal * t1 );
    t1.normalize();
    t2 -= normal * ( normal * t1 );
    t2.normalize();
  }
}
      /// Compute the tangent at each point, plus a fourth coordinate indicating the direction of the co-tangent.
      /// See http://www.terathon.com/code/tangent.php
void Mesh::calcTangents(std::vector<Vec4f>& tangent) const
{
  tangent.resize(nbp());
  std::vector<Vec3f> tangent1;
  std::vector<Vec3f> tangent2;
  calcTangents(tangent1, tangent2);
  for (unsigned int i=0;i<tangent.size();i++)
  {
    const Vec3f& normal = getPN(i);
    Vec3f t1 = tangent1[i];
    tangent[i] = t1;
    // Calculate handedness
    tangent[i][3] = ((cross(normal,t1) * tangent2[i]) < 0.0f) ? -1.0f : 1.0f;
  }
}

#ifndef MINI_FLOWVR
void Mesh::writeMesh(ChunkRenderWriter* scene, ID idIB, ID idVB, int gen) const
{
  bool useTangent = needTangents();

  std::vector<Vec4f> tangent;
  if (useTangent)
    calcTangents(tangent);

  int ndata = 0;
  int ip = ndata++;
  int in = ndata++;
  int it = (getAttrib(MESH_POINTS_TEXCOORD)?ndata++:-1);
  int itan = (useTangent?ndata++:-1);
  int ic = (getAttrib(MESH_POINTS_COLOR)?ndata++:-1);
  int iv = (getAttrib(MESH_POINTS_VALUE)?ndata++:-1);

  int* dataType = new int[ndata];
  if (ip>=0) dataType[ip] = Type::Vec3f;
  if (in>=0) dataType[in] = Type::Vec3f;
  if (it>=0) dataType[it] = Type::Vec2f;
  if (itan>=0) dataType[itan] = Type::Vec4f;
  if (ic>=0) dataType[ic] = Type::Vec4b;
  if (iv>=0) dataType[iv] = Type::Float;
  {
    ChunkVertexBuffer* vb = scene->addVertexBuffer(idVB, nbp(), ndata, dataType, bb);
    vb->gen = gen;
    char* dest = (char*) vb->data();
    for (int i=0;i<nbp();i++)
    {
      if (ip>=0) { *(Vec3f*)dest = getPP(i); dest+=sizeof(Vec3f); }
      if (in>=0) { *(Vec3f*)dest = getPN(i); dest+=sizeof(Vec3f); }
      if (it>=0) { *(Vec2f*)dest = getPT(i); dest+=sizeof(Vec2f); }
      if (itan>=0) { *(Vec4f*)dest = tangent[i]; dest+=sizeof(Vec4f); }
      if (ic>=0) { *(Vec4b*)dest = getPC(i); dest+=sizeof(Vec4b); }
      if (iv>=0) { *(float*)dest = getPV(i); dest+=sizeof(float); }
    }
  }

  if (nbp()<=256)
  {
    ChunkIndexBuffer* ib = scene->addIndexBuffer(idIB, nbf()*3, Type::Byte, ChunkIndexBuffer::Triangle);
    ib->gen = gen;
    Vec<3,char>* f = (Vec<3,char>*)ib->data();
    for (int i=0;i<nbf();i++)
    {
      f[i][0] = getFP(i)[0];
      f[i][1] = getFP(i)[1];
      f[i][2] = getFP(i)[2];
    }
  }
  else if (nbp()<=65536)
  {
    ChunkIndexBuffer* ib = scene->addIndexBuffer(idIB, nbf()*3, Type::Short, ChunkIndexBuffer::Triangle);
    ib->gen = gen;
    Vec<3,short> *f = (Vec<3,short>*)ib->data();
    for (int i=0;i<nbf();i++)
    {
      f[i][0] = getFP(i)[0];
      f[i][1] = getFP(i)[1];
      f[i][2] = getFP(i)[2];
    }
  }
  else
  {
    ChunkIndexBuffer* ib = scene->addIndexBuffer(idIB, nbf()*3, Type::Int, ChunkIndexBuffer::Triangle);
    ib->gen = gen;
    Vec<3,int> *f = (Vec<3,int>*)ib->data();
    for (int i=0;i<nbf();i++)
    {
      f[i][0] = getFP(i)[0];
      f[i][1] = getFP(i)[1];
      f[i][2] = getFP(i)[2];
    }
  }
}

void Mesh::writeParams(ChunkRenderWriter* scene, ID idP, ID idIB, ID idVB) const
{
  bool useTangent = needTangents();
  int ndata = 0;
  int ip = ndata++;
  int in = ndata++;
  int it = (getAttrib(MESH_POINTS_TEXCOORD)?ndata++:-1);
  int itan = (useTangent?ndata++:-1);
  int ic = (getAttrib(MESH_POINTS_COLOR)?ndata++:-1);
  int iv = (getAttrib(MESH_POINTS_VALUE)?ndata++:-1);

  scene->addParamID(idP, ChunkPrimParam::VBUFFER_ID,"position",idVB);
  scene->addParam(idP, ChunkPrimParam::VBUFFER_NUMDATA,"position",ip);
  scene->addParamID(idP, ChunkPrimParam::VBUFFER_ID,"normal",idVB);
  scene->addParam(idP, ChunkPrimParam::VBUFFER_NUMDATA,"normal",in);
  if (it>=0)
  {
    scene->addParamID(idP, ChunkPrimParam::VBUFFER_ID,"texcoord0",idVB);
    scene->addParam(idP, ChunkPrimParam::VBUFFER_NUMDATA,"texcoord0",it);
  }
  if (itan>=0)
  {
    scene->addParamID(idP, ChunkPrimParam::VBUFFER_ID,"tangent",idVB);
    scene->addParam(idP, ChunkPrimParam::VBUFFER_NUMDATA,"tangent",itan);
  }
  if (ic>=0)
  {
    scene->addParamID(idP, ChunkPrimParam::VBUFFER_ID,"color",idVB);
    scene->addParam(idP, ChunkPrimParam::VBUFFER_NUMDATA,"color",ic);
  }
  if (iv>=0)
  {
    scene->addParamID(idP, ChunkPrimParam::VBUFFER_ID,"value",idVB);
    scene->addParam(idP, ChunkPrimParam::VBUFFER_NUMDATA,"value",iv);
  }
  scene->addParamID(idP, ChunkPrimParam::IBUFFER_ID,"",idIB);
}
#endif //MINI_FLOWVR

/// Optimize mesh by merging identical groups and reordering faces
void Mesh::optimize()
{
    int ng = mat_groups.size();
    if (ng <= 1) return; // no need for optimizations

    // create a material -> group map
    std::map< Material*,std::vector<int> > matmap;

    for (unsigned int i=0;i<mat_groups.size();i++)
	matmap[mat_groups[i].mat].push_back(i);

    int newsize = matmap.size();
    if (newsize == ng) return; // no need for optimizations
    std::cout << "Optimizing mesh: merging "<<ng<<" groups to "<<newsize<<" groups."<<std::endl;

    // create the new groups from these materials
    std::vector<MaterialGroup> new_mat_groups;
    std::vector<Vec3i> new_faces_p;

    new_mat_groups.reserve(matmap.size());
    new_faces_p.reserve(faces_p.size());
    for (std::map<Material*,std::vector<int> >::iterator it = matmap.begin(); it!=matmap.end(); ++it)
    {
	MaterialGroup m;
	m.f0 = new_faces_p.size();
	m.mat = it->first;
	if (m.mat)
	    m.matname = m.mat->matname;
	const std::vector<int>& srcs = it->second;
	std::set<std::string> gnames;
	for (unsigned int i=0;i<srcs.size();i++)
	{
	    const MaterialGroup& src = mat_groups[srcs[i]];
	    gnames.insert(src.gname);
	    new_faces_p.insert(new_faces_p.end(), faces_p.begin()+src.f0, faces_p.begin()+src.f0+src.nbf);
	}
	m.nbf = new_faces_p.size() - m.f0;
	for (std::set<std::string>::iterator it = gnames.begin(); it != gnames.end(); ++it)
	{
	    if (!m.gname.empty()) m.gname += ' ';
	    m.gname += *it;
	}
	if (m.nbf > 0)
	    new_mat_groups.push_back(m);
    }
    faces_p.swap(new_faces_p);
    mat_groups.swap(new_mat_groups);
}


/// Merge vertices closer than the given distance
void Mesh::mergeVertices(float dist)
{
    std::vector<int> old2new_v;
    std::vector<Vec3f> new_pos;
    std::vector<Vec3f> new_pos_sum;
    std::vector<int> new_pos_count;
    std::map<Vec3f,int> pos2new;
    
    bool group    = getAttrib(MESH_POINTS_GROUP);
    int nbp0 = (group) ? nbg() : nbp();
    int new_nbp = 0;
    old2new_v.resize(nbp0);
    new_pos.reserve(nbp0);
    new_pos_sum.reserve(nbp0);
    new_pos_count.reserve(nbp0);
    for (int p0 = 0; p0 < nbp0; ++p0)
    {
        Vec3f pos = (group) ? getPP(getGP0(p0)) : getPP(p0);
        int new_p = -1;
        float new_d2 = dist*dist;
        std::pair<std::map<Vec3f,int>::iterator,bool> it = pos2new.insert(std::pair<Vec3f,int>(pos,new_nbp));
        if (!it.second)
        {
            new_p = it.first->second;
            new_d2 = 0;
        }
        else if (dist > 0.0f)
        {
            for (int p1 = 0; p1 < new_nbp; ++p1)
            {
                float d2 = (new_pos[p1] - pos).norm2();
                if (d2 <= new_d2)
                {
                    new_p = p1;
                    new_d2 = d2;
                }
            }
        }
        if (new_p != -1)
        {
            //std::cout << "vertex " << p0 << " merged, dist=" << sqrtf(new_d2) << std::endl;
            new_pos_sum[new_p] += pos;
            new_pos_count[new_p] += 1;
            if (new_d2 > 0.0f)
                new_pos[new_p] = new_pos_sum[new_p] / (float)new_pos_count[new_p];
        }
        else
        {
            new_p = new_nbp;
            ++new_nbp;
            new_pos_sum.push_back(pos);
            new_pos_count.push_back(1);
            new_pos.push_back(pos);
        }
        old2new_v[p0] = new_p;
    }
    if (new_nbp == nbp0)
    {
        std::cout << "No vertices merged with distance = " << dist << std::endl;
        return;
    }
    std::cout << nbp0 - new_nbp << "/"<<nbp0<<" vertices merged with distance = " << dist << std::endl;
    if (!group)
    {
        points_p.resize(new_nbp);
        for (int p = 0; p < new_nbp; ++p)
            points_p[p] = new_pos[p];

        if (getAttrib(MESH_POINTS_TEXCOORD))
        { // compute mean texcoords
            std::vector<Vec2f> sums;
            sums.resize(new_nbp);
            for (int p0 = 0; p0 < nbp0; ++p0)
                sums[old2new_v[p0]] += getPT(p0);
            points_t.resize(new_nbp);
            for (int p = 0; p < new_nbp; ++p)
                points_t[p] = sums[p] / (float)new_pos_count[p];
        }

        if (getAttrib(MESH_POINTS_NORMAL))
        { // compute mean normals
            std::vector<Vec3f> sums;
            sums.resize(new_nbp);
            for (int p0 = 0; p0 < nbp0; ++p0)
                sums[old2new_v[p0]] += getPN(p0);
            points_n.resize(new_nbp);
            for (int p = 0; p < new_nbp; ++p)
            {
                points_n[p] = sums[p];
                points_n[p].normalize();
            }
        }

        if (getAttrib(MESH_POINTS_COLOR))
        { // compute mean colors
            std::vector<Vec4f> sums;
            sums.resize(new_nbp);
            for (int p0 = 0; p0 < nbp0; ++p0)
                sums[old2new_v[p0]] += Vec4f(getPC(p0));
            points_c.resize(new_nbp);
            for (int p = 0; p < new_nbp; ++p)
            {
                Vec4f c = sums[p] / (float)new_pos_count[p];
                Vec4b cb;
                for (int i=0;i<4;++i)
                {
                    int v = (int)(c[i]+0.5f);
                    if (v<0) v = 0; else if (v>255) v = 255;
                    cb[i] = v;
                }
                points_c[p] = cb;
            }
        }

        if (getAttrib(MESH_POINTS_VALUE))
        { // compute mean values
            std::vector<float> sums;
            sums.resize(new_nbp);
            for (int p0 = 0; p0 < nbp0; ++p0)
                sums[old2new_v[p0]] += getPV(p0);
            points_v.resize(new_nbp);
            for (int p = 0; p < new_nbp; ++p)
                points_v[p] = sums[p] / new_pos_count[p];
        }

        for (int f=0;f<nbf();++f)
        {
            Vec3i fp = getFP(f);
            for (int i = 0; i < 3; ++i)
                fp[i] = old2new_v[fp[i]];
            if (fp[0] == fp[1] || fp[0] == fp[2] || fp[1] == fp[2])
                std::cerr << "Warning: face " << f << " is now degenerated." << std::endl;
            FP(f) = fp;
        }
    }
    else
    {
        std::cerr << "Vertices merging not yet implemented with groups." << std::endl;
    }
}

} // namespace render

} // namespace flowvr
