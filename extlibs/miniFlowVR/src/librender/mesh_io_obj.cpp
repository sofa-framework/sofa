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
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <set>

#include <flowvr/render/mesh.h>


namespace flowvr
{

namespace render
{

class MeshObjData
{
public:
  Mesh* mesh;
  std::vector<Vec3f> t_v;
  std::vector<Vec2f> t_vt;
  std::vector<Vec3f> t_vn;
  std::vector<Vec3i> t_fp;

  int v0;
  int merge;
  int smoothgroup;
  enum { SMOOTHGROUP_SHIFT = 20 };

  //std::vector<int> t_v_g; // group of points using this position


  MeshObjData(Mesh* m): mesh(m), v0(0), merge(0), smoothgroup(0) {}

  const char* skip(const char* c)
  {
    while (*c==' ' || *c=='\t') ++c;
    return c;
  }

  const char* endtoken(const char* c)
  {
    while (*c!=' ' && *c!='\t' && *c != '\0') ++c;
    return c;
  }

  void addMaterial(const Mesh::Material& m)
  {
    if (m.matname.empty()) return;
    Mesh::Material*& dest = mesh->mat_map[m.matname];
    if (dest) *dest = m;
    else dest = new Mesh::Material(m);
    std::cout << "Material "<<m.matname<<" added."<<std::endl;
  }

  void addMatGroup(const std::string& matname, const std::string& gname, int f0, int nbf)
  {
    if (nbf <= 0) return;

    Mesh::MaterialGroup matg;
    matg.matname = matname;
    matg.gname = gname;
    matg.f0 = f0;
    matg.nbf = nbf;
    std::cout << "Group "<<gname<<" material "<<matname<<" f0="<<matg.f0<<" nbf="<<matg.nbf<<std::endl;
    if (!matname.empty())
    {
      matg.mat = mesh->getM(matname);
      if (matg.mat)
      {
	std::cout << "Material "<<matname<<" found."<<std::endl;
	mesh->setAttrib(Mesh::MESH_MATERIALS,true);
      }
      else
      {
	  std::cerr << "ERROR: material "<<matname<<" not found."<<std::endl;
      }
    }
    mesh->mat_groups.push_back(matg);
  }

  Vec3f readColor(const char* c)
  {
    Vec3f res(1,1,1);
    if (*(c = skip(c)))
    {
      res.x() = (float)strtod(skip(c),(char**)&c);
      if (*(c = skip(c)))
      {
	res.y() = (float)strtod(skip(c),(char**)&c);
	if (*(c = skip(c)))
	{
	  res.z() = (float)strtod(skip(c),(char**)&c);
	}
	else
	  res.z() = res.y();
      }
      else
      {
	res.y() = res.x();
	res.z() = res.x();
      }
    }
    return res;
  }

  static std::string getParentDir(const std::string& filename)
  {
    std::string::size_type pos = filename.find_last_of("/\\");
    if (pos == std::string::npos)
      return ""; // no directory
    else
      return filename.substr(0,pos);
  }

  std::string getRelativeFile(std::string filename, const std::string& basefile)
  {
      std::cout << filename << std::endl;
    if (filename.empty() || filename[0] == '/') return filename;
    std::string base = getParentDir(basefile);
    // remove any ".."
    while ((filename.substr(0,3)=="../" || filename.substr(0,3)=="..\\") && !base.empty())
    {
      filename = filename.substr(3);
      base = getParentDir(base);
    }
    if (base.empty())
      return filename;
    else if (base[base.length()-1] == '/')
      return base + filename;
    else
      return base + "/" + filename;
  }

  bool loadmtl(const char* filename, const char* basefilename)
  {
    std::string fpath = getRelativeFile(filename, basefilename);

    FILE* fp = fopen(fpath.c_str(),"r");
    if (fp==NULL)
    {
      std::cerr << "ERROR: Loading MTL file "<<fpath<<" failed."<<std::endl;
      return false;
    }

    Mesh::Material mat;
    char line[5000];
    int l = 0;
    
    // Not Posix compatible
    //    while ((read = getline(&line, &len, fp)) != -1)
    line [sizeof(line)-1]='\0';
    while (fgets(line, sizeof(line), fp) != NULL)
    {
      ++l;
      if (line[0] == '#') continue;
      // remove EOL
      {
	int i = strlen(line)-1;
	while (i>=0 && (line[i] == '\n' || line[i] == '\r' || line[i] == ' ' || line[i] == '\t'))
	{
	  line[i] = '\0';
	  --i;
	}
      }
      if (!line[0]) continue; // empty line
      const char* c = line;
      const char* end = endtoken(c);
      std::string t(c,end);
      c = end;
      if (t == "newmtl")
      {
	addMaterial(mat);
	mat = Mesh::Material();
	mat.mtllib = filename;
	mat.matname = skip(c);
      }
      else if (t == "Ka")
      {
	mat.ambient = readColor(c);
      }
      else if (t == "Kd")
      {
	mat.diffuse = readColor(c);
      }
      else if (t == "Ks")
      {
	mat.specular = readColor(c);
      }
      else if (t == "Ke")
      {
	mat.emmisive = readColor(c);
      }
      else if (t == "Ns")
      { // specular power
	mat.shininess = (float)strtod(skip(c),(char**)&c);
      }
      else if (t == "Tf")
      { // transmisssion filter
	mat.alpha = readColor(c)[0];
      }
      else if (t == "d")
      { // dissolve
	mat.alpha = (float)strtod(skip(c),(char**)&c);
      }
      else if (t == "Ni")
      { // optical density
	mat.optical_density = (float)strtod(skip(c),(char**)&c);
      }
      else if (t == "illum")
      {
	int i = atoi(skip(c));
	switch (i)
	{
	case 0:	//	Color on and Ambient off
	    mat.ambient.clear();
	    mat.specular.clear();
	    mat.shininess = 0;
	    break;
	case 1:	//	Color on and Ambient on
	    mat.specular.clear();
	    mat.shininess = 0;
	    break;
	case 2:	//	Highlight on
	    break;
	case 3:	//	Reflection on and Ray trace on
	    break;
	case 4:	//	Transparency: Glass on
	        //	Reflection: Ray trace on
	    break;
	case 5:	//	Reflection: Fresnel on and Ray trace on
	    break;
	case 6:	//	Transparency: Refraction on
	        //	Reflection: Fresnel off and Ray trace on
	    break;
	case 7:	//	Transparency: Refraction on
	        //	Reflection: Fresnel on and Ray trace on
	    break;
	case 8:	//	Reflection on and Ray trace off
	    break;
	case 9:	//	Transparency: Glass on
	        //	Reflection: Ray trace off
	    break;
	case 10: //	Casts shadows onto invisible surfaces
	    break;
	}
      }
      else if (t == "map_Ka")
      {
	mat.map_ambient = getRelativeFile(skip(c),fpath);
      }
      else if (t == "map_Kd")
      {
	mat.map_diffuse = getRelativeFile(skip(c),fpath);
      }
      else if (t == "map_Ks")
      {
	mat.map_specular = getRelativeFile(skip(c),fpath);
      }
      else if (t == "map_Tf" || t == "map_d")
      {
	mat.map_alpha = getRelativeFile(skip(c),fpath);
      }
      else if (t == "bump" || t == "disp")
      {
	mat.map_bump = getRelativeFile(skip(c),fpath);
      }
      else
      {
	  std::cerr << "ERROR: MTL parsing on line "<<l<<": "<<line<<std::endl;
      }
    }
    addMaterial(mat);
    //if (line)
    //  free(line);
    fclose(fp);
    std::cout<<"Loaded file "<<filename<<std::endl;
    
    return true;    
  }

    Vec3i readPoint(const char** c, bool simple) //, int& index)
  {
    //Vec3f v_p;
    //Vec2f v_t;
    //Vec3f v_n;
    int iv = strtol(skip(*c),(char**)c,10); if (iv<0) iv += t_v.size(); else iv+=v0;
    int it = 0;
    int in = 0;
    if ((unsigned)iv>=t_v.size())
    {
      std::cerr << "Incorrect point "<<iv<<std::endl;
      iv = 0;
    }
    if (**c=='/')
    {
      ++(*c);
      it = strtol(*c,(char**)c,10); if (it<0) it += t_vt.size();
      if (simple) it = 0;
      else if ((unsigned)it>=t_vt.size())
      {
        std::cerr << "Incorrect texcoord "<<it<<std::endl;
        it = 0;
      }
    }
    if (**c=='/')
    {
      ++(*c);
      in = strtol(*c,(char**)c,10); if (in<0) in += t_vn.size();
      if (simple) in = 0;
      else if ((unsigned)in>=t_vn.size())
      {
        std::cerr << "Incorrect normal "<<in<<std::endl;
        in = 0;
      }
    }
    in += smoothgroup<<SMOOTHGROUP_SHIFT;
    return Vec3i(iv,it,in);
  }
/*
    // Find first equal point
    i = mesh->getGP0(t_v_g[iv]);
    int match_n = -1;
    while (i>=0)
    {
      if (mesh->getPN(i) == v_n)
      {
	match_n = i;
        if (mesh->getPT(i) == v_t)
        { // found a match
          ++merge;
          index = i;
          return;
        }
      }
      // else find next point
      i = mesh->getPGPNext(i);
    }
    // no match : new point
    i = mesh->nbp();
    mesh->PP(i)=v_p;
    mesh->PN(i)=v_n;
    if (mesh->getAttrib(Mesh::MESH_POINTS_TEXCOORD))
      mesh->PT(i)=v_t;
    if (t_v_g[iv]<0)
      t_v_g[iv] = mesh->nbg();
    mesh->PG(i) = t_v_g[iv];
    mesh->PGPNext(i) = mesh->getGP0(t_v_g[iv]);
    mesh->GP0(t_v_g[iv]) = i;
    index = i;
  }
*/

  bool load(const char* filename, const char* filter)
  {
    FILE* fp = fopen(filename,"r");
    if (fp==NULL) return false;
  
    Vec3f v3;
    Vec2f v2;
    Vec3i fp0,fp1,fp2;

    BBox bb;
    if (filter && *filter=='<')
    {
        int len = 0;
        sscanf(filter,"<%f,%f,%f>-<%f,%f,%f>%n",&bb.a[0],&bb.a[1],&bb.a[2],&bb.b[0],&bb.b[1],&bb.b[2],&len);
        filter+=len;
    }
    bool simple=false;
    if (filter && std::string(filter)=="simple")
    {
        simple = true;
        filter += 6;
    }

    std::string gname = "default";
    std::string matname = "";

    bool inc = true;

    int f0 = 0;
  
    merge = 0;

    mesh->setAttrib(Mesh::MESH_POINTS_POSITION,true);
    //mesh->setAttrib(Mesh::MESH_POINTS_GROUP,true);
    mesh->setAttrib(Mesh::MESH_FACES,true);

    //    char * line = NULL;
    //size_t len = 0;
    //    ssize_t read;
    char line[5000];
    int lnum = 0;
    
    t_v.clear(); t_v.push_back(v3);
    //t_v_g.clear(); t_v_g.push_back(-1);
    t_vt.clear(); t_vt.push_back(v2);
    t_vn.clear(); t_vn.push_back(v3);
    
    // Not Posix compatible
    //    while ((read = getline(&line, &len, fp)) != -1)
    line [sizeof(line)-1]='\0';
    while (fgets(line, sizeof(line), fp) != NULL)
    {
      // remove EOL
      {
	int i = strlen(line)-1;
	while (i>=0 && (line[i] == '\n' || line[i] == '\r' || line[i] == ' ' || line[i] == '\t'))
	{
	  line[i] = '\0';
	  --i;
	}
      }
      ++lnum;
#ifdef DEBUG
      std::cout << lnum << ' ' << line << std::endl;
#endif
      const char* c = line;
      switch (*c)
      {
      case '#':
	if (!strncmp(line,"#Vertex Count ",strlen("#Vertex Count ")))
	{
	  int n = atoi(line+strlen("#Vertex Count "));
	  std::cout << "Allocating "<<n<<" vertices."<<std::endl;
	  t_v.reserve(n+1);
	  //t_v_g.reserve(n+1);
	}
	else if (!strncmp(line,"#UV Vertex Count ",strlen("#UV Vertex Count ")))
	  t_vt.reserve(atoi(line+strlen("#UV Vertex Count "))+1);
	else if (!strncmp(line,"#Normal Vertex Count ",strlen("#Normal Vertex Count ")))
	  t_vn.reserve(atoi(line+strlen("#Normal Vertex Count "))+1);
	else if (!strncmp(line,"#Face Count ",strlen("#Face Count ")))
	{
	  //int n = atoi(line+strlen("#Face Count "));
	  //faces.reserve(2*n);
	  //points.reserve(4*n);
	  //t_p_prev.reserve(4*n);
	}
	break;
      case 'm':
	if (!strncmp(c,"mtllib",6))
	{
	  c += 6;
	  while (*(c = skip(c)))
	  {
	    const char* end = endtoken(c);
	    std::string mtlfilename(c,end);
	    c = end;
	    loadmtl(mtlfilename.c_str(), filename);
	  }
	}
	break;
      case 'v':
	++c;
	switch(*c)
	{
	case ' ': // position
	  v3.x() = (float)strtod(skip(c),(char**)&c);
	  v3.y() = (float)strtod(skip(c),(char**)&c);
	  v3.z() = (float)strtod(skip(c),(char**)&c);
	  t_v.push_back(v3);
	  //t_v_g.push_back(-t_v.size());
	  break;
	case 't': // texture
	  ++c;
	  if (!simple)
	  {
	      v2.x() = (float)strtod(skip(c),(char**)&c); //+0.5f/4096;
	      v2.y() = (float)strtod(skip(c),(char**)&c); //+0.5f/4096;
	      t_vt.push_back(v2);
	      mesh->setAttrib(Mesh::MESH_POINTS_TEXCOORD,true);
	  }
	  break;
	case 'n': // normal
	  ++c;
	  if (!simple)
	  {
	      v3.x() = (float)strtod(skip(c),(char**)&c);
	      v3.y() = (float)strtod(skip(c),(char**)&c);
	      v3.z() = (float)strtod(skip(c),(char**)&c);
	      t_vn.push_back(v3);
	      mesh->setAttrib(Mesh::MESH_POINTS_NORMAL,true);
	  }
	  break;
	}
	break;
      case 'f':
	++c;
	switch(*c)
	{
	case ' ': // face
	{
	  if (inc)
	  {
	      fp0 = readPoint(&c, simple);
	      fp1 = readPoint(&c, simple);
	    c = skip(c);
	    while (*c != '\n' && *c!='\0' && *c!='\r')
	    {
		fp2 = readPoint(&c, simple);

	      if (bb.isEmpty() || (bb.in(t_v[fp0[0]]) && bb.in(t_v[fp1[0]]) && bb.in(t_v[fp2[0]])))
	      {
		t_fp.push_back(fp0);
		t_fp.push_back(fp1);
		t_fp.push_back(fp2);
		// put the position index temporarily in the mesh face array
		mesh->FP(mesh->nbf())=Vec3i(fp0[0],fp1[0],fp2[0]);
		//mesh->FP(mesh->nbf())=f;
	      }
	      fp1 = fp2;
	      c = skip(c);
	    }
	  }
	}
	break;
	}
	break;
      case 'g':
	++c;
	addMatGroup(matname, gname, f0, mesh->nbf()-f0);
	f0 = mesh->nbf();
	gname = skip(c);
	inc = (!filter || !strncmp(gname.c_str(), filter, strlen(filter)) || !strncmp(matname.c_str(), filter, strlen(filter)));
	break;
      case 's':
	++c;
	if (!simple)
	{
	    smoothgroup = atoi(skip(c));
	    //std::cout << "smooth group "<<smoothgroup << std::endl;
	}
	break;
      case 'u':
	if (!strncmp(c,"usemtl",6))
	{
	  c += 6;
	  addMatGroup(matname, gname, f0, mesh->nbf()-f0);
	  f0 = mesh->nbf();
	  matname = skip(c);
	  inc = (!filter || !strncmp(gname.c_str(), filter, strlen(filter)) || !strncmp(matname.c_str(), filter, strlen(filter)));
	}
	break;
      }
    }
    if (gname != "default" || !matname.empty() || f0 > 0)
      addMatGroup(matname, gname, f0, mesh->nbf()-f0);

    // compute final vertex groups

    // First we compute for each point how many pair of normal/texcoord indices are used
    // The map store the final index of each combinaison
    std::cout << "Init vertex map"<<std::endl;
    int inputnbp = t_v.size();
    std::vector< std::map< std::pair<int,int>, int > > vertNormTexMap;
    vertNormTexMap.resize(inputnbp);
    for (unsigned int i = 0; i < t_fp.size(); i++)
    {
      fp0 = t_fp[i];
      if ((unsigned)(fp0[0]) >= (unsigned)inputnbp)
      {
	  std::cerr << "ERROR: invalid vertex index "<<fp0[0]<<" > "<<inputnbp-1<<std::endl;
	  t_fp[i][0] = fp0[0] = 0;
      }
      vertNormTexMap[fp0[0]][std::make_pair(fp0[2],fp0[1])] = 0;
    }

    // Then we can compute how many vertices are created
    std::cout << "Compute vertex count"<<std::endl;
    int nbp = 0;
    int nbg = 0;
    for (int i = 0; i < inputnbp; i++)
    {
      int s = vertNormTexMap[i].size();
      if (s>0)
      {
	nbp += s;
	++nbg;
      }
      if (s>1) // more that one point is created from a single position
          mesh->setAttrib(Mesh::MESH_POINTS_GROUP,true);

    }
    
    // Then we can create the final arrays
    std::cout << "Create final position"<<(mesh->getAttrib(Mesh::MESH_POINTS_TEXCOORD)?", texcoord":"")<<(mesh->getAttrib(Mesh::MESH_POINTS_NORMAL)?", normal":"")<<(mesh->getAttrib(Mesh::MESH_POINTS_GROUP)?", group":"")<<" arrays with "<<nbp<<" points."<<std::endl;

    mesh->points_p.resize(nbp);
    if (mesh->getAttrib(Mesh::MESH_POINTS_TEXCOORD))
      mesh->points_t.resize(nbp);
    if (mesh->getAttrib(Mesh::MESH_POINTS_NORMAL))
      mesh->points_n.resize(nbp);
    //if (nbg != nbp)
    if (mesh->getAttrib(Mesh::MESH_POINTS_GROUP))
    {
      //mesh->setAttrib(Mesh::MESH_POINTS_GROUP, true);
      mesh->points_g.resize(nbp);
      //mesh->groups_p0.resize(nbg);
    }

    for (int i = 0, j = 0; i < inputnbp; i++)
    {
      if (vertNormTexMap.empty()) continue;
      Vec3f p = t_v[i];

      int last_n = -1;
      int pg0 = mesh->groups_p0.size();

      //std::map<int, int> normTexMap;
      for (std::map<std::pair<int, int>, int>::iterator it = vertNormTexMap[i].begin();
	   it != vertNormTexMap[i].end(); ++it)
      {
	mesh->points_p[j] = p;
	int n = it->first.first;
	int t = it->first.second;
	if (mesh->getAttrib(Mesh::MESH_POINTS_TEXCOORD))
	  mesh->points_t[j] = t_vt[t];
	if (mesh->getAttrib(Mesh::MESH_POINTS_NORMAL))
	  mesh->points_n[j] = t_vn[(n & ((1<<SMOOTHGROUP_SHIFT)-1))];
	if (mesh->getAttrib(Mesh::MESH_POINTS_GROUP))
	{
	  mesh->points_g[j] = (int)mesh->groups_p0.size();
	  if (n != last_n)
	  {
	    if ((int)mesh->groups_p0.size() == pg0)
	      mesh->groups_p0.push_back(j);            // new group
	    else
	      mesh->groups_p0.push_back(-j);            // new subgroup
	  }
	}
	last_n = n;
	it->second = j++;
      }
    }

    std::cout << "Update face array"<<std::endl;
    // Finally we put the final indices in the triangles
    for (unsigned int i = 0; i < t_fp.size(); i++)
    {
      fp0 = t_fp[i];
      int index = vertNormTexMap[fp0[0]][std::make_pair(fp0[2],fp0[1])];
      mesh->FP(i/3)[i%3] = index;
    }


    //if (line)
    //  free(line);
    fclose(fp);
    std::cout<<"Loaded file "<<filename<<std::endl;
    std::cout<<t_v.size()-1<<" positions, "<<t_vt.size()-1<<" texcoords, "<<t_vn.size()-1<<" normals"<<std::endl;
    std::cout<<mesh->nbp()<<" final points, "<<mesh->nbf()<<" triangles, "<<mesh->nbg()<<" normal groups"<<std::endl;
    std::cout<<mesh->nbmatg()<<" material groups"<<std::endl;
    
    //if (mesh->nbg()==mesh->nbp())
    //{ // groups are not required
    //  mesh->setAttrib(Mesh::MESH_POINTS_GROUP,false);
    //}
    
    // Computing normals
    mesh->calcNormals();
    mesh->calcBBox();
    
    return true;
  }
};

bool Mesh::loadObj(const char* filename, const char* filter)
{
  MeshObjData data(this);
  return data.load(filename, filter);
}

bool Mesh::saveObj(const char* filename) const
{
  FILE* fp = fopen(filename,"w+");
  if (fp==NULL) return false;
  std::cout<<"Writing Obj file "<<filename<<std::endl;
  bool res = saveObj(fp);
  fclose(fp);
  return res;
}

bool Mesh::saveObj(FILE* fp, int& v0, int &vn0, int &vt0) const
{
  bool normal   = getAttrib(MESH_POINTS_NORMAL);
  bool texcoord = getAttrib(MESH_POINTS_TEXCOORD);
  bool group    = getAttrib(MESH_POINTS_GROUP);

  std::cout<<nbp()<<" points, "<<nbf()<<" faces"<<std::endl;
  if (group)
  {
      int nbgpos = 0;
      for (int i=0;i<nbg();i++)
          if (getGP0(i) >= 0)
              ++nbgpos;
      std::cout << nbg() << " normal groups, " << nbgpos << " position groups" << std::endl;
  }
  fprintf(fp,"# OBJ output by FlowVR Render\n");

  // first output material libs
  std::set<std::string> mtllibs;
  for (std::map<std::string, Material*>::const_iterator it = mat_map.begin(), itend = mat_map.end(); it != itend; ++it)
  {
    const Material* m = it->second;
    if (!m || m->mtllib.empty()) continue;
    if (mtllibs.insert(m->mtllib).second)
	fprintf(fp,"mtllib %s\n",m->mtllib.c_str());
  }

  fprintf(fp,"#Face Count %d\n",nbf());
  std::vector<int> groups_pos;
  int nbpos = 0;
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
	fprintf(fp,"v %f %f %f\n",getPP(p)[0],getPP(p)[1],getPP(p)[2]);
    }
  }
  else
  {
    nbpos = nbp();
    fprintf(fp,"#Vertex Count %d\n",nbpos);
    for (int i=0;i<nbp();i++)
    {
      fprintf(fp,"v %f %f %f\n",getPP(i)[0],getPP(i)[1],getPP(i)[2]);
    }
  }
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

  int mat = -1;
  int mat_f1 = 0;
  std::string lastgname;

  for (int i=0;i<nbf();i++)
  {
    if (i >= mat_f1 && mat+1 < (int)mat_groups.size())
    {
      ++mat;
      if (lastgname != mat_groups[mat].gname)
      {
	fprintf(fp,"g %s\n",mat_groups[mat].gname.c_str());
	lastgname = mat_groups[mat].gname;
      }
      if (mat_groups[mat].matname.empty())
	fprintf(fp,"usemtl default\n");
      else
	fprintf(fp,"usemtl %s\n", mat_groups[mat].matname.c_str());
      mat_f1 = mat_groups[mat].f0+mat_groups[mat].nbf;
    }
    fprintf(fp,"f");
    for (int j=0;j<3;j++)
    {
      int p = getFP(i)[j];
      int pp=p, pn=p, pt=p;
      if (group)
      {
	pp = groups_pos[getPG(p)];
	pn = getPG(p);
      }
      if (normal && texcoord)
	fprintf(fp," %d/%d/%d",pp+v0,pt+vt0,pn+vn0);
      else if (normal)
	fprintf(fp," %d//%d",pp+v0,pn+vn0);
      else if (texcoord)
	fprintf(fp," %d/%d",pp+v0,pt+vt0);
      else
	fprintf(fp," %d",pp+v0);
    }
    fprintf(fp,"\n");
  }
  v0 += nbpos;
  if (normal) vn0 += (group?nbg():nbp());
  if (texcoord) vt0 += nbp();
  return true;
}

} // namespace render

} // namespace flowvr
