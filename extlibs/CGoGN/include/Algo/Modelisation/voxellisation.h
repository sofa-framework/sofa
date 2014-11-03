#ifndef _VOXELLISATION_H_
#define _VOXELLISATION_H_

#include "Geometry/bounding_box.h"
#include "Geometry/vector_gen.h"

#include "Algo/MC/image.h"

// #include "Topology/map/embeddedMap2.h"

#include <vector>
#include <stack>
#include <map>

namespace CGoGN
{

namespace Algo
{

namespace Surface
{

namespace Modelisation
{

//struct PFP: public PFP_STANDARD
//{
//	// definition of the map
//	typedef EmbeddedMap2 MAP ;
//	typedef VEC3 VEC3;
//};

template <typename T>
int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

template <typename T>
inline void swapMax(T& min, T& max)
{
    if(min>max)
    {
        std::swap(min, max);
    }
}

template <unsigned int DIM, typename T>
inline void swapVectorMax(sofa::defaulttype::Vec<DIM, T>& min, sofa::defaulttype::Vec<DIM, T>& max)
{
    for(unsigned int i=0; i< DIM; i++)
    {
        swapMax<T>(min[i], max[i]);
    }
}

class Voxellisation
{
public:
    Voxellisation(
            Geom::Vec3i resolutions = Geom::Vec3i(),
            Geom::BoundingBox<Geom::Vec3f> bb = Geom::BoundingBox<Geom::Vec3f>(Geom::Vec3f())
            ) :
        m_taille_x(resolutions[0]+2),
        m_taille_y(resolutions[1]+2),
        m_taille_z(resolutions[2]+2),
        m_bb_min(bb.min()),
        m_bb_max(bb.max()),
        m_data(m_taille_x*m_taille_y*m_taille_z, 0),
        m_indexes(),
        m_sommets(),
        m_faces(),
        m_transfo()
    {
        m_size = 0;

        m_transfo[0] = (m_bb_max[0]-m_bb_min[0])/(m_taille_x-2);
        m_transfo[1] = (m_bb_max[1]-m_bb_min[1])/(m_taille_y-2);
        m_transfo[2] = (m_bb_max[2]-m_bb_min[2])/(m_taille_z-2);

        swapVectorMax<3,float> (m_bb_min, m_bb_max);
    }

    void removeVoxel(int x, int y, int z)
    {
        if(x>=0 && y>=0 && z>=0 && x<m_taille_x-1 && y<m_taille_y-1 && z<m_taille_z-1)
        {
            if(this->m_data[(x+1) + (y+1)*m_taille_x + (z+1)*m_taille_x*m_taille_y]!=0) --m_size;
            this->m_data[(x+1) + (y+1)*m_taille_x + (z+1)*m_taille_x*m_taille_y] = 0;
        }
    }

    void addVoxel(int x, int y, int z, int type=1)
    {
        if(x>=-1 && y>=-1 && z>=-1 && x<m_taille_x-1 && y<m_taille_y-1 && z<m_taille_z-1)
        {
            if(this->m_data[(x+1) + (y+1)*m_taille_x + (z+1)*m_taille_x*m_taille_y]==0 && type==1) ++m_size;
            this->m_data[(x+1) + (y+1)*m_taille_x + (z+1)*m_taille_x*m_taille_y] = type;
        }
    }

    void addVoxelRaw(int x, int y, int z, int type=1)
    {
        if(x>=0 && y>=0 && z>=0 && x<m_taille_x && y<m_taille_y && z<m_taille_z)
        {
            if(this->m_data[x + y*m_taille_x + z*m_taille_x*m_taille_y]==0 && type==1) ++m_size;
            this->m_data[x + y*m_taille_x + z*m_taille_x*m_taille_y] = type;
        }
    }

    void addVoxel(Geom::Vec3i a, int type=1)
    {
        if(a[0]>=-1 && a[1]>=-1 && a[2]>=-1 && a[0]<m_taille_x-1 && a[1]<m_taille_y-1 && a[2]<m_taille_z-1)
        {
            if(this->m_data[(a[0]+1) + (a[1]+1)*m_taille_x + (a[2]+1)*m_taille_x*m_taille_y]==0 && type==1) ++m_size;
            this->m_data[(a[0]+1) + (a[1]+1)*m_taille_x + (a[2]+1)*m_taille_x*m_taille_y] = type;
        }
    }

    void addVoxelRaw(Geom::Vec3i a, int type=1)
    {
        if(a[0]>=0 && a[1]>=0 && a[2]>=0 && a[0]<m_taille_x && a[1]<m_taille_y && a[2]<m_taille_z)
        {
            if(this->m_data[a[0] + a[1]*m_taille_x + a[2]*m_taille_x*m_taille_y]==0 && type==1) ++m_size;
            this->m_data[a[0] + a[1]*m_taille_x + a[2]*m_taille_x*m_taille_y] = type;
        }
    }

    int getVoxel(int x, int y, int z)
    {
        if(x>=-1 && y>=-1 && z>=-1 && x<m_taille_x-1 && y<m_taille_y-1 && z<m_taille_z-1)
            return m_data[(x+1) + (y+1)*m_taille_x + (z+1)*m_taille_x*m_taille_y];
        else
            return 0;
    }

    int getVoxelRaw(int x, int y, int z)
    {
        if(x>=0 && y>=0 && z>=0 && x<m_taille_x && y<m_taille_y && z<m_taille_z)
            return m_data[x + y*m_taille_x + z*m_taille_x*m_taille_y];
        else
            return -1;
    }

    int getVoxel(Geom::Vec3i a)
    {
        if(a[0]>=-1 && a[1]>=-1 && a[2]>=-1 && a[0]<m_taille_x-1 && a[1]<m_taille_y-1 && a[1]<m_taille_z-1)
            return m_data[(a[0]+1) + (a[1]+1)*m_taille_x + (a[2]+1)*m_taille_x*m_taille_y];
        else
            return 0;
    }

    int getVoxelRaw(Geom::Vec3i a)
    {
        if(a[0]>=0 && a[1]>=0 && a[2]>=0 && a[0]<m_taille_x && a[1]<m_taille_y && a[1]<m_taille_z)
            return m_data[a[0] + a[1]*m_taille_x + a[2]*m_taille_x*m_taille_y];
        else
            return 0;
    }

    void clear()
    {
        m_size = 0;
        m_data.clear();
        m_data = std::vector<int>(m_taille_x*m_taille_y*m_taille_z, 0);
        m_indexes.clear();
        m_sommets.clear();
        m_faces.clear();
    }

    int getTailleX()
    {
        return m_taille_x;
    }

    int getTailleY()
    {
        return m_taille_y;
    }

    int getTailleZ()
    {
        return m_taille_z;
    }

    int getResolutionX()
    {
        return m_taille_x-2;
    }

    int getResolutionY()
    {
        return m_taille_y-2;
    }

    int getResolutionZ()
    {
        return m_taille_z-2;
    }

    int getResolution(int resolution)
    {
        int res = -1;
        switch(resolution)
        {
        case 0:
            res = m_taille_x-2;
            break;
        case 1:
            res = m_taille_y-2;
            break;
        case 2:
            res = m_taille_z-2;
            break;
        }
        return res;
    }

    int size()
    {
        return m_size;
    }

    /*
      * Fonction qui réalise le remplissage d'un polygone convexe
      */
    void voxellisePolygone(std::vector<Geom::Vec3i>& polygone)
    {
        Geom::Vec3i a, b, c = polygone.back();
        int x, y, z, dx, dy, dz, swap, ddy, ddz, sx, sy, sz;
        for(unsigned int i=0; i<polygone.size()-1;++i)
        {
            a = polygone[i];
            b = polygone[i+1];

            if(a==b)
                voxelliseLine(a, c);
            else
            {
                x = a[0], y = a[1], z = a[2];
                dx = abs(b[0]-a[0]);
                dy = abs(b[1]-a[1]);
                dz = abs(b[2]-a[2]);
                swap=0;

                if(dy > dx)
                {
                    if(dy > dz)
                    {
                        std::swap(dx,dy);
                        swap=1;
                    }
                    else
                    {
                        std::swap(dx,dz);
                        swap=2;
                    }
                }
                else
                {
                    if(dx < dz)
                    {
                        std::swap(dx,dz);
                        swap=2;
                    }
                }

                sx = sign(b[0]-a[0]);
                sy = sign(b[1]-a[1]);
                sz = sign(b[2]-a[2]);

                ddy = (dy<<1)-dx;
                ddz = (dz<<1)-dx;
                voxelliseLine(Geom::Vec3i(x, y, z), c);
                for(int i=0; i<dx; ++i)
                {
                    while(ddy>=0)
                    {
                        ddy -= dx<<1;
                        if(swap==1)
                            x+=sx;
                        else
                            y+=sy;
                    }
                    voxelliseLine(Geom::Vec3i(x, y, z), c);
                    while(ddz>=0)
                    {
                        ddz -= dx<<1;
                        if(swap==2)
                            x+=sx;
                        else
                            z+=sz;
                    }
                    voxelliseLine(Geom::Vec3i(x, y, z), c);
                    ddy += dy<<1;
                    ddz += dz<<1;
                    if(swap==1)
                        y+=sy;
                    else if(swap==2)
                        z+=sz;
                    else
                        x+=sx;
                    voxelliseLine(Geom::Vec3i(x, y, z), c);
                }
            }
        }
    }

    /*
      * Fonction qui réalise un trac de droite discrte en 3D du voxel 'a' au voxel 'b'
      * L'algorithme utilis est celui de Bresenham, adapt  la 3D
      */
    void voxelliseLine(Geom::Vec3i a, Geom::Vec3i b)
    {
        int x, y, z, dx, dy, dz, swap, ddy, ddz, sx, sy, sz;

        if(a==b)
            addVoxel(a);
        else
        {
            x = a[0], y = a[1], z = a[2];
            dx = abs(b[0]-a[0]);
            dy = abs(b[1]-a[1]);
            dz = abs(b[2]-a[2]);
            swap=0;

            if(dy > dx)
            {
                if(dy > dz)
                {
                    std::swap(dx,dy);
                    swap=1;
                }
                else
                {
                    std::swap(dx,dz);
                    swap=2;
                }
            }
            else
            {
                if(dx < dz)
                {
                    std::swap(dx,dz);
                    swap=2;
                }
            }

            sx = sign(b[0]-a[0]);
            sy = sign(b[1]-a[1]);
            sz = sign(b[2]-a[2]);

            ddy = (dy<<1)-dx;
            ddz = (dz<<1)-dx;
            addVoxel(x, y, z);
            for(int i=0; i<dx; ++i)
            {
                while(ddy>=0)
                {
                    ddy -= dx<<1;
                    if(swap==1)
                        x+=sx;
                    else
                        y+=sy;
                }
                addVoxel(x, y, z); //On affiche les points intermdiaires -> ligne 6-connexe
                while(ddz>=0)
                {
                    ddz -= dx<<1;
                    if(swap==2)
                        x+=sx;
                    else
                        z+=sz;
                }
                addVoxel(x, y, z); //On affiche les points intermdiaires -> ligne 6-connexe
                ddy += dy<<1;
                ddz += dz<<1;
                if(swap==1)
                    y+=sy;
                else if(swap==2)
                    z+=sz;
                else
                    x+=sx;
                addVoxel(x, y, z);
            }
        }
    }

    /*
      * Fonction qui part d'un des sommets de la Bounding box et qui va marquer les pixels non déjà marqués comme faisant partie de l'extérieur
      * Utilisation de algorithme de croissance de rgion
      */
    void marqueVoxelsExterieurs()
    {
        CGoGNout << "Marquage des voxels extérieurs.." << CGoGNflush;
        std::stack<Geom::Vec3i> pile;
        Geom::Vec3i voxel_courant;

        pile.push(Geom::Vec3i(0,0,0));

        while(!pile.empty())
        {
            //Tant qu'il y a des voxels  traiter
            voxel_courant = pile.top();
            pile.pop();
            addVoxelRaw(voxel_courant,2);
            if(getVoxelRaw(voxel_courant[0]+1,voxel_courant[1], voxel_courant[2])==0)
                pile.push(Geom::Vec3i(voxel_courant[0]+1, voxel_courant[1], voxel_courant[2]));
            if(getVoxelRaw(voxel_courant[0]-1, voxel_courant[1], voxel_courant[2])==0)
                pile.push(Geom::Vec3i(voxel_courant[0]-1, voxel_courant[1], voxel_courant[2]));
            if(getVoxelRaw(voxel_courant[0], voxel_courant[1]+1, voxel_courant[2])==0)
                pile.push(Geom::Vec3i(voxel_courant[0],voxel_courant[1]+1,voxel_courant[2]));
            if(getVoxelRaw(voxel_courant[0],voxel_courant[1]-1,voxel_courant[2])==0)
                pile.push(Geom::Vec3i(voxel_courant[0],voxel_courant[1]-1,voxel_courant[2]));
            if(getVoxelRaw(voxel_courant[0],voxel_courant[1],voxel_courant[2]+1)==0)
                pile.push(Geom::Vec3i(voxel_courant[0],voxel_courant[1],voxel_courant[2]+1));
            if(getVoxelRaw(voxel_courant[0],voxel_courant[1],voxel_courant[2]-1)==0)
                pile.push(Geom::Vec3i(voxel_courant[0],voxel_courant[1],voxel_courant[2]-1));
        }
        CGoGNout << ".. fait." << CGoGNendl;
    }

    /*
      * Fonction qui extrait les faces extrieu res des voxels en regardant quelles faces appartiennent  un voxel extrieur et un voxel d'intersection
      */
    void extractionBord()
    {
        CGoGNout << "Extraction du bord.." << CGoGNflush;

        m_indexes.clear();
        m_faces.clear();
        m_faces.reserve(m_size*6);  //Au maximum on a 6 fois plus de faces que le nombre de voxels (cas d'1 seul voxel qui intersecte la surface du maillage
        m_sommets.clear();
        m_sommets.reserve(m_size*6*4);  //On a 4 sommets par face
        int x, y, z;

        for(int i=0; i<m_taille_x-2; ++i)
        {
            for(int j=0; j<m_taille_y-2; ++j)
            {
                for(int k=0; k<m_taille_z-2; ++k)
                {
                    if(getVoxel(i,j,k)==1)
                    {
                        //Si le voxel courant intersecte le bord du maillage de base
                        if(getVoxel(i-1,j,k)==2)
                        {
                            //Si le voxel de gauche est un voxel de l'extrieur
                            //Sommets formant la face : 8, 5, 4, 1
                            x = i-1; y = j; z = k;
                            ajouteSommet(++x,y,z);
                            ajouteSommet(x,y,++z);
                            ajouteSommet(x,++y,z);
                            ajouteSommet(x,y,--z);
                        }
                        if(getVoxel(i+1,j,k)==2)
                        {
                            //Si le voxel de droite est un voxel de l'extrieur
                            //Sommets formant la face : 7, 2, 3, 6
                            x = i+1; y = j; z = k;
                            ajouteSommet(x,y,z);
                            ajouteSommet(x,++y,z);
                            ajouteSommet(x,y,++z);
                            ajouteSommet(x,--y,z);
                        }
                        if(getVoxel(i,j-1,k)==2)
                        {
                            //Si le voxel en dessous est un voxel de l'extrieur
                            //Sommets formant la face : 8, 7, 6, 5
                            x = i; y = j-1; z = k;
                            ajouteSommet(x,++y,z);
                            ajouteSommet(++x,y,z);
                            ajouteSommet(x,y,++z);
                            ajouteSommet(--x,y,z);
                        }
                        if(getVoxel(i,j+1,k)==2)
                        {
                            //Si le voxel au dessus est un voxel de l'extrieur
                            //Sommets formant la face : 1, 4, 3, 2
                            x = i; y = j+1; z = k;
                            ajouteSommet(x,y,z);
                            ajouteSommet(x,y,++z);
                            ajouteSommet(++x,y,z);
                            ajouteSommet(x,y,--z);
                        }
                        if(getVoxel(i,j,k-1)==2)
                        {
                            //Si le voxel derrire est un voxel de l'extrieur
                            //Sommets formant la face : 8, 1, 2 ,7
                            x = i; y = j; z = k-1;
                            ajouteSommet(x,y,++z);
                            ajouteSommet(x,++y,z);
                            ajouteSommet(++x,y,z);
                            ajouteSommet(x,--y,z);
                        }
                        if(getVoxel(i,j,k+1)==2)
                        {
                            //Si le voxel devant est un voxel de l'extrieur
                            //Sommets formant la face : 5, 6, 3, 4
                            x = i; y = j; z = k+1;
                            ajouteSommet(x,y,z);
                            ajouteSommet(++x,y,z);
                            ajouteSommet(x,++y,z);
                            ajouteSommet(--x,y,z);
                        }
                    }
                }
            }
        }
        CGoGNout << ".. fait. " << CGoGNendl;
    }

    void ajouteSommet(int x, int y, int z)
    {
        std::map<int,int>::iterator index_sommet;
        if((index_sommet=m_indexes.find(x + y*m_taille_x + z*m_taille_x*m_taille_y))==m_indexes.end())
        {
            //Si le sommet n'a pas encore t ajout
            m_indexes[x + y*m_taille_x + z*m_taille_x*m_taille_y] = m_sommets.size();   //On prcise l'index du nouveau sommet
            m_sommets.push_back(Geom::Vec3f(m_bb_min[0]+x*m_transfo[0], m_bb_min[1]+y*m_transfo[1], m_bb_min[2]+z*m_transfo[2]));    //On ajoute le sommet avec ses coordonnes relles
        }
        if(index_sommet==m_indexes.end())
            m_faces.push_back(m_sommets.size()-1);  //On ajoute le sommet au tableau renseignant les faces
        else
            m_faces.push_back(index_sommet->second);   //On ajoute le sommet au tableau renseignant les faces
    }

    /*
      * Fonction qui remplit une voxellisation en regardant l'ensemble des voxels qui n'ont pas encore été marqués (ni extérieur, ni surface)
      */
    void remplit()
    {
        for(int i=0; i<m_taille_x-2; ++i)
        {
            for(int j=0; j<m_taille_y-2; ++j)
            {
                for(int k=0; k<m_taille_z-2; ++k)
                {
                    if(getVoxel(i,j,k)==0)
                        //Si le voxel fait partie de l'intérieur
                        addVoxel(i,j,k,1);
                }
            }
        }
    }

    /*
      * Fonction qui dilate une voxellisation
      */
    void dilate(unsigned int iterations=1)
    {
        //On agrandit la taille de la voxellisation
        std::vector<int> new_m_data((m_taille_x+2)*(m_taille_y+2)*(m_taille_z+2),2);
        for(int i=0; i<m_taille_x; ++i)
        {
            for(int j=0; j<m_taille_y; ++j)
            {
                for(int k=0; k<m_taille_z; ++k)
                {
                    new_m_data[(i+1) + (j+1) *(m_taille_x+2) + (k+1) *(m_taille_x+2)*(m_taille_y+2)] = m_data[i + j*m_taille_x + k*m_taille_x*m_taille_y];
                }
            }
        }
        m_data = std::vector<int>(new_m_data);
        m_taille_x += 2;
        m_taille_y += 2;
        m_taille_z += 2;
        m_bb_min[0] -= m_transfo[0];
        m_bb_min[1] -= m_transfo[1];
        m_bb_min[2] -= m_transfo[2];
        m_bb_max[0] += m_transfo[0];
        m_bb_max[1] += m_transfo[1];
        m_bb_max[2] += m_transfo[2];

        std::vector<Geom::Vec3i> element_ajoutes;
        std::vector<Geom::Vec3i>::iterator it;

        while(iterations>0)
        {
            for(int i=0; i<m_taille_x-2; ++i)
            {
                for(int j=0; j<m_taille_y-2; ++j)
                {
                    for(int k=0; k<m_taille_z-2; ++k)
                    {
                        if(getVoxel(i,j,k)==1)
                        {
                            //Si le voxel appartient  la surface de la cage
                            for(int ci = -1; ci<2; ++ci)
                            {
                                for(int cj = -1; cj<2; ++cj)
                                {
                                    for(int ck = -1; ck<2; ++ck)
                                    {
                                        if(getVoxel(i+ci,j+cj,k+ck)==2)
                                        {
                                            //Si le voxel de gauche appartient  l'extrieur
                                            addVoxel(i+ci,j+cj,k+ck,3);
                                            element_ajoutes.push_back(Geom::Vec3i(i+ci,j+cj,k+ck));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for(it = element_ajoutes.begin(); it!=element_ajoutes.end(); ++it)
            {
                //On remplace la valeur temporaire affecte aux voxels ajouts
                addVoxel(*it,1);
            }
            element_ajoutes.clear();
            --iterations;
        }
    }

    /*
      * Fonction qui transforme une voxellisation en une image pour l'utilisation de l'algorithme de MarchingCube
      */
    Algo::Surface::MC::Image<int>* getImage()
    {
        Algo::Surface::MC::Image<int>* image = new Algo::Surface::MC::Image<int>(
                    &m_data[0], m_taille_x, m_taille_y, m_taille_z,
                m_transfo[0], m_transfo[1], m_transfo[2], false);
        return image;
    }

    int getNbSommets()
    {
        return m_sommets.size();
    }

    int getNbFaces()
    {
        return m_faces.size()/4;
    }

    void checkVoxels(int type=1)
    {
        int voxels = 0;
        for(int i=0; i<m_taille_x; ++i)
        {
            for(int j=0; j<m_taille_y; ++j)
            {
                for(int k=0; k<m_taille_z; ++k)
                {
                    voxels += m_data[i+ j*m_taille_x + k*m_taille_x*m_taille_y]==type?1:0;
                }
            }
        }
        CGoGNout << "Il y a " << voxels << " voxel(s)" << CGoGNendl;
    }

    void checkSommets()
    {
        for(unsigned int i=0; i<m_sommets.size(); ++i) {
            CGoGNout << "Sommet : " << m_sommets[i] << CGoGNendl;
        }
    }

    void check()
    {
        CGoGNout << "m_bb_min  = {" << m_bb_min << "}" << CGoGNendl;
        CGoGNout << "m_transfo  = {" << m_transfo << "}" << CGoGNendl;
    }

private:
    int m_size;

    int m_taille_x;
    int m_taille_y;
    int m_taille_z;

    Geom::Vec3f m_bb_min;
    Geom::Vec3f m_bb_max;

    std::vector<int> m_data;    //Vecteur renseignant l'ensemble des voxels entourant le maillage
    std::map<int,int> m_indexes;    //Hashmap qui permet de vrifier si un sommet a dj t ajout  la liste des sommets

public:
    std::vector<Geom::Vec3f> m_sommets; //Vecteur renseignant les coordonnes relles des sommets de la surface
    std::vector<int> m_faces;   //Vecteur renseignant les sommets attribus  chaque face

public:
    Geom::Vec3f m_transfo;
};

} // namespace Modelisation

} // namespace Surface

} // namespace Algo

} // namespace CGoGN

#endif // _VOXELLISATION_H_
