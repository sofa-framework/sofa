#ifndef MATRICE_H
#define MATRICE_H

#include <vector>
#include <map>
#include <stack>
#include "Geometry/bounding_box.h"
#include "Geometry/vector_gen.h"

namespace CGoGN {
namespace Algo {
namespace Surface {
namespace Modelisation {

class Voxellisation {
   public:
        Voxellisation(unsigned int taille_x, unsigned int taille_y, unsigned int taille_z, Geom::BoundingBox<Geom::Vec3f> bb)
            :   m_taille_x(taille_x+2), m_taille_y(taille_y+2), m_taille_z(taille_z+2),
              m_bb_min(bb.min()), m_bb_max(bb.max()), m_data(m_taille_x*m_taille_y*m_taille_z, 0),
            m_indexes(), m_sommets(), m_faces()
        {
            m_size = 0;
        }

        void removeVoxel(int x, int y, int z) {
            if(this->m_data[(x+1) + (y+1)*m_taille_x + (z+1)*m_taille_x*m_taille_y]!=0) {
                this->m_data[(x+1) + (y+1)*m_taille_x + (z+1)*m_taille_x*m_taille_y] = 0;
                --m_size;
            }
        }

        void addVoxel(int x, int y, int z, int type=1) {
            if(this->m_data[(x+1) + (y+1)*m_taille_x + (z+1)*m_taille_x*m_taille_y]==0) {
                this->m_data[(x+1) + (y+1)*m_taille_x + (z+1)*m_taille_x*m_taille_y] = type;
                ++m_size;
            }
        }

        void addVoxel(Geom::Vec3i a, int type=1) {
            if(this->m_data[(a[0]+1) + (a[1]+1)*m_taille_x + (a[2]+1)*m_taille_x*m_taille_y]==0) {
                this->m_data[(a[0]+1) + (a[1]+1)*m_taille_x + (a[2]+1)*m_taille_x*m_taille_y] = type;
                ++m_size;
            }
        }

        int getVoxel(int x, int y, int z) {
            return m_data[(x+1) + (y+1)*m_taille_x + (z+1)*m_taille_x*m_taille_y];
        }

        void clear() {
            m_size = 0;
            m_data.clear();
            m_data = std::vector<int>(m_taille_x*m_taille_y*m_taille_z, 0);
            m_indexes.clear();
            m_sommets.clear();
            m_faces.clear();
        }

        int getTailleX() {
            return m_taille_x;
        }

        int getTailleY() {
            return m_taille_y;
        }

        int getTailleZ() {
            return m_taille_z;
        }

        int size() {
            return m_size;
        }

        void check(int type=1) {
            int voxels = 0;
            for(int i=0; i<m_taille_x; ++i) {
                for(int j=0; j<m_taille_y; ++j) {
                    for(int k=0; k<m_taille_z; ++k) {
                        voxels += m_data[i+ j*m_taille_x + k*m_taille_x*m_taille_y]==type?1:0;
                    }
                }
            }
            CGoGNout << "Il y a " << voxels << " voxel(s)" << CGoGNendl;
        }

        /*
          * Fonction qui part d'un des sommets de la bounding box et qui va marquer les pixels non déjà marqués comme faisant partie de l'extérieur
          * Utilisation de algorithme de croissance de région
          */
        void marqueVoxelsExterieurs() {
            CGoGNout << "Marquage des voxels extérieurs.." << CGoGNflush;
            std::stack<Geom::Vec3i>* pile = new std::stack<Geom::Vec3i>();
            Geom::Vec3i voxel_courant;

            //Marquage du contour extérieur
            for(int j=0; j<m_taille_y; ++j) {
                for(int k=0; k<m_taille_z; ++k) {
                    m_data[j*m_taille_x + k*m_taille_x*m_taille_y] = 2;
                    m_data[m_taille_x-1 + j*m_taille_x + k*m_taille_x*m_taille_y] = 2;
                }
            }

            for(int i=0; i<m_taille_x; ++i) {
                for(int k=0; k<m_taille_z; ++k) {
                    m_data[i + k*m_taille_x*m_taille_y] = 2;
                    m_data[i + (m_taille_y-1)*m_taille_x +k*m_taille_x*m_taille_y] = 2;
                }
            }

            for(int i=0; i<m_taille_x; ++i) {
                for(int j=0; j<m_taille_y; ++j) {
                    m_data[i + j*m_taille_x] = 2;
                    m_data[i + j*m_taille_x + (m_taille_z-1)*m_taille_x*m_taille_y] = 2;
                }
            }

            if(getVoxel(0,0,0)==0)
                pile->push(Geom::Vec3i(0,0,0));
            else if(getVoxel(getTailleX()-2,0,0)==0)
                pile->push(Geom::Vec3i(getTailleX()-2,0,0));
            else if(getVoxel(0,getTailleY()-2,0)==0)
                pile->push(Geom::Vec3i(0,getTailleY()-2,0));
            else if(getVoxel(0,0,getTailleZ()-2)==0)
                pile->push(Geom::Vec3i(0,0,getTailleZ()-2));
            else if(getVoxel(getTailleX()-2,getTailleY()-2,0)==0)
                pile->push(Geom::Vec3i(getTailleX()-2,getTailleY()-2,0));
            else if(getVoxel(getTailleX()-2,0,getTailleZ()-2)==0)
                pile->push(Geom::Vec3i(getTailleX()-2,0,getTailleZ()-2));
            else if(getVoxel(0,getTailleY()-2,getTailleZ()-2)==0)
                pile->push(Geom::Vec3i(0,getTailleY()-2,getTailleZ()-2));
            else if(getVoxel(getTailleX()-2,getTailleY()-2,getTailleZ()-2)==0)
                pile->push(Geom::Vec3i(getTailleX()-2,getTailleY()-2,getTailleZ()-2));
            while(!pile->empty()) {
                //Tant qu'il y a des voxels à traiter
                voxel_courant = pile->top();
                pile->pop();
                addVoxel(voxel_courant,2);
                if(getVoxel(voxel_courant[0]+1,voxel_courant[1], voxel_courant[2])==0)
                    pile->push(Geom::Vec3i(voxel_courant[0]+1, voxel_courant[1], voxel_courant[2]));
                if(getVoxel(voxel_courant[0]-1, voxel_courant[1], voxel_courant[2])==0)
                    pile->push(Geom::Vec3i(voxel_courant[0]-1, voxel_courant[1], voxel_courant[2]));
                if(getVoxel(voxel_courant[0], voxel_courant[1]+1, voxel_courant[2])==0)
                    pile->push(Geom::Vec3i(voxel_courant[0],voxel_courant[1]+1,voxel_courant[2]));
                if(getVoxel(voxel_courant[0],voxel_courant[1]-1,voxel_courant[2])==0)
                    pile->push(Geom::Vec3i(voxel_courant[0],voxel_courant[1]-1,voxel_courant[2]));
                if(getVoxel(voxel_courant[0],voxel_courant[1],voxel_courant[2]+1)==0)
                    pile->push(Geom::Vec3i(voxel_courant[0],voxel_courant[1],voxel_courant[2]+1));
                if(getVoxel(voxel_courant[0],voxel_courant[1],voxel_courant[2]-1)==0)
                    pile->push(Geom::Vec3i(voxel_courant[0],voxel_courant[1],voxel_courant[2]-1));
            }
            delete pile;
            CGoGNout << ".. fait" << CGoGNendl;
        }

        void extractionBord() {
            CGoGNout << "Extraction du bord.." << CGoGNflush;
            int x, y, z;

            float transfo_x = (m_bb_max[0]-m_bb_min[0])/(m_taille_x-2);
            float transfo_y = (m_bb_max[1]-m_bb_min[1])/(m_taille_y-2);
            float transfo_z = (m_bb_max[2]-m_bb_min[2])/(m_taille_z-2);

            std::map<int,int>::iterator index_sommet;
            for(int i=0; i<m_taille_x-2; ++i) {
                for(int j=0; j<m_taille_y-2; ++j) {
                    for(int k=0; k<m_taille_z-2; ++k) {
                        if(getVoxel(i,j,k)==1) {
                            //Si le voxel courant intersecte le bord du maillage de base
                            if(getVoxel(i-1,j,k)==2) {
                                //Si le voxel de gauche est un voxel de l'extérieur
                                //Sommets formant la face : 8, 5, 4, 1
                                x = i-1; y = j; z = k;
                                for(int l=0; l<4; ++l) {
                                    switch(l) {
                                        case 0 :
                                            ++x;
                                        break;
                                        case 1 :
                                            ++z;
                                        break;
                                        case 2 :
                                            ++y;
                                        break;
                                        case 3 :
                                            --z;
                                        break;
                                    }

                                    if((index_sommet=m_indexes.find(x + y*m_taille_x + z*m_taille_x*m_taille_y))==m_indexes.end()) {
                                        //Si le sommet n'a pas encore été ajouté
                                        m_indexes[x + y*m_taille_x + z*m_taille_x*m_taille_y] = m_sommets.size();   //On précise l'index du nouveau sommet
                                        m_sommets.push_back(Geom::Vec3f(m_bb_min[0]+x*transfo_x, m_bb_min[1]+y*transfo_y, m_bb_min[2]+z*transfo_z));    //On ajoute le sommet avec ses coordonnées réelles
                                    }
                                    if(index_sommet==m_indexes.end()) {
                                        m_faces.push_back(m_sommets.size()-1);  //On ajoute le sommet au tableau renseignant les faces
                                    }
                                    else {
                                        m_faces.push_back(index_sommet->second);   //On ajoute le sommet au tableau renseignant les faces
                                    }
                                }
                            }
                            if(getVoxel(i+1,j,k)==2) {
                                //Si le voxel de droite est un voxel de l'extérieur
                                //Sommets formant la face : 7, 2, 3, 6
                                x = i+1; y = j; z = k;
                                for(int l=0; l<4; ++l) {
                                    switch(l) {
                                        case 1 :
                                            ++y;
                                        break;
                                        case 2 :
                                            ++z;
                                        break;
                                        case 3 :
                                            --y;
                                        break;
                                    }

                                    if((index_sommet=m_indexes.find(x + y*m_taille_x + z*m_taille_x*m_taille_y))==m_indexes.end()) {
                                        //Si le sommet n'a pas encore été ajouté
                                        m_indexes[x + y*m_taille_x + z*m_taille_x*m_taille_y] = m_sommets.size();   //On précise l'index du nouveau sommet
                                        m_sommets.push_back(Geom::Vec3f(m_bb_min[0]+x*transfo_x, m_bb_min[1]+y*transfo_y, m_bb_min[2]+z*transfo_z));    //On ajoute le sommet avec ses coordonnées réelles
                                    }
                                    if(index_sommet==m_indexes.end()) {
                                        m_faces.push_back(m_sommets.size()-1);  //On ajoute le sommet au tableau renseignant les faces
                                    }
                                    else {
                                        m_faces.push_back(index_sommet->second);   //On ajoute le sommet au tableau renseignant les faces
                                    }
                                }
                            }
                            if(getVoxel(i,j-1,k)==2) {
                                //Si le voxel en dessous est un voxel de l'extérieur
                                //Sommets formant la face : 8, 7, 6, 5
                                x = i; y = j-1; z = k;
                                for(int l=0; l<4; ++l) {
                                    switch(l) {
                                        case 0 :
                                            ++y;
                                        break;
                                        case 1 :
                                            ++x;
                                        break;
                                        case 2 :
                                            ++z;
                                        break;
                                        case 3 :
                                            --x;
                                        break;
                                    }

                                    if((index_sommet=m_indexes.find(x + y*m_taille_x + z*m_taille_x*m_taille_y))==m_indexes.end()) {
                                        //Si le sommet n'a pas encore été ajouté
                                        m_indexes[x + y*m_taille_x + z*m_taille_x*m_taille_y] = m_sommets.size();   //On précise l'index du nouveau sommet
                                        m_sommets.push_back(Geom::Vec3f(m_bb_min[0]+x*transfo_x, m_bb_min[1]+y*transfo_y, m_bb_min[2]+z*transfo_z));    //On ajoute le sommet avec ses coordonnées réelles
                                    }
                                    if(index_sommet==m_indexes.end()) {
                                        m_faces.push_back(m_sommets.size()-1);  //On ajoute le sommet au tableau renseignant les faces
                                    }
                                    else {
                                        m_faces.push_back(index_sommet->second);   //On ajoute le sommet au tableau renseignant les faces
                                    }
                                }
                            }
                            if(getVoxel(i,j+1,k)==2) {
                                //Si le voxel au dessus est un voxel de l'extérieur
                                //Sommets formant la face : 1, 4, 3, 2
                                x = i; y = j+1; z = k;
                                for(int l=0; l<4; ++l) {
                                    switch(l) {
                                        case 1 :
                                            ++z;
                                        break;
                                        case 2 :
                                            ++x;
                                        break;
                                        case 3 :
                                            --z;
                                        break;
                                    }

                                    if((index_sommet=m_indexes.find(x + y*m_taille_x + z*m_taille_x*m_taille_y))==m_indexes.end()) {
                                        //Si le sommet n'a pas encore été ajouté
                                        m_indexes[x + y*m_taille_x + z*m_taille_x*m_taille_y] = m_sommets.size();   //On précise l'index du nouveau sommet
                                        m_sommets.push_back(Geom::Vec3f(m_bb_min[0]+x*transfo_x, m_bb_min[1]+y*transfo_y, m_bb_min[2]+z*transfo_z));    //On ajoute le sommet avec ses coordonnées réelles
                                    }
                                    if(index_sommet==m_indexes.end()) {
                                        m_faces.push_back(m_sommets.size()-1);  //On ajoute le sommet au tableau renseignant les faces
                                    }
                                    else {
                                        m_faces.push_back(index_sommet->second);   //On ajoute le sommet au tableau renseignant les faces
                                    }
                                }
                            }
                            if(getVoxel(i,j,k-1)==2) {
                                //Si le voxel derrière est un voxel de l'extérieur
                                //Sommets formant la face : 8, 1, 2 ,7
                                x = i; y = j; z = k-1;
                                for(int l=0; l<4; ++l) {
                                    switch(l) {
                                        case 0 :
                                            ++z;
                                        break;
                                        case 1 :
                                            ++y;
                                        break;
                                        case 2 :
                                            ++x;
                                        break;
                                        case 3 :
                                            --y;
                                        break;
                                    }

                                    if((index_sommet=m_indexes.find(x + y*m_taille_x + z*m_taille_x*m_taille_y))==m_indexes.end()) {
                                        //Si le sommet n'a pas encore été ajouté
                                        m_indexes[x + y*m_taille_x + z*m_taille_x*m_taille_y] = m_sommets.size();   //On précise l'index du nouveau sommet
                                        m_sommets.push_back(Geom::Vec3f(m_bb_min[0]+x*transfo_x, m_bb_min[1]+y*transfo_y, m_bb_min[2]+z*transfo_z));    //On ajoute le sommet avec ses coordonnées réelles
                                    }
                                    if(index_sommet==m_indexes.end()) {
                                        m_faces.push_back(m_sommets.size()-1);  //On ajoute le sommet au tableau renseignant les faces
                                    }
                                    else {
                                        m_faces.push_back(index_sommet->second);   //On ajoute le sommet au tableau renseignant les faces
                                    }
                                }
                            }
                            if(getVoxel(i,j,k+1)==2) {
                                //Si le voxel devant est un voxel de l'extérieur
                                //Sommets formant la face : 5, 6, 3, 4
                                x = i; y = j; z = k+1;
                                for(int l=0; l<4; ++l) {
                                    switch(l) {
                                        case 1 :
                                            ++x;
                                        break;
                                        case 2 :
                                            ++y;
                                        break;
                                        case 3 :
                                            --x;
                                        break;
                                    }

                                    if((index_sommet=m_indexes.find(x + y*m_taille_x + z*m_taille_x*m_taille_y))==m_indexes.end()) {
                                        //Si le sommet n'a pas encore été ajouté
                                        m_indexes[x + y*m_taille_x + z*m_taille_x*m_taille_y] = m_sommets.size();   //On précise l'index du nouveau sommet
                                        m_sommets.push_back(Geom::Vec3f(m_bb_min[0]+x*transfo_x, m_bb_min[1]+y*transfo_y, m_bb_min[2]+z*transfo_z));    //On ajoute le sommet avec ses coordonnées réelles
                                    }
                                    if(index_sommet==m_indexes.end()) {
                                        m_faces.push_back(m_sommets.size()-1);  //On ajoute le sommet au tableau renseignant les faces
                                    }
                                    else {
                                        m_faces.push_back(index_sommet->second);   //On ajoute le sommet au tableau renseignant les faces
                                    }
                                }
                            }
                        }
                    }
                }
            }
            CGoGNout << ".. fait. " << m_faces.size()/4. << " faces représentant le bord." << CGoGNendl;
        }

        int getNbSommets() { return m_sommets.size(); }

        int getNbFaces() { return m_faces.size()/4; }

    private:
        int m_size;

        int m_taille_x;
        int m_taille_y;
        int m_taille_z;

        Geom::Vec3f m_bb_min;
        Geom::Vec3f m_bb_max;

        std::vector<int> m_data;    //Vecteur renseignant l'ensemble des voxels entourant le maillage
        std::map<int,int> m_indexes;    //Hashmap qui permet de vérifier si un sommet a déjà été ajouté à la liste des sommets

    public:
        std::vector<Geom::Vec3f> m_sommets; //Vecteur renseignant les coordonnées réelles des sommets de la surface
        std::vector<int> m_faces;   //Vecteur renseignant les sommets attribués à chaque face
};

}   //Modelisation
}   //Surface
}   //Algo
}   //CGoGN

#endif // MATRICE_H


