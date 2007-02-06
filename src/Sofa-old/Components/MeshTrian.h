#ifndef SOFA_COMPONENTS_MESHTRIAN_H
#define SOFA_COMPONENTS_MESHTRIAN_H

#include "Common/Mesh.h"

namespace Sofa
{

namespace Components
{

using namespace Common;

/// Cette classe permet la fabrication d'un visuel pour un fichier de type trian
/// ces fichiers se presentent de la maniere suivante
/// nombre de sommets
///liste des coordonnees des sommets ex 1.45 1.25 6.85
/// nombre de faces
///liste de toutes les faces ex 1 2 3 0 0 0 les 3 derniers chiffres ne sont pas utilises pour le moment

class MeshTrian : public Mesh
{
private:

    void readTrian(FILE *file);

public:

    MeshTrian(const std::string& filename)
    {
        init(filename);
    }

    void init(std::string filename);
};

} // namespace Components

} // namespace Sofa

#endif
