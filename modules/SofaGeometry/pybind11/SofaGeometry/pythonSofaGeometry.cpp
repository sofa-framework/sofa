#include "pythonSofaGeometry.h"

PYBIND11_MODULE(SofaGeometry, m) {
    init_vec3(m);
    init_ray(m);
    init_plane(m);
}
