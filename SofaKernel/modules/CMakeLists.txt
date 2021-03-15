# These are real modules. If you need components from there you
# need to first add find_package(XXX) in your CMakeLists.txt

# Common
sofa_add_module(SofaSimpleFem SofaSimpleFem ON)
sofa_add_module(SofaRigid SofaRigid ON)
sofa_add_module(SofaDeformable SofaDeformable ON)
sofa_add_module(SofaObjectInteraction SofaObjectInteraction ON)
sofa_add_module(SofaMeshCollision SofaMeshCollision ON)
sofa_add_module(SofaEngine SofaEngine ON)
sofa_add_module(SofaExplicitOdeSolver SofaExplicitOdeSolver ON)
sofa_add_module(SofaImplicitOdeSolver SofaImplicitOdeSolver ON)
sofa_add_module(SofaLoader SofaLoader ON)

sofa_add_module(Sofa.GL Sofa.GL ON
    WHEN_TO_SHOW "SOFA_WITH_OPENGL"
    VALUE_IF_HIDDEN OFF)
