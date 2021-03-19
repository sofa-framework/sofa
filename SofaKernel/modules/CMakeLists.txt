# These are real modules. If you need components from there you
# need to first add find_package(XXX) in your CMakeLists.txt

# Common
sofa_add_plugin(SofaSimpleFem SofaSimpleFem ON)
sofa_add_plugin(SofaRigid SofaRigid ON)
sofa_add_plugin(SofaDeformable SofaDeformable ON)
sofa_add_plugin(SofaObjectInteraction SofaObjectInteraction ON)
sofa_add_plugin(SofaMeshCollision SofaMeshCollision ON)
sofa_add_plugin(SofaEngine SofaEngine ON)
sofa_add_plugin(SofaExplicitOdeSolver SofaExplicitOdeSolver ON)
sofa_add_plugin(SofaImplicitOdeSolver SofaImplicitOdeSolver ON)
sofa_add_plugin(SofaLoader SofaLoader ON)

sofa_add_module(Sofa.GL Sofa.GL ON
    WHEN_TO_SHOW "SOFA_WITH_OPENGL"
    VALUE_IF_HIDDEN OFF)
