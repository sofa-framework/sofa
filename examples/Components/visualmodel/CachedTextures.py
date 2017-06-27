import Sofa
import SofaPython.Tools

def createScene(root):

    root.createObject('DefaultAnimationLoop')
    root.createObject('VisualStyle', displayFlags="showVisual" )
    root.createObject('DefaultVisualManagerLoop')

    groupNode = root.createChild("Group")

    groupNode.createObject('OglIntVariable', name='sampler', id='Sampler', value='0')
    groupNode.createObject('OglShader', name='shader', fileVertexShaders=["shaders/applyTexture.vert"], fileFragmentShaders=["shaders/applyTexture.frag"])

    row = 3
    col = 3
    count = row * col

    margin = 3

    for y in range(0, row):
        for x in range(0, col):
            i = x + y * col
            visualNode = groupNode.createChild("Visual " + str(i))
            visualNode.createObject('OglTexture', name='texture', textureFilename='textures/board.png' if 0 == i % 2 else 'textures/floor2.bmp', textureUnit='0') # cached=False to disable texture caching (non-procedurale texture are never cached)
            visualNode.createObject('OglModel', name='mesh', fileMesh="mesh/cubeUV.obj", putOnlyTexCoords=True, translation=str(x * margin) + " " + str(y * margin) + " 0")
