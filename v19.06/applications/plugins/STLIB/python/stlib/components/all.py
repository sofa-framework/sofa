# -*- coding: utf-8 -*-
"""
All...

"""

import Sofa
for (name,desc) in Sofa.getAvailableComponents():
    code = """def %s(owner, **kwargs):
        \"\"\"%s\"\"\"
        if kwargs == None:
                kwargs = {}
        owner.createObject(\"%s\", **kwargs)
""" % (name,desc,name)
    exec(code)

