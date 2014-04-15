import os

def path():
    current_dir = os.path.dirname(__file__)
    return os.path.abspath( os.path.join( current_dir, '..', '..' ) )
