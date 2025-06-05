import numpy as np

def generate_test_target():

    point_num=50  
    xmin=0.25
    xmax=0.75
    zmin=0.4
    zmax=0.75
    yconst=0.5
    
    # # circular target
    Rx=(xmax-xmin)/2
    Rz=(zmax-zmin)/2
    centerX=xmin+Rx
    centerZ=zmin+Rz      

    theta = np.linspace(0, 2*np.pi, point_num) 
    X = centerX + Rx*np.cos(theta)
    Z = centerZ + Rz*np.sin(theta)
    Y=yconst*np.ones((point_num,1))
    reference_states=np.column_stack((X,Y,Z))

    return reference_states



