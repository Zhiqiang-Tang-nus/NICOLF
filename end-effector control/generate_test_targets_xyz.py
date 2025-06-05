import numpy as np

def generate_training_target():

    target_points=40
    t=np.linspace(0, target_points, num=target_points)

    xmin=np.random.rand(1)*0.4
    xmax=np.random.rand(1)*0.4+0.6
    Xphase=np.random.rand(1)*2*np.pi
    Xfreq=(np.random.rand(1)*1/2+1/2)/target_points
    X=(xmax+xmin)/2+(xmax-xmin)/2*np.sin(2*np.pi*Xfreq*t+Xphase)

    ymin=np.random.rand(1)*0.2
    ymax=np.random.rand(1)*0.3+0.5
    Yphase=np.random.rand(1)*2*np.pi
    Yfreq=(np.random.rand(1)*1/2+1/2)/target_points
    Y=(ymax+ymin)/2+(ymax-ymin)/2*np.sin(2*np.pi*Yfreq*t+Yphase)

    zmin=np.random.rand(1)*0.4
    zmax=np.random.rand(1)*0.4+0.6   
    Zphase=np.random.rand(1)*2*np.pi
    Zfreq=(np.random.rand(1)*1/2+1/2)/target_points
    Z=(zmax+zmin)/2+(zmax-zmin)/2*np.sin(2*np.pi*Zfreq*t+Zphase)
    
    reference_states=np.row_stack((X,Y,Z)).transpose()


    initial_states=np.asarray([0.4, 0.1, 0.5])

    return reference_states, initial_states



