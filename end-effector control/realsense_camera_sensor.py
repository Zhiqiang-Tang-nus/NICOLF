import numpy as np

def POI_position(pipe,alignedFs,pointcloud):

    origin_point=np.asarray([20, -222, 730])

    # acquire aligned frame
    fs = pipe.wait_for_frames()
    aligned_frames = alignedFs.process(fs)   
    color_frame = aligned_frames.get_color_frame() 
    depth_frame = aligned_frames.get_depth_frame()
    depthWidth = depth_frame.get_width()
    depthHeight = depth_frame.get_height()

    # acquire pointcloud
    points = pointcloud.calculate(depth_frame)
    vertices = np.asarray(points.get_vertices(2)).reshape(depthHeight, depthWidth, 3)    

    # acquire point of interest in rgb
    fig = np.asarray(color_frame.get_data())
    row_min=400
    row_max=600
    col_min=410
    col_max=870
    fig = fig[row_min:row_max,col_min:col_max,:]
    fig=fig.astype(float)
    color_G=fig[:,:,1]/np.sqrt(np.square(fig[:,:,0])+np.square(fig[:,:,1])+np.square(fig[:,:,2]))
    index=np.where((color_G>0.65))
    index=np.asarray(index)
    index=index.transpose()       
    Centroid=np.mean(index, axis = 0)
    Centroid=Centroid+[row_min, col_min]
    Centroid=np.round(Centroid)
    Centroid=Centroid.astype(int)

# acquire poi position
    position=vertices[Centroid[0],Centroid[1],:]
    position=np.round(position,3)*1000            
    position=position-origin_point

     
    return position
