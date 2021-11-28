def draw_prediction_on_image(
     image, keypoints_with_scores, crop_region=None, close_figure=False,
     output_image_height=None):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.patches as patches
    import numpy as np
    """Draws the keypoint predictions on image"""
    height, width, channel = image.shape
    aspect_ratio = float(width) / height
    fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
    # To remove the huge white borders
    fig.tight_layout(pad=0)
    ax.margins(0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.axis('off')
    im = ax.imshow(image)
    line_segments = LineCollection([], linewidths=(4), linestyle='solid')
    ax.add_collection(line_segments)
    # Turn off tick labels
    scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)
    (keypoint_locs, keypoint_edges,
    edge_colors) = _keypoints_and_edges_for_display(
        keypoints_with_scores, height, width)
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
    if keypoint_edges.shape[0]:
     line_segments.set_segments(keypoint_edges)
     line_segments.set_color(edge_colors)
    if keypoint_locs.shape[0]:
     scat.set_offsets(keypoint_locs)
    if crop_region is not None:
     xmin = max(crop_region['x_min'] * width, 0.0)
     ymin = max(crop_region['y_min'] * height, 0.0)
     rec_width = min(crop_region['x_max'], 0.99) * width - xmin
     rec_height = min(crop_region['y_max'], 0.99) * height - ymin
     rect = patches.Rectangle(
         (xmin,ymin),rec_width,rec_height,
         linewidth=1,edgecolor='b',facecolor='none')
     ax.add_patch(rect)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
       fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    if output_image_height is not None:
     output_image_width = int(output_image_height / height * width)
     image_from_plot = cv2.resize(
         image_from_plot, dsize=(output_image_width, output_image_height),
          interpolation=cv2.INTER_CUBIC)
    return image_from_plot

def _keypoints_and_edges_for_display(keypoints_with_score,height,
                                      width,keypoint_threshold=0.11):
    import numpy as np
    """Returns high confidence keypoints and edges"""
    keypoints_all = []
    keypoint_edges_all = []
    edge_colors = []
    num_instances,_,_,_ = keypoints_with_score.shape
    for id in range(num_instances):
     kpts_x = keypoints_with_score[0,id,:,1]
     kpts_y = keypoints_with_score[0,id,:,0]
     kpts_scores = keypoints_with_score[0,id,:,2]
     kpts_abs_xy = np.stack(
         [width*np.array(kpts_x),height*np.array(kpts_y)],axis=-1)
     kpts_above_thrs_abs = kpts_abs_xy[kpts_scores > keypoint_threshold,: ]
     keypoints_all.append(kpts_above_thrs_abs)
     for edge_pair,color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
       if (kpts_scores[edge_pair[0]] > keypoint_threshold and 
           kpts_scores[edge_pair[1]] > keypoint_threshold):
         x_start = kpts_abs_xy[edge_pair[0],0]
         y_start = kpts_abs_xy[edge_pair[0],1]
         x_end = kpts_abs_xy[edge_pair[1],0]
         y_end = kpts_abs_xy[edge_pair[1],1]
         lien_seg = np.array([[x_start,y_start],[x_end,y_end]])
         keypoint_edges_all.append(lien_seg)
         edge_colors.append(color)
    if keypoints_all:
     keypoints_xy = np.concatenate(keypoints_all,axis=0)
    else:
     keypoints_xy = np.zeros((0,17,2))
    if keypoint_edges_all:
     edges_xy = np.stack(keypoint_edges_all,axis=0)
    else:
     edges_xy = np.zeros((0,2,2))
    return keypoints_xy,edges_xy,edge_colors

# Dictionary to map joints of body part
KEYPOINT_DICT = {
     'nose':0,
     'left_eye':1,
     'right_eye':2,
     'left_ear':3,
     'right_ear':4,
     'left_shoulder':5,
     'right_shoulder':6,
     'left_elbow':7,
     'right_elbow':8,
     'left_wrist':9,
     'right_wrist':10,
     'left_hip':11,
     'right_hip':12,
     'left_knee':13,
     'right_knee':14,
     'left_ankle':15,
     'right_ankle':16
 } 

# map bones to matplotlib color name
KEYPOINT_EDGE_INDS_TO_COLOR = {
     (0,1): 'm',
     (0,2): 'c',
     (1,3): 'm',
     (2,4): 'c',
     (0,5): 'm',
     (0,6): 'c',
     (5,7): 'm',
     (7,9): 'm',
     (6,8): 'c',
     (8,10): 'c',
     (5,6): 'y',
     (5,11): 'm',
     (6,12): 'c',
     (11,12): 'y',
     (11,13): 'm',
     (13,15): 'm',
     (12,14): 'c',
     (14,16): 'c'
 } 

def convert_to_df(list_keypoint, cat):
    import pandas as pd
    df = pd.DataFrame(list_keypoint.items(), columns = ['image_name','keypoint'])
    nose_y = []
    nose_x = []
    nose_score = []
    left_eye_y = []
    left_eye_x = []
    left_eye_score = []
    right_eye_y = []
    right_eye_x = []
    right_eye_score = []
    left_ear_y = []
    left_ear_x = []
    left_ear_score = []
    right_ear_y = []
    right_ear_x = []
    right_ear_score = []
    left_shoulder_y = []
    left_shoulder_x = []
    left_shoulder_score = []
    right_shoulder_y = []
    right_shoulder_x = []
    right_shoulder_score = []
    left_elbow_y = []
    left_elbow_x = []
    left_elbow_score = []
    right_elbow_y = []
    right_elbow_x = []
    right_elbow_score = []
    left_wrist_y = []
    left_wrist_x = []
    left_wrist_score = []
    right_wrist_y = []
    right_wrist_x = []
    right_wrist_score = []
    left_hip_y = []
    left_hip_x = []
    left_hip_score = []
    right_hip_y = []
    right_hip_x = []
    right_hip_score = []
    left_knee_y = []
    left_knee_x = []
    left_knee_score = []
    right_knee_y = []
    right_knee_x = []
    right_knee_score = []
    left_ankle_y = []
    left_ankle_x = []
    left_ankle_score = []
    right_ankle_y = []
    right_ankle_x = []
    right_ankle_score = []
    #Bruteforce to extract all keypoints into a data frame
    for row in df['keypoint']:
        nose_y.append(row[0][0])
        nose_x.append(row[0][1])
        nose_score.append(row[0][2])
        left_eye_y.append(row[1][0])
        left_eye_x.append(row[1][1])
        left_eye_score.append(row[1][2])
        right_eye_y.append(row[2][0])
        right_eye_x.append(row[2][1])
        right_eye_score.append(row[2][2])
        left_ear_y.append(row[3][0])
        left_ear_x.append(row[3][1])
        left_ear_score.append(row[3][2])
        right_ear_y.append(row[4][0])
        right_ear_x.append(row[4][1])
        right_ear_score.append(row[4][2])
        left_shoulder_y.append(row[5][0])
        left_shoulder_x.append(row[5][1])
        left_shoulder_score.append(row[5][2])
        right_shoulder_y.append(row[6][0])
        right_shoulder_x.append(row[6][1])
        right_shoulder_score.append(row[6][2])
        left_elbow_y.append(row[7][0])
        left_elbow_x.append(row[7][1])
        left_elbow_score.append(row[7][2])
        right_elbow_y.append(row[8][0])
        right_elbow_x.append(row[8][1])
        right_elbow_score.append(row[8][2])
        left_wrist_y.append(row[9][0])
        left_wrist_x.append(row[9][1])
        left_wrist_score.append(row[9][2])
        right_wrist_y.append(row[10][0])
        right_wrist_x.append(row[10][1])
        right_wrist_score.append(row[10][2])
        left_hip_y.append(row[11][0])
        left_hip_x.append(row[11][1])
        left_hip_score.append(row[11][2])
        right_hip_y.append(row[12][0])
        right_hip_x.append(row[12][1])
        right_hip_score.append(row[12][2])
        left_knee_y.append(row[13][0])
        left_knee_x.append(row[13][1])
        left_knee_score.append(row[13][2])
        right_knee_y.append(row[14][0])
        right_knee_x.append(row[14][1])
        right_knee_score.append(row[14][2])
        left_ankle_y.append(row[15][0])
        left_ankle_x.append(row[15][1])
        left_ankle_score.append(row[15][2])
        right_ankle_y.append(row[16][0])
        right_ankle_x.append(row[16][1])
        right_ankle_score.append(row[16][2])
    df.insert(loc=0, column='category',value=cat)
    df.insert(loc=3, column='nose_y',value=nose_y)
    df.insert(loc=4, column='nose_x',value=nose_x)
    df.insert(loc=5, column='nose_score',value=nose_score)
    df.insert(loc=6, column='left_eye_y',value=left_eye_y)
    df.insert(loc=7, column='left_eye_x',value=left_eye_x)
    df.insert(loc=8, column='left_eye_score',value=left_eye_score)
    df.insert(loc=9, column='right_eye_y',value=right_eye_y)
    df.insert(loc=10, column='right_eye_x',value=right_eye_x)
    df.insert(loc=11, column='right_eye_score',value=right_eye_score)
    df.insert(loc=12, column='left_ear_y',value=left_ear_y)
    df.insert(loc=13, column='left_ear_x',value=left_ear_x)
    df.insert(loc=14, column='left_ear_score',value=left_ear_score)
    df.insert(loc=15, column='right_ear_y',value=right_ear_y)
    df.insert(loc=16, column='right_ear_x',value=right_ear_x)
    df.insert(loc=17, column='right_ear_score',value=right_ear_score)
    df.insert(loc=18, column='left_shoulder_y',value=left_shoulder_y)
    df.insert(loc=19, column='left_shoulder_x',value=left_shoulder_x)
    df.insert(loc=20, column='left_shoulder_score',value=left_shoulder_score)
    df.insert(loc=21, column='right_shoulder_y',value=right_shoulder_y)
    df.insert(loc=22, column='right_shoulder_x',value=right_shoulder_x)
    df.insert(loc=23, column='right_shoulder_score',value=right_shoulder_score)
    df.insert(loc=24, column='left_elbow_y',value=left_elbow_y)
    df.insert(loc=25, column='left_elbow_x',value=left_elbow_x)
    df.insert(loc=26, column='left_elbow_score',value=left_elbow_score)
    df.insert(loc=27, column='right_elbow_y',value=right_elbow_y)
    df.insert(loc=28, column='right_elbow_x',value=right_elbow_x)
    df.insert(loc=29, column='right_elbow_score',value=right_elbow_score)
    df.insert(loc=30, column='left_wrist_y',value=left_wrist_y)
    df.insert(loc=31, column='left_wrist_x',value=left_wrist_x)
    df.insert(loc=32, column='left_wrist_score',value=left_wrist_score)
    df.insert(loc=33, column='right_wrist_y',value=right_wrist_y)
    df.insert(loc=34, column='right_wrist_x',value=right_wrist_x)
    df.insert(loc=35, column='right_wrist_score',value=right_wrist_score)
    df.insert(loc=36, column='left_hip_y',value=left_hip_y)
    df.insert(loc=37, column='left_hip_x',value=left_hip_x)
    df.insert(loc=38, column='left_hip_score',value=left_hip_score)
    df.insert(loc=39, column='right_hip_y',value=right_hip_y)
    df.insert(loc=40, column='right_hip_x',value=right_hip_x)
    df.insert(loc=41, column='right_hip_score',value=right_hip_score)
    df.insert(loc=42, column='left_knee_y',value=left_knee_y)
    df.insert(loc=43, column='left_knee_x',value=left_knee_x)
    df.insert(loc=44, column='left_knee_score',value=left_knee_score)
    df.insert(loc=45, column='right_knee_y',value=right_knee_y)
    df.insert(loc=46, column='right_knee_x',value=right_knee_x)
    df.insert(loc=47, column='right_knee_score',value=right_knee_score)
    df.insert(loc=48, column='left_ankle_y',value=left_ankle_y)
    df.insert(loc=49, column='left_ankle_x',value=left_ankle_x)
    df.insert(loc=50, column='left_ankle_score',value=left_ankle_score)
    df.insert(loc=51, column='right_ankle_y',value=right_ankle_y)
    df.insert(loc=52, column='right_ankle_x',value=right_ankle_x)
    df.insert(loc=53, column='right_ankle_score',value=right_ankle_score)
    return df


def get_prediction(keypoints):
    import pandas as pd
    import pickle
    import numpy as np
    df_by_cat = []
    list_keypoint = {}
    list_cat = []
    cat = "unknown"
    try:
        list_keypoint['input_image'] = keypoints
        list_cat.append(cat)
    except:
        print('Error finding keypoints on Image: ' + image_path)    
    df_by_cat.append(convert_to_df(list_keypoint, cat))
    df = pd.concat([df_by_cat[0]], axis = 0)
    df = df.drop(['image_name','keypoint'], axis = 1)
    X_predict = df.drop('category', axis =1 )
    loaded_model = pickle.load(open('model.sav', 'rb'))
    #Get the prediction with probability
    y_prediction = loaded_model.predict(X_predict)
    y_prob = loaded_model.predict_proba(X_predict)[0][np.argmax(loaded_model.predict_proba(X_predict))]
    return y_prediction[0], y_prob*100