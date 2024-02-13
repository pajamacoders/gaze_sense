import json
from GazeEstimation import GazeEstimator
from tqdm import tqdm
if __name__=="__main__":
    gaze_sense=GazeEstimator()
    camera = {"org_image_width":1920, "org_image_height":1080,
              "focal_length_x":1015, "focal_length_y":1004}
    # with open('edge_2024212_2212.json', 'r') as f:
    with open('edge2.json', 'r') as f:
        data = json.load(f)
    # print(data)
    for recog_info in tqdm(data): # for each row in database 
        # for face in recog_info.get('recog_face', []): # for each person
        """
        face 를 api 의 입력으로 받아 주시도 분석 결과를 return 하도록 설계 하면 될듯
        api 입력:
        1. face(recog_face의 1개, headpose, bbox 포함)
        2. camera info(intrinsic & extrinsic)
        3. display position in camera coordinate frame
        """
        gaze_sense.estimate_gaze_from_head_pose(recog_info, camera, None)
            