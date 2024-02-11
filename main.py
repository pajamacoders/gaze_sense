import json


if __name__=="__main__":
    with open('edge.json', 'r') as f:
        data = json.load(f)
    # print(data)
    for recog_info in data: # for each row in database 
        for face in recog_info.get('recog_face', []): # for each person
            """
            face 를 api 의 입력으로 받아 주시도 분석 결과를 return 하도록 설계 하면 될듯
            api 입력:
            1. face(recog_face의 1개, headpose, bbox 포함)
            2. camera info(intrinsic & extrinsic)
            3. display position in camera coordinate frame
            """
            print(face)
            