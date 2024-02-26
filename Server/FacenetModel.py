import logging

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms

from SQL import insert_processor
from SQL import query_processor

log = logging.getLogger('facenet')
log.setLevel(logging.ERROR)

# Current problem being debugged
# The program throws errors when attempting to compare the face seen with the faces in the database
# Several errors were related to the feature-vectors needed to compare faces. As such, much of my time was
# spent learning how uses these vectors to compare faces.
class Facenet:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Running on device: {}'.format(self.device))
        # 人脸检测模型
        self.mtcnn = MTCNN(
            image_size=160, margin=20, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        ).eval()
        # 人脸识别模型
        self.resnet = InceptionResnetV1(pretrained='vggface2', device=self.device).eval().to(self.device)
        self.loader = transforms.Compose([transforms.ToTensor()])
        self.load_saved_features() #changed while debugging.

    def load_saved_features(self):
        # 从数据库加载人脸特征向量
        results = query_processor.load_face_image_feature_vector()
        self.feature_lib: list[dict] = [{
            'id': id,
            'name': name,
            'feature_vector': torch.frombuffer(feature_vector.tobytes(), dtype=torch.float32)
        } for id, name, feature_vector in results]

    def face_detect(self, image_data):
        # 人脸的位置坐标和概率
        boxes, probs = self.mtcnn.detect(image_data)
        return boxes, probs

    def boxes_to_images(self, image_data, boxes):
        aligned_images = [self.loader(image_data.crop(box).resize((160, 160))) for box in boxes]
        return torch.stack(aligned_images).to(self.device)

    def get_features(self, image_data):
        # 计算人脸特征向量
        features = self.resnet(image_data).detach().cpu()
        return features

    def face_recognize(self, images):
        # 计算人脸特征向量
        embeddings_features = self.get_features(images)
        return embeddings_features

    def face_features_compare(self, features):
        
        if features != None:
            print("-------------")
            print(self.feature_lib) 
            print("-------------")
            dists = [[(feature - feature_in_lib['feature_vector']).norm().item()
                      for feature_in_lib in self.feature_lib]
                     for feature in features]
            
            # 求最小值，即为识别到的人脸 Find the minimum value, which is the recognized face
            recognized_face = np.argmin(dists, axis=1)
            IDs = []
            Names = []
            for i in range(len(recognized_face)):
                if dists[i][recognized_face[i]] < 0.25:
                    IDs.append(self.feature_lib[recognized_face[i]['id']])
                    Names.append(self.feature_lib[recognized_face[i]['name']])
            return IDs, Names

    def register_new_face(self, id, name, image_data, new_feature_vector):
        # 注册新的人脸
        self.feature_lib.append({
            'id': id,
            'name': name,
            'feature_vector': new_feature_vector
        })
        insert_processor.store_face_image(id, np.uint8(image_data), new_feature_vector.numpy())
        print(f"register: {id, name}")


facenet = Facenet()
