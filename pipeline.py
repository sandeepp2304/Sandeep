import wget
import json

import face_recognition
import numpy as np
import warnings
import os
import os.path
import imutils
import dlib
import pandas as pd
import difflib
import scipy.cluster.hierarchy as sch
from PIL import ImageFile
from collections import Counter
# from pymongo import MongoClient
from sklearn.feature_extraction import text
from sklearn.cluster import AgglomerativeClustering
from IPython.core.interactiveshell import InteractiveShell
vectorizer = text.TfidfVectorizer(lowercase=False, token_pattern=r'\S+')
ImageFile.LOAD_TRUNCATED_IMAGES = True
InteractiveShell.ast_node_interactivity = "last_expr"
pd.options.display.max_columns = 50
warnings.filterwarnings("ignore")


class ML:
    def __init__(self):
        self.clustered_dataframe = list()
        self.face_detector = dlib.get_frontal_face_detector()
        self.pose_predictor_68_point = dlib.shape_predictor('./profile_matching_app/shape_predictor_68_face_landmarks.dat')
        self.face_encoder = dlib.face_recognition_model_v1('./profile_matching_app//dlib_face_recognition_resnet_model_v1.dat')

    # def fetch_db_data(self, query):
    #
    #     client = MongoClient('192.168.20.59', 27017)
    #     db = client.social_data
    #     collection = db[query]
    #     data = pd.DataFrame(list(collection.find()))
    #
    #     # drop the profiles, not having the image
    #     final_data = data.dropna(subset=['image'])
    #     self.download_images(final_data)
    #     # self.image_information(final_data)

    def download_images(self, final_data):

        # for downloading the image from urls
        for url in final_data['image']:
            print(url)
            wget.download(url, "./profile_matching_app/images")
        self.image_information(final_data)

    def image_information(self, final_data):

        # information of the images
        a = os.listdir("./profile_matching_app/images")
        no_0f_images_in_folder = len(a)
        no_of_images_in_dataset = len(final_data['image'])
        print(no_of_images_in_dataset)
        # print("percentage of images remained", ((no_0f_images_in_folder / no_of_images_in_dataset) * 100))
        self.image_pre_processing(final_data)

    def whirldata_face_detectors(self, img, number_of_times_to_upsample=1):
        return self.face_detector(img, number_of_times_to_upsample)

    def whirldata_face_encodings(self, face_image, num_jitters=1):
        face_locations = self.whirldata_face_detectors(face_image)
        pose_predictor = self.pose_predictor_68_point
        predictors = [pose_predictor(face_image, face_location) for face_location in face_locations]
        return [np.array(self.face_encoder.compute_face_descriptor(face_image, predictor, num_jitters)) for predictor in
                predictors][0]

    def image_pre_processing(self, final_data):

        X = []
        y = []
        img_path = "./profile_matching_app/images"
        a = os.listdir('./profile_matching_app/images')
        count = 0

        # Loop through each image for the current person
        for each_image in a:
            each_image = os.path.join(img_path, each_image)
            image = face_recognition.load_image_file(each_image)
            image = imutils.resize(image, width=600)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) == 0:
                count += 1
                # If there are no people  in a training image, skip the image.
                print("Image not suitable for training --- Didn't find a face in: {}".format(each_image) if len(
                    face_bounding_boxes) < 1 else "Found more than one face")
            else:
                # Add face encoding for current image to the training set
                X.append(self.whirldata_face_encodings(image))
                y.append(each_image)
        print("done encoding", count)

        # removing images from the folder
        self.remove_image()

        # Using the dendrogram to find the optimal number of clusters
        dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))

        # Fitting Hierarchical Clustering to the dataset
        hc = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, affinity='euclidean', linkage='ward')
        y_hc = hc.fit_predict(X)

        # Submission
        image_clusters = pd.DataFrame({'Name': y, 'Clusters': y_hc})
        path_length = len(img_path)
        link = path_length + 1
        image_clusters['result'] = image_clusters['Name'].map(lambda x: str(x)[link:])

        # droping the unwanted features
        image_clusters = image_clusters.drop(columns=['Name'], axis=1)

        # Submission
        image_data = pd.DataFrame({'common': image_clusters['result'], 'Clusters': image_clusters['Clusters']})
        # Sorting data according to user_id in ascending order
        image_data = image_data.sort_values('Clusters', axis=0, ascending=True, inplace=False, kind='quicksort',
                                            na_position='last')

        corpus = []
        for sentance in final_data["image"].values:
            review = sentance.split('/')[-1]
            corpus.append(review)
            final_data.append(corpus)

        # assign new column to existing dataframe
        final_data = final_data.assign(common=corpus)

        # Write it to a new
        final_data['common'] = final_data['common'].astype(str)
        image_data['common'] = image_data['common'].astype(str)

        df3 = pd.merge(final_data, image_data, on=['common'])
        df3 = df3.sort_values('Clusters', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
        df3 = df3.dropna(axis=0, subset=['Clusters'])

        # replacing na values as null
        df3["username"].fillna("null", inplace=True)
        df3["description"].fillna("null", inplace=True)
        df3["name"].fillna("null", inplace=True)

        unique_list = list(set(df3.Clusters.values))

        # based on cluster value putting data in dictonary
        pre_cluster = {}
        for i in unique_list:
            pre_cluster[str(i)] = df3.loc[df3['Clusters'] == i]

        # main loop from where all methods are calling
        for i in unique_list:
            data = pre_cluster[str(i)]
            ab = Counter(data['platform'])
            if len(ab.values()) > 1:
                self.score_data(data)
            else:
                pass

    def remove_image(self):
        location = os.listdir("./profile_matching_app/images")
        for i in location:
            os.remove("./profile_matching_app/images/" + i)

    # for name matching, we used jaccard similarity matching algorithm
    def name_score_data(self, query, document):
        if 'null' in query:
            return 0.0
        else:
            intersection = set(query).intersection(set(document))
            union = set(query).union(set(document))
            return len(intersection) / len(union)

    # for description matching,we used TFIDF &cosine similarity matching algorithm
    def description_score_data(self, description, remaining_description):
        if 'null' in description:
            return 0.0
        else:
            tfidf = vectorizer.fit_transform([description, remaining_description])
            result = (tfidf * tfidf.T).A[0, 1]
            return result

    # for username matching,we used w_v and cosine_similarity based algorithm:
    def username_score_data(self, v1, v2):
        if v1 is None or v2 is None:
            return float(0.0)
        else:
            seq = difflib.SequenceMatcher(isjunk=None, a=v1, b=v2)
            d = seq.ratio()
            return d

    # in case of unbalance
    def score_data(self, dataframe):
        platforms_count = Counter(dataframe['platform'])
        platforms_count_list = sorted(platforms_count.values(), reverse=True)
        platforms_count_set = list(set(platforms_count_list))
        if len(platforms_count_set) > 1:
            second_large = platforms_count_set[-2]
        else:
            second_large = platforms_count_set[0]
        occurence_second_large = Counter(platforms_count_list)[second_large]
        if occurence_second_large is 1:
            for i in platforms_count:
                if platforms_count[i] == second_large:
                    target_row_platform = i
                    break
        else:
            available_target_rows = [k for k, v in platforms_count.items() if v == second_large]
            if 'Twitter' in available_target_rows:
                target_row_platform = 'Twitter'
            elif 'Instagram' in available_target_rows:
                target_row_platform = 'Instagram'
            elif 'Facebook' in available_target_rows:
                target_row_platform = 'Facebook'
            else:
                target_row_platform = 'LinkedIn'

        selected_target_variable = dataframe.loc[dataframe['platform'] == target_row_platform]
        compare_with = dataframe.loc[dataframe['platform'] != target_row_platform]

        for single_target_variable in selected_target_variable.index:
            name_score = []
            username_score = []
            description_score = []
            for i in compare_with.index:
                if 'null' in selected_target_variable['name'][single_target_variable].lower():
                    similarity_name = 0.0
                else:
                    similarity_name = self.name_score_data(
                        selected_target_variable['name'][single_target_variable].lower(),
                        compare_with['name'][i].lower())
                if 'null' in selected_target_variable['username'][single_target_variable].lower():
                    similarity_username = 0.0
                else:
                    similarity_username = self.username_score_data(
                        selected_target_variable['username'][single_target_variable].lower(),
                        compare_with['username'][i].lower())
                if 'null' in selected_target_variable['description'][single_target_variable].lower():
                    similarity_description = 0.0
                else:
                    similarity_description = self.description_score_data(
                        selected_target_variable['description'][single_target_variable].lower(),
                        compare_with['description'][i].lower())

                name_score.append(similarity_name)
                username_score.append(similarity_username)
                description_score.append(similarity_description)

            single_target = selected_target_variable.loc[selected_target_variable.index == single_target_variable]
            single_target['name_score'] = 1.0
            single_target['username_score'] = 1.0
            single_target['description_score'] = 1.0
            new_dataframe = compare_with.assign(name_score=name_score, username_score=username_score,
                                                description_score=description_score)
            complete_df = pd.concat([new_dataframe, single_target], axis=0)
            cols = ['name_score', 'username_score', 'description_score']
            complete_df['normalisation_score'] = complete_df[cols].astype(float).mean(axis=1).round(3)
            # print(complete_df)
            self.result(complete_df)

    def result(self, dataframe):
        dataframe['normalisation_score_label'] = dataframe['normalisation_score'].apply(
            lambda val: "A" if (np.isreal(val) and val >= 0.35) else val)
        dataframe = dataframe[dataframe.normalisation_score_label == "A"]
        counter_values = Counter(dataframe['normalisation_score_label'])
        counter = list(counter_values.values())[0]
        if counter > 1:

            dataframe.reset_index(drop=True, inplace=True)
            print(dataframe)
            final_json = dataframe.to_dict(orient='index')
            self.clustered_dataframe.append(final_json)
            # print(json.loads(final_json))


def main(search_profiles):
    if len(search_profiles) > 0:
        print(os.getcwd())
        ml_obj = ML()
        # ml_obj.fetch_db_data('Taiba')
        data = pd.DataFrame(search_profiles)
        # drop the profiles, not having the image
        final_data = data.dropna(subset=['image'])
        ml_obj.download_images(final_data)

        print(ml_obj.clustered_dataframe)
        return ml_obj.clustered_dataframe
    else:
        return 'No data found'


if __name__ == '__main__':

    face_detector = dlib.get_frontal_face_detector()
    pose_predictor_68_point = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    print(os.getcwd())
    # pose_predictor_68_point = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    face_encoder = dlib.face_recognition_model_v1('./dlib_face_recognition_resnet_model_v1.dat')
    # face_encoder = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)
    clustered_datafram = []
    obj = ML()
    obj.fetch_db_data('Taiba')
    print(clustered_datafram)
