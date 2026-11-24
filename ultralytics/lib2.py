#lib2.py

import itertools

import cv2

import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.decomposition import PCA
import os
import math

import lib

from sklearn.preprocessing import LabelEncoder

from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import queue

import pandas as pd
import matplotlib.pyplot as plt


_project_directory = os.path.dirname(os.path.abspath(__file__))
_tmp_pdp = os.path.join("_tmp_pdp.jpg")
_empty_image = os.path.join(_project_directory,'_empty.jpg')


#====def boost_training(out_k_fold_path) NISE+PCA only=====
def concat_images_horizontally_memory(image_paths, is_add_pdp=True, pdp_size=10, is_add_pdp_inside=False):
    image_list = []
    if (is_add_pdp):
        for img1 in image_paths:
            if is_add_pdp_inside:
                add_qr_pdp_to_small_image_inside(img1, _tmp_pdp, pdp_size)
            else:
                add_qr_pdp_to_small_image(img1, _tmp_pdp, pdp_size)
            image_list.append(Image.open(_tmp_pdp))
    else:
        image_list = [Image.open(path) for path in image_paths]

    base_height = image_list[0].height
    resized = []
    for img in image_list:
        if img.height != base_height:
            new_width = int(img.width * base_height / img.height)
            img = img.resize((new_width, base_height))
        resized.append(img.convert("RGB"))

    # arrays = []
    # for idx, img in enumerate(resized):
    #     arrays.append(np.array(img))
    #     if idx < len(resized) - 1:
    #         spacer = np.ones((base_height, space, 3), dtype=np.uint8) * 255
    #         arrays.append(spacer)

    arrays = [np.array(img) for img in resized]
    concatenated = np.concatenate(arrays, axis=1)
    return Image.fromarray(concatenated)


def concat_images_vertically_memory(image_narray_list):
    base_width = image_narray_list[0].width
    resized = []
    for img in image_narray_list:
        if img.width != base_width:
            new_height = int(img.height * base_width / img.width)
            img = img.resize((base_width, new_height))
        resized.append(img.convert("RGB"))

    # arrays = []
    # for idx, img in enumerate(resized):
    #     arrays.append(np.array(img))
    #     if idx < len(resized) - 1:
    #         spacer = np.ones((base_width, space, 3), dtype=np.uint8) * 255
    #         arrays.append(spacer)

    arrays = [np.array(img) for img in resized]
    concatenated = np.concatenate(arrays, axis=0)
    return Image.fromarray(concatenated)

def concat_images_horizontally(image_paths, output_path):
    images = [Image.open(path) for path in image_paths]

    base_height = images[0].height
    resized_images = []
    for img in images:
        if img.height != base_height:
            new_width = int(img.width * base_height / img.height)
            img = img.resize((new_width, base_height))
        resized_images.append(img)

    arrays = [np.array(img) for img in resized_images]

    concatenated = np.concatenate(arrays, axis=1)


    result_img = Image.fromarray(concatenated)
    result_img.save(output_path)

def concat_images_vertically(image_paths, output_path):
    images = [Image.open(path) for path in image_paths]

    base_width = images[0].width
    resized_images = []
    for img in images:
        if img.width != base_width:
            new_height = int(img.height * base_width / img.width)
            img = img.resize((base_width, new_height))
        resized_images.append(img)

    arrays = [np.array(img) for img in resized_images]

    concatenated = np.concatenate(arrays, axis=0)

    result_img = Image.fromarray(concatenated)
    result_img.save(output_path)


def list_to_square_2d(input_list):
    length = len(input_list)
    side = int(math.sqrt(length))

    if side * side != length:
        raise ValueError("List length must be a perfect square.")

    return [input_list[i * side:(i + 1) * side] for i in range(side)]


def sqrt_and_ceil(x):
    return math.ceil(math.sqrt(x))

def create_boost_training_images(project_directory, out_k_fold_path, is_add_pdp=False, is_add_pdp_inside=False, pdp_size=10):
    # create empty image
    width, height = 40, 40
    white_image = Image.new('RGB', (width, height), color='white')
    white_image.save(_empty_image, format='JPEG')

    # 640x640,1 strip 40x40 --> 409600 /40x40 = 256 --> square image (16x16)
    # 640x640,1 strip 20x20 --> 409600 /20x20 = 1024 --> square image (32x32)
    if(is_add_pdp):
        # chunk_size = 128
        chunk_size = 256
    else:
        # chunk_size = 1024
        chunk_size = 256



    out_k_fold_training_boost_path = os.path.join(project_directory, out_k_fold_path+"_boost")
    if os.path.exists(out_k_fold_training_boost_path):
        shutil.rmtree(out_k_fold_training_boost_path)
    os.makedirs(out_k_fold_training_boost_path)

    list_folder = ["training", "validation"]
    # list_folder = ["training"]

    folder_fold_nos = [f for f in os.listdir(os.path.join(project_directory, out_k_fold_path))]
    for folder_fold_no in folder_fold_nos:

        if os.path.exists( os.path.join(out_k_fold_training_boost_path, folder_fold_no) ):
            shutil.rmtree(os.path.join(out_k_fold_training_boost_path, folder_fold_no))
        os.makedirs(os.path.join(out_k_fold_training_boost_path, folder_fold_no))

        for folder1 in list_folder:
            if os.path.exists(os.path.join(out_k_fold_training_boost_path, folder_fold_no, folder1)):
                shutil.rmtree(os.path.join(out_k_fold_training_boost_path, folder_fold_no, folder1))
            os.makedirs(os.path.join(out_k_fold_training_boost_path, folder_fold_no, folder1))

            os.makedirs(os.path.join(out_k_fold_training_boost_path, folder_fold_no, folder1, "images"))
            os.makedirs(os.path.join(out_k_fold_training_boost_path, folder_fold_no, folder1, "labels"))



        # merge training images
        for folder1 in list_folder:

            all_images = load_image_files_from_folder( os.path.join(project_directory, out_k_fold_path, folder_fold_no, folder1, "images") )

            # label_file_path = os.path.join(project_directory, out_k_fold_path, folder1, "labels")

            # all_images = []
            # for img1 in all_imagesx:
            #     all_images.append(Image.open( os.path.join(project_directory, out_k_fold_path, folder_fold_no, folder1, "images", img1) ))

            # 640x640,1 strip 40x40 --> 409600 /40x40 = 256
            # 640x640,1 strip 20x20 --> 409600 /20x20 = 1024
            set_images, remainder = split_img_file_list(all_images, chunk_size)

            create_labels(set_images,
                          project_directory, out_k_fold_path, folder_fold_no,
                          folder1, out_k_fold_training_boost_path)

            create_images(set_images, pdp_size,
                             project_directory, folder_fold_no, folder1,
                             out_k_fold_training_boost_path,
                             is_add_pdp, chunk_size, is_add_pdp_inside)





        #create dataset.yaml
        with open(os.path.join(project_directory, out_k_fold_path, folder_fold_no, 'dataset.yaml'),
                  'r', encoding='utf-8') as f:
            yaml = f.read()
            parts = yaml.split('nc:')

            yaml_new = ''

            yaml_new = yaml_new + "train: " + os.path.join(out_k_fold_training_boost_path, folder_fold_no,"training") +"\n"
            yaml_new = yaml_new + "validation: " + os.path.join(out_k_fold_training_boost_path, folder_fold_no,"validation") + "\n"

            yaml_new = yaml_new + "nc:"+ parts[1]

        with open(os.path.join(out_k_fold_training_boost_path, folder_fold_no, 'dataset.yaml'),
                  'w', encoding='utf-8') as f:
            f.write(yaml_new.strip())



        #test
        list_pdp_create = ["test"]
        # list_pdp_create = ["test", "validation"]
        for pdp_folder in list_pdp_create:
            if os.path.exists(os.path.join(out_k_fold_training_boost_path, folder_fold_no, pdp_folder)):
                os.remove(os.path.join(out_k_fold_training_boost_path, folder_fold_no, pdp_folder))
                # lib.remove_and_create_folder(os.path.join(out_k_fold_training_boost_path, folder_fold_no,"test"))

            src_folder = os.path.join(out_k_fold_path, folder_fold_no, pdp_folder)
            dst_folder = os.path.join(out_k_fold_training_boost_path, folder_fold_no, pdp_folder)
            shutil.copytree(src_folder, dst_folder)

            # create pdp
            print("PDP create test image")

            if (is_add_pdp):
                # image_paths_test = [os.path.join(dst_folder, "images", f) for f in os.listdir(dst_folder)
                #                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                image_folder = os.path.join(dst_folder, "images")
                label_folder = os.path.join(dst_folder, "labels")
                print(image_folder)
                image_files_test = [f for f in os.listdir(image_folder) if
                                    f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

                for img1_test in image_files_test:
                    img_full_path = os.path.join(image_folder, img1_test)

                    name_without_extension = os.path.splitext(os.path.basename(img_full_path))[0]
                    label_test_path = os.path.join(label_folder, name_without_extension + ".txt")
                    with open(label_test_path,
                              'r', encoding='utf-8') as f:
                        test_label = f.read()
                    test_label = test_label.split(' ')[0]



                    if is_add_pdp_inside:
                        add_qr_pdp_to_small_image_inside(img_full_path, img_full_path, pdp_size)
                        attatch_test(img_full_path, label_test_path, test_label)

                    else:
                        add_qr_pdp_to_small_image(img_full_path, img_full_path, pdp_size)
                        attatch_test(img_full_path, label_test_path, test_label)


def resize_images_in_directory(input_file, output_file, width, height):


    img = cv2.imread(input_file)

    resized_img = cv2.resize(img, (width, height))
    cv2.imwrite(output_file, resized_img)

def create_labels(set_images,
                  project_directory, out_k_fold_path, folder_fold_no,
                  folder1, out_k_fold_training_boost_path):

    # 00=move x,y , 01= y , 10= x ,11 = not move
    start_offsets = [[0, 0], [0, 1], [1, 0], [1, 1]]

    count = 0
    for images in set_images:
        image_files = [f for f in images if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        for start_offset in start_offsets:

            # 2. create labels for image

            norm_i = 0.03125
            norm_j = 0.03125

            norm_step = 0.03125 * 2
            width_height = 0.03125 * 2

            count_img = 0
            count_line = 0
            str_labels = ""

            for image_file_name in image_files:

                name_without_extension = os.path.splitext(os.path.basename(image_file_name))[0]
                if name_without_extension != '_empty':
                    with open(os.path.join(project_directory, out_k_fold_path, folder_fold_no, folder1, "labels",
                                           name_without_extension + '.txt'), 'r', encoding='utf-8') as file:
                        first_char_is_label = file.read(1)

                    if norm_i > 0.96875:
                        count_img = 0
                        count_line = count_line + 1

                        norm_i = 0.03125
                        norm_j = norm_j + norm_step

                    is_add_line = False
                    is_add_h = False

                    # 00=move x,y , 01= y , 10= x ,11 = not move

                    if start_offset[0] == 0 and start_offset[1] == 0:
                        if count_img % 2 == 1:  #
                            is_add_h = True
                        if count_line % 2 == 1:  #
                            is_add_line = True
                    elif start_offset[0] == 0 and start_offset[1] == 1:
                        if count_img % 2 == 0:  #
                            is_add_h = True
                        if count_line % 2 == 1:  #
                            is_add_line = True
                    elif start_offset[0] == 1 and start_offset[1] == 0:
                        if count_img % 2 == 1:  #
                            is_add_h = True
                        if count_line % 2 == 0:  #
                            is_add_line = True
                    elif start_offset[0] == 1 and start_offset[1] == 1:
                        if count_img % 2 == 0:  #
                            is_add_h = True
                        if count_line % 2 == 0:  #
                            is_add_line = True

                    if is_add_h and is_add_line:
                        str_labels = str_labels + first_char_is_label + " " + str(norm_i) + " " + str(
                            norm_j) + " " + str(width_height) + " " + str(width_height) + "\n"

                    norm_i = norm_i + norm_step

                    count_img = count_img + 1
                else:
                    break

            with open(os.path.join(out_k_fold_training_boost_path, folder_fold_no, folder1, "labels",
                                   str(start_offset[0])+str(start_offset[1]) + "_" + str(count) + '.txt'), 'w', encoding='utf-8') as f:
                f.write(str_labels.strip())
        count = count + 1



def create_images(set_images, pdp_size,
                             project_directory, folder_fold_no, folder1,
                             out_k_fold_training_boost_path,
                             is_add_pdp, chunk_size, is_add_pdp_inside):


    # --create image--

    count = 0
    for images in set_images:

        # 1. create image multi labels
        # make it to square
        lack_images = chunk_size - len(images)
        if (lack_images != 0):
            for add_image in range(lack_images):
                images.append(_empty_image)

        # 1024 perfect square but 512 (square root(512) = 22....) not perfect square-->(23x23)529
        square_img_list = list_to_square_2d(images)

        list_of_vertical_img = []
        for line_img in square_img_list:
            img_array = concat_images_horizontally_memory(line_img, is_add_pdp, pdp_size, is_add_pdp_inside)
            list_of_vertical_img.append(img_array)

        # img_concate.save(os.path.join(project_directory, out_k_fold_training_boost_path, folder_fold_no, folder1, "images", str(count) + '.jpg'))
        final_image = concat_images_vertically_memory(list_of_vertical_img)
        final_image.save(
            os.path.join(project_directory, out_k_fold_training_boost_path, folder_fold_no, folder1, "images",
                         str(count) + '.jpg'), format="JPEG")

        stride40(os.path.join(project_directory, out_k_fold_training_boost_path, folder_fold_no, folder1, "images",
                         str(count) + '.jpg'),
                 os.path.join(project_directory, out_k_fold_training_boost_path, folder_fold_no, folder1, "images"),
                 str(count))

        os.remove(os.path.join(project_directory, out_k_fold_training_boost_path, folder_fold_no, folder1, "images",
                         str(count) + '.jpg'))

        count = count + 1




def stride40(main_img_path, out_dir, file_name_with_out_ext):
    main_img = Image.open(main_img_path)
    empty_h = Image.open(_empty_image).resize((640, 40))
    empty_v = Image.open(_empty_image).resize((40, 640))

    main_img = main_img.resize((640, 640))
    stride = 80

    start_offsets = [[0, 0], [0, 1], [1, 0], [1, 1]]

    for start_offset in start_offsets:
        canvas = Image.new("RGB", (640, 640), (255, 255, 255))
        canvas.paste(main_img, (0, 0))

        y = 0 if start_offset[0] == 0 else 40
        while y < 640:
            canvas.paste(empty_h, (0, y))
            y += stride

        x = 0 if start_offset[1] == 0 else 40
        while x < 640:
            canvas.paste(empty_v, (x, 0))
            x += stride

        out_file = f"{start_offset[0]}{start_offset[1]}_{file_name_with_out_ext}.jpg"
        canvas.save(os.path.join(out_dir, out_file))

def load_image_files_from_folder(folder_path):

    image_file_paths = [f for f in os.listdir(folder_path)]
    images = []
    for file in image_file_paths:
        images.append( os.path.join(folder_path, file) )

    return images


#640x640,1 strip 20x20 --> 409600 /20x20 = 1024
# def split_img_file_list(lst_images, chunk_size=1024): #image input size 640x640
def split_img_file_list(lst_images, chunk_size=256):  # image input size 640x640

    q = queue.Queue()

    for img in lst_images:
        q.put(img)

    all_training_images = len(lst_images)

    set_no = all_training_images//chunk_size
    last_set_images = all_training_images % chunk_size
    remainder = chunk_size - last_set_images

    set_images = []

    for i in range(set_no):
        images = []

        for j in range(chunk_size):
            images.append(q.get())

        set_images.append(images)

    images = []
    while not q.empty():
        images.append(q.get())

    set_images.append(images)

    # img_fill_empty = lib.generate_strip_color(255, 255, 255, 20, 20)
    return set_images, remainder



# Missing Completely At Random (MCAR)
# approximation missing 20%
def mcar(df, missing_percent = 0.2): #Cell-wise Missingness
    np.random.seed(42)
    #*df.shape unpack --> rows, columns
    mask = np.random.rand(*df.shape) < missing_percent
    df[mask] = np.nan

    return df

def mcar_exact(df, missing_percent=0.2):
    np.random.seed(42)

    total_cells = df.size
    num_missing = int(total_cells * missing_percent)

    all_indices = np.arange(total_cells) #np.arange(5) --> [0 1 2 3 4]
    missing_indices = np.random.choice(all_indices, size=num_missing, replace=False) #False = no repeat data

    row_indices, col_indices = np.unravel_index(missing_indices, df.shape) # make index to row,col

    df_missing = df.copy()
    df_missing.values[row_indices, col_indices] = np.nan

    return df_missing

def row_wise_mcar(df, missing_percent=0.2):
    np.random.seed(42)
    num_rows = int(df.shape[0] * missing_percent)
    missing_rows = np.random.choice(df.index, size=num_rows, replace=False) #replace=False random not same row

    for row in missing_rows:
        col_to_nan = np.random.choice(df.columns)
        df.loc[row, col_to_nan] = np.nan

    return df

def impute_mean(df, missing_values_map={"": np.nan, "?": np.nan}):
    temp_df = df.copy()

    if missing_values_map:
        temp_df.replace(missing_values_map, inplace=True)

    for feature in temp_df.columns:
        temp_df[feature] = temp_df[feature].fillna(temp_df[feature].mean())

    return temp_df

def impute_knn(df, missing_values_map={"": np.nan, "?": np.nan}, n_neighbors=5):
    temp_df = df.copy()

    if missing_values_map:
        temp_df.replace(missing_values_map, inplace=True)

    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = pd.DataFrame(imputer.fit_transform(temp_df), columns=temp_df.columns, index=temp_df.index)

    return df_imputed


def impute_interpolation_linear(df, missing_values_map={"": np.nan, "?": np.nan}):
    temp_df = df.copy()

    if missing_values_map:
        temp_df.replace(missing_values_map, inplace=True)

    temp_df['Linear_Interpolated'] = temp_df['Value'].interpolate(method='linear')

    return temp_df

def impute_interpolation_poly(df, missing_values_map={"": np.nan, "?": np.nan}, poly_order=2):
    temp_df = df.copy()

    if missing_values_map:
        temp_df.replace(missing_values_map, inplace=True)

    temp_df['Polynomial_Interpolated'] = temp_df['Value'].interpolate(method='polynomial', order=poly_order)

    return temp_df

def impute_interpolation_cubic_spline(df, column, missing_values_map={"": np.nan, "?": np.nan}):
    temp_df = df.copy()

    if missing_values_map:
        temp_df.replace(missing_values_map, inplace=True)

    temp_df[column] = temp_df[column].interpolate(method='spline', order=3)

    return temp_df

def impute_linear_regression(df, missing_values_map={"": np.nan, "?": np.nan}):
    temp_df = df.copy()

    if missing_values_map:
        temp_df.replace(missing_values_map, inplace=True) #replace with nan


    feature_columns = temp_df.columns

    for feature in feature_columns:
        predictors = [col for col in df.columns if col != feature]

        df_train = df.dropna(subset=[feature])
        #masking nan temp_df[temp_df[feature].isna()]
        df_missing = temp_df[temp_df[feature].isna()] #rows have missing

        X_train = df_train[predictors]
        y_train = df_train[feature]

        X_missing = df_missing[predictors]

        model = LinearRegression()
        model.fit(X_train, y_train)

        #each predict column  df.loc[rows, columns]
        df.loc[df[feature].isna(), feature] = model.predict(X_missing)

    return df





def csv_to_pca_csv(project_directory, csv, label_column):
    # Load the CSV file
    df = pd.read_csv(csv)
    # Separating out the features and the label
    X = df.drop(columns=[label_column])
    y = df[label_column]

    X = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=2)  # Adjust n_components as needed
    principal_components = pca.fit_transform(X)

    # Convert the principal components to a DataFrame
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    final_df = pd.concat([principal_df, y], axis=1)
    final_df.to_csv(os.path.join(project_directory, 'pca_result.csv'), index=False)
    return os.path.join(project_directory, 'pca_result.csv')


def load_wine(project_directory):
    path_csv = os.path.join(project_directory, "wine.csv")
    label_header = "Wine"
    df = pd.read_csv(path_csv)

    header_name = df.columns.tolist()
    header_name.remove(label_header)

    # pca
    csv_pca = csv_to_pca_csv(project_directory, path_csv, label_header)
    header_name_pca = ["PC1", "PC2"]
    return path_csv, header_name, label_header, csv_pca, header_name_pca



def load_header_iris(project_directory):
    path_csv = os.path.join(project_directory, "iris.csv")
    label_header = "variety"

    df = pd.read_csv(path_csv)
    header_name = df.columns.tolist()
    header_name.remove(label_header)

    # pca
    csv_pca = csv_to_pca_csv(project_directory, path_csv, label_header)
    header_name_pca = ["PC1", "PC2"]
    return path_csv, header_name, label_header, csv_pca, header_name_pca


def load_abalone(project_directory, imputation='knn'):
    path_csv = os.path.join(project_directory, "abalone.csv")
    label_header = "Sex"
    df = pd.read_csv(path_csv)

    encoder = LabelEncoder()
    df[label_header] = encoder.fit_transform(df[label_header])

    if imputation=='linear_regression':
        df = impute_linear_regression(df)
    elif imputation=='interpolation_linear':
        df = impute_interpolation_linear(df)
    elif imputation == 'interpolation_poly':
        df = impute_interpolation_poly(df)
    elif imputation == 'interpolation_cubic_spline':
        df = impute_interpolation_cubic_spline(df)
    elif imputation=='knn':
        df = impute_knn(df)

    header_name = df.columns.tolist()
    header_name.remove(label_header)

    # pca
    csv_pca = csv_to_pca_csv(project_directory, path_csv, label_header)
    header_name_pca = ["PC1", "PC2"]
    return path_csv, header_name, label_header, csv_pca, header_name_pca


def load_raisin(project_directory):
    path_csv = os.path.join(project_directory, "Raisin_Dataset.csv")
    label_header = "Class"
    df = pd.read_csv(path_csv)
    header_name = df.columns.tolist()
    header_name.remove(label_header)

    return path_csv, header_name, label_header


def load_header_breast_cancer(project_directory):
    path_csv = os.path.join(project_directory, "breast-cancer-wisconsin1.csv")
    label_header = "diagnosis"
    df = pd.read_csv(path_csv)
    header_name = df.columns.tolist()
    header_name.remove(label_header)
    header_name.remove("id")

    return path_csv, header_name, label_header

#QRcodePosition Detection Patterns input 20x20 --> output 60x60
#(left + img + right) + 30+20+30 = 80 px = (top + img +  bot)
def create_position_detection_pattern(size=10):
    pattern = np.ones((size, size), dtype=np.uint8) * 255

    if size >= 7:
        cv2.rectangle(pattern, (0, 0), (size - 1, size - 1), 0, thickness=1)
    if size >= 7:
        cv2.rectangle(pattern, (2, 2), (size - 3, size - 3), 255, thickness=1)
    if size >= 7:
        cv2.rectangle(pattern, (4, 4), (size - 5, size - 5), 0, thickness=-1)

    return pattern

def overlay_pdp(image, pdp, position):
    h, w = pdp.shape
    img_h, img_w = image.shape[:2]
    pdp_color = cv2.cvtColor(pdp, cv2.COLOR_GRAY2BGR)

    if position == 'top-left':
        image[0:h, 0:w] = pdp_color
    elif position == 'top-right':
        image[0:h, img_w-w:img_w] = pdp_color
    elif position == 'bottom-left':
        image[img_h-h:img_h, 0:w] = pdp_color

    return image

def expand_canvas_for_pdp(image, pdp_size=10):
    top = left = bottom = right = pdp_size
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255,255,255))


def add_qr_pdp_to_small_image(image_path, output_path, pdp_size=10):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (20, 20))
    image = expand_canvas_for_pdp(image, pdp_size)

    pdp = create_position_detection_pattern(size=pdp_size)
    image = overlay_pdp(image, pdp, 'top-left')
    image = overlay_pdp(image, pdp, 'top-right')
    image = overlay_pdp(image, pdp, 'bottom-left')



    cv2.imwrite(output_path, image)


def overlay_pdp_inside(image, pdp, position):
    h, w = pdp.shape
    img_h, img_w = image.shape[:2]

    if len(pdp.shape) == 2:
        pdp_color = cv2.cvtColor(pdp, cv2.COLOR_GRAY2BGR)
    else:
        pdp_color = pdp

    if position == 'top-left':
        x, y = 0, 0
    elif position == 'top-right':
        x, y = img_w - w, 0
    elif position == 'bottom-left':
        x, y = 0, img_h - h
    else:
        return image

    if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
        print(f"PDP too large for position {position}")
        return image

    #white insert
    image[y:y + h, x:x + w] = (255, 255, 255)
    #PDP
    image[y:y + h, x:x + w] = pdp_color

    return image



def add_qr_pdp_to_small_image_inside(image_path, output_path, pdp_size=10):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (40, 40))

    pdp = create_position_detection_pattern(size=pdp_size)
    image = overlay_pdp_inside(image, pdp, 'top-left')
    image = overlay_pdp_inside(image, pdp, 'top-right')
    image = overlay_pdp_inside(image, pdp, 'bottom-left')

    resized_img = cv2.resize(image, (40, 40))

    cv2.imwrite(output_path, resized_img)



def find_corner_pdp_fast(contours):
    pdps = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if abs(w - h) < 5 and w < 50 and h < 50:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    pdps.append((cx, cy))
    return pdps

def group_triplets_greedy(pdps, distance_thresh=50):
    used = set()
    triplets = []

    for i, pt in enumerate(pdps):
        if i in used:
            continue

        nearby = []
        for j, other in enumerate(pdps):
            if i != j and j not in used:
                dist = np.linalg.norm(np.array(pt) - np.array(other))
                if dist < distance_thresh:
                    nearby.append((j, dist))

        if len(nearby) >= 2:
            nearby = sorted(nearby, key=lambda x: x[1])[:2]
            idxs = [i] + [j for j, _ in nearby]
            triplets.append([pdps[idx] for idx in idxs])
            used.update(idxs)

    return triplets

def detect_pdp_with_template(gray_image, pdp_template, threshold=0.8):
    result = cv2.matchTemplate(gray_image, pdp_template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= threshold)

    pdp_w, pdp_h = pdp_template.shape[::-1]
    centers = []
    for (x, y) in zip(*loc[::-1]):
        cx = x + pdp_w // 2
        cy = y + pdp_h // 2
        centers.append((cx, cy))  # center ของ PDP
    return centers

def detect_qr_by_corner_pdp_fast(image_path, output_dir):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    pdp_template = create_position_detection_pattern(size=10)
    pdps = detect_pdp_with_template(gray, pdp_template)


    triplets = group_triplets_greedy(pdps, distance_thresh=55)

    found = 0
    for triplet in triplets:
        xs = [pt[0] for pt in triplet]
        ys = [pt[1] for pt in triplet]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        pad = 2
        crop = image[max(0, min_y - pad): max_y + pad, max(0, min_x - pad): max_x + pad]
        output_file = os.path.join(output_dir, f"qr_crop_{found + 1}.png")
        cv2.imwrite(output_file, crop)
        print(f"Saved: {output_file}")
        found += 1

    if found == 0:
        print("No valid triplets found")


def attatch_test(input_img_path, label_path, class_id):
    input_img = Image.open(input_img_path)
    w_in, h_in = input_img.size

    canvas_size = 640
    canvas = Image.new("RGB", (canvas_size, canvas_size), "white")

    x_offset = (canvas_size - w_in) // 2
    y_offset = (canvas_size - h_in) // 2

    canvas.paste(input_img, (x_offset, y_offset))
    canvas.save(input_img_path)

    x_center = (x_offset + w_in / 2) / canvas_size
    y_center = (y_offset + h_in / 2) / canvas_size
    width = w_in / canvas_size
    height = h_in / canvas_size

    label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    os.remove(label_path)

    with open(label_path, "w") as f:
        f.write(label_line)


#mobilenet
def create_mobilenet_train_val_test(project_directory, output_k_folds):

    output_k_folds_mobilenet_target_dir = os.path.join(project_directory, output_k_folds + "_mobilenet")
    print("Creating:", output_k_folds_mobilenet_target_dir)

    if os.path.exists(output_k_folds_mobilenet_target_dir):
        shutil.rmtree(output_k_folds_mobilenet_target_dir)
    os.makedirs(output_k_folds_mobilenet_target_dir)

    for fold in os.listdir(output_k_folds):
        full_path = os.path.join(output_k_folds, fold)
        if not os.path.isdir(full_path):
            continue

        folder_name = os.path.basename(full_path)
        target = os.path.join(output_k_folds_mobilenet_target_dir, folder_name)

        if os.path.exists(target):
            shutil.rmtree(target)
        os.makedirs(target)

        training_validation_test = [
            f1 for f1 in os.listdir(full_path)
            if os.path.isdir(os.path.join(full_path, f1))
        ]

        for yolo_data in training_validation_test:
            yolo_data_path = os.path.join(full_path, yolo_data)
            source_images_dir = os.path.join(yolo_data_path, "images")
            source_labels_dir = os.path.join(yolo_data_path, "labels")

            target_subdir = os.path.join(target, yolo_data)  # เช่น train/val/test
            os.makedirs(target_subdir, exist_ok=True)

            mobilenet_label(source_images_dir, source_labels_dir, target_subdir)


def mobilenet_label(source_images_dir, source_labels_dir, target_dir):
    for label_file in os.listdir(source_labels_dir):
        if not label_file.endswith('.txt'):
            continue

        label_path = os.path.join(source_labels_dir, label_file)
        image_file = label_file.replace('.txt', '.jpg')

        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            continue

        class_id = lines[0].split()[0]

        class_dir = os.path.join(target_dir, class_id)
        os.makedirs(class_dir, exist_ok=True)

        src_img_path = os.path.join(source_images_dir, image_file)
        dst_img_path = os.path.join(class_dir, image_file)

        if os.path.exists(src_img_path):
            shutil.copy2(src_img_path, dst_img_path)
