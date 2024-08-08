import os, csv
import pandas as pd
from PIL import Image
import sqlite3
import mysql.connector as sql
from collections import defaultdict
import datasets as ds
from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split

def create_filtered_csv(path, labels, filename='CN_hpz_dataset.csv'):
    filtered_rows = []
    # Search for description rows with wanted labels inside 
    with open(path, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            for label in labels:
                if (label in row[1]) or (label in row[2]):
                    filtered_rows.append(row + [label])
    # Write new csv file containing only the description with the desired labels
    with open(os.path.join('datasets', 'CN_coin_descriptions', filename), 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in filtered_rows:
            csvwriter.writerow(row)


def create_img_caption_csv(img_class_superpath, coin_desc_path, filename='new.csv'):
    # Get the labels and their description out of coin desc csv
    img_desc_pairs = []
    with open(coin_desc_path, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        # Every row should have the coin id, two descriptions and a class identified inside one of them
        for row in csvreader:
            coin_id, desc1, desc2, label = row
            desc = desc1 if label in desc1 else desc2 # The description paired with corresponding image path
            img_class_path = os.path.join(img_class_superpath, label) # Path leading to label-specific folder
            img_files = os.listdir(img_class_path) 
            for img_file in img_files: 
                if f"_{coin_id}_" in img_file: # If the image with the given coin id is found
                    img_path = os.path.join(img_class_path, img_file)
                    img_desc_pairs.append([img_path, desc]) # Include image path + description pairing

    # Write csv file for coin dataset
    with open(os.path.join('datasets', 'CN_coin_descriptions', filename), 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in img_desc_pairs:
            csvwriter.writerow(row)

def create_img_caption_label_csv(img_class_superpath, coin_desc_path, filename='new.csv'):
    # Get the labels and their description out of coin desc csv
    img_desc_pairs = []
    with open(coin_desc_path, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        # Every row should have the coin id, two descriptions and a class identified inside one of them
        for row in csvreader:
            coin_id, desc1, desc2, label = row
            desc = desc1 if label in desc1 else desc2 # The description paired with corresponding image path
            img_class_path = os.path.join(img_class_superpath, label) # Path leading to label-specific folder
            img_files = os.listdir(img_class_path) 
            for img_file in img_files: 
                if f"_{coin_id}_" in img_file: # If the image with the given coin id is found
                    img_path = os.path.join(img_class_path, img_file)
                    img_desc_pairs.append([img_path, desc, label]) # Include image path + description pairing

    # Write csv file for coin dataset
    with open(os.path.join('datasets', 'CN_coin_descriptions', filename), 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in img_desc_pairs:
            csvwriter.writerow(row)

def img_desc_pairs_from_nlp_sql_db(labels, img_class_superpath, \
                                   csv_exportfile='CN_whole_dataset.csv', \
                                   user='root', pwd='s77km7A.5O_0', \
                                   db='rootdb', host='localhost', port='3306'):
    conn = sql.connect(host=host, port=port, user=user, password=pwd, database=db)
    cursor = conn.cursor()
    #comm = "SELECT data_types.id FROM data_types JOIN data_designs ON data_types.id_design_r = data_designs.id WHERE data_designs.design_en = 'Altar, lighted and garlanded.';"
    #comm = "SELECT design_en FROM data_designs WHERE design_en LIKE '%zeus%' OR design_en LIKE '%Zeus%';"
    img_paths = []
    img_desc = {}
    img_label = defaultdict(lambda: [])
    # Set query command base to build upon for every label, results yield coin type id and english description of said coin 
    comm_base = "SELECT data_types.id, data_designs.design_en FROM data_types JOIN data_designs ON data_types.id_design_r = data_designs.id WHERE "
    for label in labels:
        # Add labels to command base
        comm = comm_base + "data_designs.design_en LIKE '%" + label + "%';"
        cursor.execute(comm)

        # Select all image files for given label and use results from query
        img_class_path = os.path.join(img_class_superpath, label)
        img_files = os.listdir(img_class_path)
        for type_id, desc in cursor.fetchall():
            for img_file in img_files:
                # If an image starts with the current type id given in the query result
                if f"CN_type_{type_id}" in img_file:
                    img_path = os.path.join(img_class_path, img_file)
                    if not img_path in img_paths: img_paths.append(img_path)
                    img_desc[img_path] = desc
                    if not label in img_label[img_path]: img_label[img_path].append(label)             
    conn.close()

    data = [(img, img_label[img], img_desc[img]) for img in img_paths]
        
    
    # Write csv file for coin dataset
    with open(os.path.join('datasets', 'CN_coin_descriptions', csv_exportfile), 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in data:
            csvwriter.writerow(row)

def csv_train_eval_split(original_path, train_partition=0.8):
    train_data, eval_data = [("image_path", "label", "caption")], [("image_path", "label", "caption")]
    if 0.0 < train_partition < 1.0:
        number_data, train_number = 0, 0
        with open(original_path, newline='', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader: number_data += 1
            train_number = int(train_partition * number_data)
            
        with open(original_path, newline='', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                if train_number > 0:
                    train_data.append(row)
                    train_number -= 1
                else:
                    eval_data.append(row)

    orig_path_wo_end = original_path[:-4]
    with open(orig_path_wo_end + "_train.csv", 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in train_data:
            csvwriter.writerow(row)
            
    with open(orig_path_wo_end + "_eval.csv", 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in eval_data:
            csvwriter.writerow(row)

def change_formatting(path):
    rows = [("image_path", "caption")]
    with open(path, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            #print(row)
            #print(row[1].replace(',','&'))
            rows.append((row[0], row[1].replace(',',',')))
            #print(rows)
            #return
    with open(path[:-4] + "_reformat.csv", 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in rows:
            csvwriter.writerow(row)

def add_column_names(path):
    rows = [("image_path", "label", "caption")]
    with open(path, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            rows.append(row)
    with open(path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in rows:
            csvwriter.writerow(row)
    
def dataloader_from_nlp_sql_db(img_class_superpath, labels=None, train_eval_split=0.8, \
                               export_path = os.path.join("datasets", "obj_det", "all"), \
                               user='root', pwd='s77km7A.5O_0', \
                               db='rootdb', host='localhost', port='3306'):
    try:
        if labels is None: labels = os.listdir(img_class_superpath)
        conn = sql.connect(host=host, port=port, user=user, password=pwd, database=db)
        cursor = conn.cursor()
        #comm = "SELECT data_types.id FROM data_types JOIN data_designs ON data_types.id_design_r = data_designs.id WHERE data_designs.design_en = 'Altar, lighted and garlanded.';"
        #comm = "SELECT design_en FROM data_designs WHERE design_en LIKE '%zeus%' OR design_en LIKE '%Zeus%';"
        img_paths = []
        img_desc = {}
        img_label = defaultdict(lambda: [])
        # Set query command base to build upon for every label, results yield coin type id and english description of said coin 
        comm_base = "SELECT data_types.id, data_designs.design_en FROM data_types JOIN data_designs ON data_types.id_design_r = data_designs.id WHERE "
        for label in labels:
            # Add labels to command base
            comm = comm_base + "data_designs.design_en LIKE '%" + label + "%';"
            cursor.execute(comm)

            # Select all image files for given label and use results from query
            img_class_path = os.path.join(img_class_superpath, label)
            img_files = os.listdir(img_class_path)
            for type_id, desc in cursor.fetchall():
                for img_file in img_files:
                    # If an image starts with the current type id given in the query result
                    if f"CN_type_{type_id}" in img_file:
                        img_path = os.path.join(img_class_path, img_file)
                        if not img_path in img_paths: img_paths.append(img_path)
                        img_desc[img_path] = desc
                        if not label in img_label[img_path]: img_label[img_path].append(label)                   
        conn.close()
        #images = [Image.open(img).convert('RGB') for img in img_paths]
        #descs  = [img_desc[img] for img in img_paths]
        #labels = [img_label[img] for img in img_paths]
        #imgs_descs_labels = {'image': images, 'caption': descs, 'label': labels}
        print(img_paths[:10])
        imgs_descs_labels = [[img, img_label[img], img_desc[img]] for img in img_paths]
        data = pd.DataFrame(imgs_descs_labels, columns=('image_path', 'label', 'caption'))
        train_data, test_data = train_test_split(data, train_size=train_eval_split)
        ds_dict = DatasetDict({'train': Dataset.from_pandas(train_data), 'eval': Dataset.from_pandas(test_data)})
        def add_images(sample):
            return {'image': Image.open(sample['image_path']).convert('RGB')}
        ds_dict = ds_dict.map(add_images)
        ds_dict.save_to_disk(export_path)
        
    except Exception as e:
        print(e)
        input("")
        
img_superpath = os.path.join("datasets", "CN_dataset_obj_detection_04_23", "dataset_obj_detection")
#create_filtered_csv('CN_coin_descriptions.csv', ['Zeus', 'Poseidon', 'Hades', 'zeus', 'poseidon', 'hades'])
#create_img_caption_csv(os.path.join("CN_dataset_obj_detection_04_23", "dataset_obj_detection"), \
#                       os.path.join('CN_coin_descriptions', 'CN_hpz_dataset.csv'), filename='hpz_dataset.csv')
#img_desc_pairs_from_nlp_sql_db(os.listdir(img_superpath), os.path.join("datasets", "CN_dataset_obj_detection_04_23", "dataset_obj_detection"))
#change_formatting(os.path.join('CN_coin_descriptions', 'CN_hpz_dataset.csv'))
csv_train_eval_split(os.path.join('datasets', 'CN_coin_descriptions', 'CN_whole_dataset.csv'))
#dataloader_from_nlp_sql_db(img_superpath) # ['hades', 'poseidon', 'zeus']
