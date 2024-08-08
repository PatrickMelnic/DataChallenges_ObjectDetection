# -*- coding: latin-1 -*-

import numpy as np
from transformers import pipeline
#from clip.transformers.examples.pytorch.contrastive_image_text import run_clip
from datasets import load_dataset
from sentence_transformers import SentenceTransformer as ST
from sentence_transformers import util
from sklearn.metrics import log_loss
from PIL import Image
from PIL import ImageDraw
import os, csv

#from owl_vit.scenic.scenic.main import main as owl_main

# Set visible graphic card of 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = "cuda" if False else "cpu"

class LocalClassImageDataset:
    def __init__(self, class_folder_path, class_labels=[]):
        # valid/existing class labels
        self.__p__   = class_folder_path #superpath for entire dataset
        # class label listing
        self.__cl__  = [c for c in class_labels if os.path.exists(os.path.join(self.__p__, c))]
        if self.__cl__ == []: # extra if no given class_labels were found
            self.__cl__ = [cp[len(self.__p__):] for cp in os.listdir(self.__p__)]

        self.__n__  = len(self.__cl__) # number of valid/existing class labels
        # paths for individual classes
        self.__cp__ = [os.path.join(self.__p__, c) for c in self.__cl__]
        self.__c__  = 0 # current class chosen
        self.__i__  = 0 # current image in current class chosen
        # image paths of c-th class
        self.change_class(0)
        
    def __img_iter__(self):
        # Perform next image iteration, then
        self.__i__ += 1
        if self.__i__ >= len(self.__ip__): self.__i__ = 0

    def get_current_class(self):
        return self.__cl__[self.__c__]

    def get_image_path(self):
        return self.__ip__[self.__i__]
    
    def get_current_image(self):
        return Image.open(self.__ip__[self.__i__])

    def change_class(self, class_label):
        if isinstance(class_label, int):
            self.__c__ = class_label if class_label > 0 else 0
        elif isinstance(class_label, str):
            try:
                self.__c__ = self.__cl__.index(class_label)
            except ValueError:
                raise ValueError(f'"{class_label}" is not an existing class')
        else:
            raise ValueError(f'"{class_label}" is not a valid class label')
        print(f"current class changed to {self.__c__}")
        self.__ip__ = [os.path.join(self.__cp__[self.__c__], path) for path in os.listdir(self.__cp__[self.__c__])]
        

def CLIP_implementation(dataset, train_classes):
    #return SentenceTransformer("clip-ViT-B-32")
    # Train on images of individual class
    print(run_clip)
    """
    for i,c in enumerate(train_classes):
        # Encode text descriptions
        text_emb = model.encode(c)

        # Get all image paths to get images
        img_paths = os.listdir(img_class_paths[i])
        for img in img_paths:
            # Encode image
            img_emb = model.encode(Image.open(img))
            
            # Compute cosine similarities
            logits = util.cos_sim(img_emb, text_emb)

            # symmetric loss function
            labels = np.arange(n)
            loss_i = log_loss(logits, labels)
            loss_t = log_loss(logits, labels)
            loss = (loss_i + loss_t)/2
    """
            

def OWL_ViT_implementation():
    return pipeline(model="google/owlv2-base-patch16-ensemble", task="zero-shot-object-detection")

"""
def generate_TFDS(image_class_path):
        builder = tfds.folder_dataset.ImageFolder(image_class_path)
        print(builder.info)  # num examples, labels... are automatically calculated
        ds = builder.as_dataset(split='train', shuffle_files=True)
        tfds.show_examples(ds, builder.info)
"""

def find_non_utf8(path):
    files_folders = os.listdir(path)
    for f in files_folders:
        curr_path = os.path.join(path, f)
        if os.path.isfile(curr_path):                
            with open(curr_path, "rb") as fb:
                file_bytes = fb.read()
                try: 
                    file_bytes.decode()
                except UnicodeError:
                    print(curr_path)
                    return 666
        else:
            find_non_utf8(curr_path)


def read_csv_dataset(path):
    # Search for description rows with wanted labels inside
    rows = []
    with open(path, newline='', encoding='utf-8') as csvfile:
        for row in csvfile:
            if x < 3:
                print(row)

def main():
    # Path with all the class folders and inside corresponding images
    dataset_path = os.path.join("datasets", "CN_coin_descriptions", "CN_hpz_dataset.csv")
    #dataset = read_csv_dataset(dataset_path)
    #return
    
    # List of object/person classes to be detected
    #class_labels = os.listdir(img_class_superpath)[-2::-50]
    #class_labels = ['hades', 'poseidon', 'zeus']
    # ['abacus', 'abundantia', 'acrostolium', 'acroteria', 'actaeon', 'aegis', 'aeneas', 'aequitas', 'agonistic crown', 'agrippa', 'agrippina minor', 'alexander iii', 'altar', 'amphora', 'anchialos', 'anchises', 'anchor', 'androclus', 'andromeda', 'animal', 'annona', 'antinous', 'antiochus ii theos', 'antlers', 'antonia minor', 'antoninus pius', 'anubis', 'aphrodite', 'apis', 'aplustre', 'apollo', 'apollon', 'apple', 'arch', 'archer', 'ares', 'ariadne', 'arm', 'armour', 'arrow', 'artemis', 'ascanius', 'asclepius', 'astragal', 'athena', 'athlete', 'attis', 'augustus', 'aulos', 'bag', 'barley', 'base', 'basin', 'basket', 'bear', 'bee', 'beehive', 'belt', 'berry', 'biga', 'bird', 'boar', 'board', 'bonus eventus', 'boot', 'bow', 'boxer', 'branch', 'breast', 'britannicus', 'bucranium', 'bull', 'bun', 'bust', 'byzas', 'caduceus', 'caligula', 'cantharus', 'cap', 'capaneus', 'capricorn', 'caracalla', 'cattle', 'centaur', 'cerberus', 'cestus', 'cetus', 'chain', 'chair', 'charis', 'child', 'chin', 'chlamys', 'cippus', 'cista', 'cithara', 'city gate', 'city goddess', 'claudius', 'cloak', 'cloth', 'club', 'coat', 'coil', 'column', 'commodus', 'concordia', 'conifer', 'corn', 'corn wreath', 'cornucopia', 'corybant', 'cotyle', 'crab', 'crayfish', 'crepidoma', 'crescent', 'cretan bull', 'crispina', 'cross', 'crown', 'cubit role', 'cuirass', 'cup', 'cybele', 'dagger', 'dardanos', 'decius', 'deer', 'demeter', 'demos', 'diadem', 'diadumenian', 'diomedes', 'dionysus', 'disc', 'discus', 'dog', 'dolphin', 'domitia longina', 'domitian', 'door', 'double ax', 'double chiton', 'drapery', 'drum', 'eagle', 'ear', 'earring', 'earth', 'eirene', 'elagabalus', 'elephant', 'emperor', 'eros', 'erymanthian boar', 'europa', 'eurypylos', 'eurystheus', 'eye', 'faustina maior', 'faustina minor', 'feather', 'figure', 'fillet', 'fish', 'flower', 'foot', 'fox', 'fruit', 'gaius caesar', 'galley', 'gallienus', 'ganymed', 'garland', 'garment', 'gate', 'ge', 'genius', 'germanicus', 'geta', 'globe', 'gold', 'gordian', 'gorgo', 'gorgoneion', 'grain', 'grape', 'griffin', 'hades', 'hadrian', 'hair', 'hammer', 'hand', 'handle', 'harpocrates', 'he-goat', 'head', 'headband', 'headdress', 'hebros', 'hecate', 'helios', 'helle', 'helmet', 'helmsman', 'hephaestos', 'hera', 'heracles', 'herennius etruscus', 'herm', 'hermes', 'herophile', 'heros', 'himation', 'hind', 'hippocampus', 'hippolyte', 'homonoia', 'horn', 'horse', 'hound', 'human', 'hydra', 'hydria', 'hygieia', 'ibis', 'iole', 'isis', 'ivy', 'ivy leaf', 'ivy wreath', 'jar', 'javelin', 'jaw', 'julia domna', 'julia mamaea', 'julia titi', 'kaikos', 'kalathos', 'kausia', 'keroessa', 'kline', 'knee', 'knife', 'knot', 'koronis', 'kotys i', 'krater', 'kyrbasia', 'ladder', 'ladon', 'lamp', 'lance', 'laurel branch', 'laurel tree', 'laurel wreath', 'leaf', 'liberalitas', 'lion', 'lion skin', 'lituus', 'livia', 'lizard', 'lotus', 'lucilla', 'lucius caesar', 'lucius verus', 'lyre', 'mace', 'macrinus', 'maenad', 'maesa', 'man', 'mane', 'mantle', 'marcus antonius', 'marcus aurelius', 'marsyas', 'maximinus thrax', 'maximus', 'medusa', 'melsas', 'mestos', 'military attire', 'modius', 'mouth', 'mural crown', 'nape', 'naval ram', 'necklace', 'nemean lion', 'nemesis', 'nereid', 'nero', 'nerva', 'nike', 'nymph', 'nymphaeum', 'oak', 'oak wreath', 'oar', 'obelisk', 'object', 'octavia', 'oinochoe', 'oiskos', 'olive', 'olive-branch', 'omphale', 'omphalos', 'ore', 'ornament', 'orpheus', 'ostrich', 'otacilia severa', 'owl', 'ox', 'palm', 'palm branch', 'palmette', 'paludamentum', 'pan', 'panther', 'parazonium', 'paris', 'patera', 'paula', 'peacock', 'pediment', 'pedum', 'pegasus', 'pellet', 'pelta', 'pendant', 'pergamos', 'perinthos', 'persephone', 'perseus', 'persian', 'pertinax', 'petasus', 'philetairos', 'philippus arabs', 'pig', 'pilos', 'pincer', 'plant', 'plautilla', 'plectrum', 'plinth', 'plotina', 'plough', 'poimes', 'pollux', 'polos', 'pontos', 'poppaea', 'poppy', 'poseidon', 'pot', 'prisoner', 'prize crown', 'protesilaos', 'protome', 'prow', 'ptolemy iii euergetes', 'purse', 'quadriga', 'quiver', 'radiate', 'ram', 'raven', 'ray', 'reed', 'rein', 'remus', 'rhyton', 'ribbon', 'river-god', 'robe', 'rock', 'roe', 'roma', 'romulus', 'roof', 'rooster', 'rose', 'rudder', 'sabina', 'sail', 'salonina', 'saloninus', 'sandal', 'satrap', 'satyr', 'scale', 'scales', 'scallop', 'scepter', 'scroll', 'selene', 'septimius severus', 'serapis', 'serpent staff', 'severus alexander', 'she-wolf', 'shell', 'shield', 'ship', 'shoulder', 'sickle', 'silenus', 'simpulum', 'simulacrum', 'sistrum', 'situla', 'skostokos', 'snake', 'soaemias', 'soldier', 'spear', 'sphinx', 'spithridates', 'spoke', 'squid', 'stag', 'stalk', 'standard', 'star', 'starflower', 'statue', 'statuette', 'stefane', 'step', 'stern', 'stick', 'stone', 'strap', 'strymon', 'stump of tree', 'swastika', 'sword', 'syrinx', 'table', 'taenia', 'telephos', 'telesphorus', 'temple', 'thalassa', 'theos megas', 'thorn', 'throne', 'thronia', 'thunderbolt', 'thymiaterion', 'thyrsus', 'tiara', 'tiberius', 'titus', 'toga', 'tongue', 'tonzos', 'torch', 'tortoise', 'tower', 'trajan', 'tranquillina', 'trebonianus gallus', 'tree', 'trident', 'tripod', 'triptolemos', 'triskele', 'triton', 'trophy', 'tunic', 'tunny', 'turtle', 'twins', 'tyche', 'tympanon', 'umbo', 'urn', 'valerian', 'vase', 'veil', 'vespasian', 'vessel', 'vexillum', 'vine', 'virtus', 'volusian', 'wave', 'wheel', 'whip', 'wine skin', 'wing', 'wolf', 'woman', 'worm', 'wreath', 'wrestler', 'youth', 'zeus', 'zodiac']
    #dataset = LocalClassImageDataset(img_class_superpath, class_labels)
    #dataset = load_dataset("imagefolder", data_dir=img_class_superpath)
    #model = CLIP_implementation(dataset, class_labels)
    model = OWL_ViT_implementation()
    

    #print(dataset.get_current_class())
    #print(dataset.get_image_path())
    #dataset.change_class("zeus")
    image = Image.open("datasets\CN_dataset_obj_detection_04_23\dataset_obj_detection\Poseidon\CN_type_17548_cn_coin_42294_p_rev.jpg")
    text = "Poseidon standing left, holding dolphin and hippocampus."
    predictions = model(image, candidate_labels=text)
    draw = ImageDraw.Draw(image)

    for prediction in predictions:
        box = prediction["box"]
        label = prediction["label"]
        score = prediction["score"]

        xmin, ymin, xmax, ymax = box.values()
        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
        draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="blue")

    image.save(os.path.join("results", "no_coin_dataset_pretraining", "poseidon_expected.jpg"), "JPEG")
    return 0


if __name__ == "__main__":
    main()
    
