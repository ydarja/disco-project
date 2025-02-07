# -*- coding: utf-8 -*- 

""" 
Title: Leveraging attention in discourse classification for genre diverse data
Description: Reads rs4 files. Outputs a data frame for processing down the line
Author: Darja Jepifanova, Marco Floess
Date: 2025-02-xx
""" 

# Import necessary modules 
import os
import csv
import shutil
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# from flair.data import Sentence

# List all files in the given directory with the .rsd extension
def list_rsd_file_paths(directory): 
    
    try: 
        # Get all file names in the specified directory 
        file_names = os.listdir(directory) 

        # Filter out files ending with .rs4 
        rs4_file_paths = [directory + '/' + file for file in file_names if file.endswith('.rsd')] 

        return rs4_file_paths 
        
    except Exception as e: 

        print(f"An error occurred: {e}") 

        return [] 



def rsd_file_paths_to_dict(rsd_file_paths, fine_grained=True):

    group_genre_file_dict = {
        'conversational':{
            'conversation':{},
            'interview':{},
            'reddit':{},
            'vlog':{}
        },
        'prose':{
            'fiction':{},
            'news':{},
            'speech':{},
            'whow':{}
        },
        'science':{
            'academic':{},
            'bio':{},
            'textbook':{},
            'voyage':{}
        }
    }

    labels = []

    for file_path in rsd_file_paths:

        ids = []
        texts = []
        parents = []
        relations = []

        with open(file_path, 'r', encoding='utf-8') as file: 
            for line in file:

                row = line.split('\t')

                try:
                    ids.append(int(row[0]))
                    texts.append(row[1])
                    parents.append(int(row[6]))
                    relations.append(row[7])

                except IndexError: 
                    print(f"Skipping row with insufficient columns in file: {file_path}")
                    ids = ids[:len(relations)]
                    texts = texts[:len(relations)]
                    parents = parents[:len(relations)]

        edu_pairs = []

        for i in range(len(ids)):
            if relations[i][-1] == 'r' and parents[i] in ids:

                edu_text = ['<s>'] + texts[i].split(' ') + ['<sep>'] + texts[ids.index(parents[i])].split(' ') + ['<n>']

                if fine_grained:
                    edu_pairs.append([edu_text, relations[i][:-2]])
                    labels.append(relations[i][:-2])
                else: # coarse
                    edu_pairs.append([edu_text, relations[i][:relations[i].rfind('-')] if relations[i].find('same') else relations[i][:-2]])
                    labels.append(relations[i][:relations[i].rfind('-')] if relations[i].find('same') else relations[i][:-2])

            elif relations[i][-1] == 'm' and parents[i] in ids:
                edu_text = ['<n>'] + texts[i].split(' ') + ['<sep>'] + texts[ids.index(parents[i])].split(' ') + ['<n>']

                if fine_grained:
                    edu_pairs.append([edu_text, relations[i][:-2]])
                    labels.append(relations[i][:-2])
                else: # coarse
                    edu_pairs.append([edu_text, relations[i][:relations[i].rfind('-')] if relations[i].find('same') else relations[i][:-2]])
                    labels.append(relations[i][:relations[i].rfind('-')] if relations[i].find('same') else relations[i][:-2])
                    

        file_genre = file_path[file_path.find('_')+1:file_path.rfind('_')]
        file_name = file_path[file_path.rfind('_')+1:file_path.find('.')]

        if file_genre in ['conversation', 'interview', 'reddit', 'vlog']:
            group_genre_file_dict['conversational'][file_genre][file_name] = edu_pairs
            
        elif file_genre in ['fiction', 'news', 'speech', 'whow']:
            group_genre_file_dict['prose'][file_genre][file_name] = edu_pairs

        elif file_genre in ['academic', 'bio', 'textbook', 'voyage']:
            group_genre_file_dict['science'][file_genre][file_name] = edu_pairs
       
    return group_genre_file_dict, set(labels)


def verbal_group_genre_file_dict(group_genre_file_dict):

    print("############################################################")
    print("Currently this parsing functionality outputs a dictionary of the structure:")
    print("\{Group:\{Genre:\{File:[EDU_Pairs]\}\}\}")

    print(f"This implementation groups genres into {len(group_genre_file_dict.keys())} groups: {group_genre_file_dict.keys()}")

    for group_name, group_genre_dict in group_genre_file_dict.items():
        for genre_name, genre_file_dict in group_genre_dict.items():
            for file_name, edu_pairs in genre_file_dict.items():
                print(f"Group: {group_name}, Genre: {genre_name}, File: {file_name}, RelationCount: {len(edu_pairs)}")



def load_data(directory, batch_size=8):
    """
    Load data, process it, and return a DataLoader for training.
    """
    rsd_file_paths = list_rsd_file_paths(directory)
    group_genre_file_dict, labels = rsd_file_paths_to_dict(rsd_file_paths, False)
    label_map = {value: idx for idx, value in enumerate(sorted(labels))}

    # Flatten the data into a list of EDU pairs and relations for training
    edu_pairs_list = []
    for genre in group_genre_file_dict.values():
        for sub_genre in genre.values():
            for file_data in sub_genre.values():
                edu_pairs_list.extend(file_data)

    # Prepare the data: EDU pairs and label indices
    data = [
        (edu_pair, label_map.get(relation.lower(), 42))
        for edu_pair, relation in edu_pairs_list
    ]

    # Create the DataLoader
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)
    return dataloader

# organize files in the directories according to the standard train/dev/test splits
def organize_splits():
    splits = {
    "train": [
        "GUM_academic_art", "GUM_academic_census", "GUM_academic_economics",
        "GUM_academic_enjambment", "GUM_academic_epistemic", "GUM_academic_games",
        "GUM_academic_huh", "GUM_academic_implicature", "GUM_academic_lighting",
        "GUM_academic_mutation", "GUM_academic_replication", "GUM_academic_salinity",
        "GUM_academic_theropod", "GUM_academic_thrones", "GUM_bio_bernoulli",
        "GUM_bio_chao", "GUM_bio_enfant", "GUM_bio_fillmore", "GUM_bio_galois",
        "GUM_bio_goode", "GUM_bio_gordon", "GUM_bio_hadid", "GUM_bio_higuchi",
        "GUM_bio_holt", "GUM_bio_jerome", "GUM_bio_marbles", "GUM_bio_moreau",
        "GUM_bio_nida", "GUM_bio_padalecki", "GUM_bio_theodorus", "GUM_conversation_atoms",
        "GUM_conversation_blacksmithing", "GUM_conversation_christmas", "GUM_conversation_erasmus",
        "GUM_conversation_family", "GUM_conversation_gossip", "GUM_conversation_scientist",
        "GUM_conversation_toys", "GUM_conversation_vet", "GUM_conversation_zero",
        "GUM_court_carpet", "GUM_court_equality", "GUM_court_fire", "GUM_court_prince",
        "GUM_essay_distraction", "GUM_essay_dividends", "GUM_essay_sexlife", "GUM_fiction_claus",
        "GUM_fiction_error", "GUM_fiction_frankenstein", "GUM_fiction_garden", "GUM_fiction_giants",
        "GUM_fiction_honour", "GUM_fiction_moon", "GUM_fiction_oversite", "GUM_fiction_pag",
        "GUM_fiction_pixies", "GUM_fiction_rose", "GUM_fiction_sneeze", "GUM_fiction_time",
        "GUM_fiction_veronique", "GUM_fiction_wedding", "GUM_interview_ants", "GUM_interview_brotherhood",
        "GUM_interview_chomsky", "GUM_interview_cocktail", "GUM_interview_daly", "GUM_interview_dungeon",
        "GUM_interview_herrick", "GUM_interview_licen", "GUM_interview_mcguire", "GUM_interview_mckenzie",
        "GUM_interview_messina", "GUM_interview_onion", "GUM_interview_peres", "GUM_interview_shalev",
        "GUM_interview_stardust", "GUM_letter_flood", "GUM_letter_gorbachev", "GUM_letter_roomers",
        "GUM_letter_zora", "GUM_news_afghan", "GUM_news_asylum", "GUM_news_clock", "GUM_news_crane",
        "GUM_news_defector", "GUM_news_election", "GUM_news_expo", "GUM_news_flag", "GUM_news_hackers",
        "GUM_news_ie9", "GUM_news_imprisoned", "GUM_news_korea", "GUM_news_lanterns", "GUM_news_soccer",
        "GUM_news_stampede", "GUM_news_taxes", "GUM_news_warhol", "GUM_news_warming", "GUM_news_worship",
        "GUM_podcast_addiction", "GUM_podcast_brave", "GUM_podcast_collaboration", "GUM_reddit_bobby",
        "GUM_reddit_callout", "GUM_reddit_card", "GUM_reddit_conspiracy", "GUM_reddit_gender",
        "GUM_reddit_introverts", "GUM_reddit_polygraph", "GUM_reddit_racial", "GUM_reddit_ring",
        "GUM_reddit_social", "GUM_reddit_space", "GUM_reddit_steak", "GUM_reddit_stroke",
        "GUM_reddit_superman", "GUM_speech_albania", "GUM_speech_data", "GUM_speech_destiny",
        "GUM_speech_floyd", "GUM_speech_humanitarian", "GUM_speech_maiden", "GUM_speech_nixon",
        "GUM_speech_remarks", "GUM_speech_school", "GUM_speech_telescope", "GUM_speech_trump",
        "GUM_textbook_alamo", "GUM_textbook_anthropology", "GUM_textbook_artwork", "GUM_textbook_cognition",
        "GUM_textbook_entrepreneurship", "GUM_textbook_evoethics", "GUM_textbook_grit", "GUM_textbook_history",
        "GUM_textbook_sociology", "GUM_textbook_spacetime", "GUM_textbook_stats", "GUM_vlog_appearance",
        "GUM_vlog_college", "GUM_vlog_covid", "GUM_vlog_exams", "GUM_vlog_hair", "GUM_vlog_hiking",
        "GUM_vlog_lipstick", "GUM_vlog_mermaid", "GUM_vlog_pizzeria", "GUM_vlog_pregnant", "GUM_vlog_wine",
        "GUM_voyage_chatham", "GUM_voyage_cleveland", "GUM_voyage_cuba", "GUM_voyage_fortlee",
        "GUM_voyage_guadeloupe", "GUM_voyage_isfahan", "GUM_voyage_lodz", "GUM_voyage_merida",
        "GUM_voyage_phoenix", "GUM_voyage_socotra", "GUM_voyage_sydfynske", "GUM_voyage_thailand",
        "GUM_voyage_tulsa", "GUM_voyage_york", "GUM_whow_arrogant", "GUM_whow_ballet", "GUM_whow_basil",
        "GUM_whow_chicken", "GUM_whow_cupcakes", "GUM_whow_elevator", "GUM_whow_flirt", "GUM_whow_glowstick",
        "GUM_whow_languages", "GUM_whow_packing", "GUM_whow_parachute", "GUM_whow_procrastinating",
        "GUM_whow_quidditch", "GUM_whow_quinoa", "GUM_whow_skittles"
    ],
    "dev": [
    "GUM_academic_exposure", "GUM_academic_librarians", "GUM_bio_byron", "GUM_bio_emperor",
    "GUM_conversation_grounded", "GUM_conversation_risk", "GUM_court_loan", "GUM_essay_evolved",
    "GUM_fiction_beast", "GUM_fiction_lunre", "GUM_interview_cyclone", "GUM_interview_gaming",
    "GUM_letter_arendt", "GUM_news_homeopathic", "GUM_news_iodine", "GUM_podcast_wrestling",
    "GUM_reddit_macroeconomics", "GUM_reddit_pandas", "GUM_speech_impeachment", "GUM_speech_inauguration",
    "GUM_textbook_governments", "GUM_textbook_labor", "GUM_vlog_portland", "GUM_vlog_radiology",
    "GUM_voyage_athens", "GUM_voyage_coron", "GUM_whow_joke", "GUM_whow_overalls"
],
    "test": [
        "GUM_academic_discrimination", "GUM_academic_eegimaa", "GUM_bio_dvorak", 
        "GUM_bio_jespersen", "GUM_conversation_lambada", "GUM_conversation_retirement", "GUM_court_mitigation",
        "GUM_essay_fear", "GUM_fiction_falling", "GUM_fiction_teeth", "GUM_interview_hill", "GUM_interview_libertarian",
        "GUM_letter_mandela", "GUM_news_nasa", "GUM_news_sensitive", "GUM_podcast_bezos", "GUM_reddit_escape", "GUM_reddit_monsters",
        "GUM_speech_austria", "GUM_speech_newzealand", "GUM_textbook_chemistry", "GUM_textbook_union", "GUM_vlog_london","GUM_vlog_studying",
        "GUM_voyage_oakland", "GUM_voyage_vavau", "GUM_whow_cactus", "GUM_whow_mice"
    ]
}
    output_dirs = {
    "train": "data/train",
    "dev": "data/dev",
    "test": "data/test"
}
    for split, output_dir in output_dirs.items():
        os.makedirs(output_dir, exist_ok=True)

    for split, file_list in splits.items():
        target_dir = output_dirs[split]

        for file_name in file_list:
            file_path = os.path.join('data/', f"{file_name}.rsd")
            
            if os.path.exists(file_path):
                shutil.copy(file_path, target_dir)  # Use shutil.move() to move instead of copying
                print(f"Copied {file_name}.rsd to {target_dir}")
            else:
                print(f"File {file_name}.rsd not found in data")


def main(): 

    rsd_file_paths = list_rsd_file_paths('data/train')

    group_genre_file_dict, relations = rsd_file_paths_to_dict(rsd_file_paths)  

    #verbal_group_genre_file_dict(group_genre_file_dict)

    load_data('data/train')

    #organize_splits()




if __name__ == "__main__": 
    main()