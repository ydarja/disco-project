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



def rsd_file_paths_to_dict(rsd_file_paths):

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

    for file_path in rsd_file_paths:

        ids = []
        texts = []
        parents = []
        relations = []

        with open(file_path, 'r', encoding='utf-8') as file: 
            reader = csv.reader(file, delimiter='\t') 
    
            for row in reader:
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
            if relations[i][-1] == 't' and parents[i] in ids:
                edu_pairs.append([['<s>'] + texts[i].split(' ') + ['<sep>'] + texts[ids.index(parents[i])].split(' ') + ['<n>'], relations[i][:-2]])
            elif relations[i][-1] == 'm' and parents[i] in ids:
                edu_pairs.append([['<n>'] + texts[i].split(' ') + ['<sep>'] + texts[ids.index(parents[i])].split(' ') + ['<n>'], relations[i][:-2]])

        file_genre = file_path[file_path.find('_')+1:file_path.rfind('_')]
        file_name = file_path[file_path.rfind('_')+1:file_path.find('.')]

        if file_genre in ['conversation', 'interview', 'reddit', 'vlog']:
            group_genre_file_dict['conversational'][file_genre][file_name] = edu_pairs
            
        elif file_genre in ['fiction', 'news', 'speech', 'whow']:
            group_genre_file_dict['prose'][file_genre][file_name] = edu_pairs

        elif file_genre in ['academic', 'bio', 'textbook', 'voyage']:
            group_genre_file_dict['science'][file_genre][file_name] = edu_pairs
            
    return group_genre_file_dict


def verbal_group_genre_file_dict(group_genre_file_dict):

    print("############################################################")
    print("Currently this parsing functionality outputs a dictionary of the structure:")
    print("\{Group:\{Genre:\{File:[EDU_Pairs]\}\}\}")

    print(f"This implementation groups genres into {len(group_genre_file_dict.keys())} groups: {group_genre_file_dict.keys()}")

    for group_name, group_genre_dict in group_genre_file_dict.items():
        for genre_name, genre_file_dict in group_genre_dict.items():
            for file_name, edu_pairs in genre_file_dict.items():
                print(f"Group: {group_name}, Genre: {genre_name}, File: {file_name}, RelationCount: {len(edu_pairs)}")


def main(): 

    rsd_file_paths = list_rsd_file_paths('data')

    group_genre_file_dict = rsd_file_paths_to_dict(rsd_file_paths)

    verbal_group_genre_file_dict(group_genre_file_dict)





if __name__ == "__main__": 
    main()