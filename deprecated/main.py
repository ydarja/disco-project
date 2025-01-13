# -*- coding: utf-8 -*- 

""" 
Title: Leveraging attention in discourse classification for genre diverse data
Description: Reads rs4 files. Outputs a data frame for processing down the line
Author: Darja Jepifanova, Marco Floess
Date: 2025-02-xx
""" 

# Import necessary modules 
import os
import xmltodict
import pandas as pd 


# List all files in the given directory with the .rs4 extension
def list_rs4_file_paths(directory): 
    
    try: 
        # Get all file names in the specified directory 
        file_names = os.listdir(directory) 

        # Filter out files ending with .rs4 
        rs4_file_paths = [directory + '/' + file for file in file_names if file.endswith('.rs4')] 

        return rs4_file_paths 
        
    except Exception as e: 

        print(f"An error occurred: {e}") 

        return [] 


def rs4_files_to_df(rs4_file_paths):

    segments_groups = []

    for file_path in rs4_file_paths:

        with open(file_path, encoding='utf-8') as file:
            
            data = xmltodict.parse(file.read()) 

        for segment in data['rst']['body']['segment']:
            segments_groups.append([
                file_path[file_path.rfind('/')+1:file_path.rfind('.rs4')],
                'segment', 
                segment.get('@id', None), 
                segment.get('@parent', None), 
                segment.get('@relname', None), 
                None, 
                segment.get('#text', None)
            ])

        for group in data['rst']['body']['group']:
            segments_groups.append([
                file_path[file_path.rfind('/')+1:file_path.rfind('.rs4')],
                'group', 
                group.get('@id', None), 
                group.get('@parent', None), 
                group.get('@relname', None), 
                group.get('@type', None), 
                None
            ])


    return pd.DataFrame(segments_groups, columns = ['file_name', 'segment_or_group', 'id', 'parent', 'relname', 'type', 'text'])


def rs4_files_to_dict(rs4_file_paths):

    group_genre_pair_dict = {
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

    for file_path in rs4_file_paths:

        with open(file_path, encoding='utf-8') as file:
            
            data = xmltodict.parse(file.read()) 

        segment_ids = []
        segment_parents = []
        segment_relnames = []
        segment_texts = []

        for segment in data['rst']['body']['segment']:
            segment_ids.append(int(segment.get('@id', None)))
            segment_parents.append(int(segment.get('@parent', None)))
            segment_relnames.append(segment.get('@relname', None))
            segment_texts.append(segment.get('#text', None))

        input_pairs = []

        for i in range(len(segment_ids)):
            # Case 1
            if segment_parents[i] in segment_ids:
                input_pairs.append([['<s>'] + segment_texts[i].split(' ') + ['<sep>'] + segment_texts[segment_parents[i] - 1].split(' ') + ['<n>'], segment_relnames[i]])
        
            # Case 2
            if segment_parents[i] not in segment_ids and segment_parents.count(segment_parents[i]) > 1:
                indices = [j for j, parent in enumerate(segment_parents) if parent == segment_parents[i] and j != i]
        
                texts = []
                for k in indices:
                    texts = texts + segment_texts[k].split(' ')
            
                input_pairs.append([['<?>'] + segment_texts[i].split(' ') + ['<sep>'] + texts + ['<?>'], segment_relnames[i]])

        file_genre = file_path[file_path.find('_')+1:file_path.rfind('_')]
        file_name = file_path[file_path.rfind('_')+1:file_path.find('.')]

        if file_genre in ['conversation', 'interview', 'reddit', 'vlog']:
            group_genre_pair_dict['conversational'][file_genre][file_name] = input_pairs
            
        elif file_genre in ['fiction', 'news', 'speech', 'whow']:
            group_genre_pair_dict['prose'][file_genre][file_name] = input_pairs

        elif file_genre in ['academic', 'bio', 'textbook', 'voyage']:
            group_genre_pair_dict['science'][file_genre][file_name] = input_pairs
            
    return group_genre_pair_dict


def verbal_group_genre_pair_dict(group_genre_pair_dict):

    print("############################################################")
    print("Currently this parsing functionality outputs a dictionary of the structure:")
    print("Group:Genre:File:Input_Pairs")

    print(f"This implementation groups genres into {len(group_genre_pair_dict.keys())} groups: {group_genre_pair_dict.keys()}")

    for group_name, group_genre_dict in group_genre_pair_dict.items():
        for genre_name, genre_file_dict in group_genre_dict.items():
            for file_name, input_pairs in genre_file_dict.items():
                print(f"Group: {group_name}, Genre: {genre_name}, File: {file_name}, RelationCount: {len(input_pairs)}")


def main(): 

    rs4_file_paths = list_rs4_file_paths('data')

    # segment_group_df = rs4_files_to_df(rs4_file_paths) 
    group_genre_pair_dict = rs4_files_to_dict(rs4_file_paths)

    # segment_group_df.to_csv('data/intermediate_formats/segment_group_df.csv', index=False)
    verbal_group_genre_pair_dict(group_genre_pair_dict)





if __name__ == "__main__": 
    main()