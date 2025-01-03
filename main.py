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


    return pd.DataFrame(segments_groups, columns = ['file name', 'segment, group', 'id', 'parent', 'relname', 'type', 'text'])




def main(): 

    rs4_file_paths = list_rs4_file_paths('data')

    segment_group_df = rs4_files_to_df(rs4_file_paths) 

    segment_group_df.to_csv('data/intermediate_formats/segment_group_df.csv', index=False)

    print(segment_group_df)



if __name__ == "__main__": 
    main()