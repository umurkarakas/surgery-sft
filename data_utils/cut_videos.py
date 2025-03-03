import ffmpeg
import pandas as pd
import os
import re

def sanitize_filename(filename):
    """
    Sanitize a filename by removing invalid characters.
    
    Args:
        filename (str): The original filename to sanitize
        
    Returns:
        str: A sanitized filename with invalid characters replaced by underscores
    """
    # Define the regex pattern to match unwanted characters
    pattern = r'[<>:"/\\|?*.\s]+'
    
    # Replace unwanted characters with an underscore
    sanitized = re.sub(pattern, '_', filename)
    
    # Remove leading and trailing underscores
    sanitized = sanitized.strip('_')
    
    return sanitized

def cut_video(video_file, output_folder, phase, start_time, end_time):
    """
    Cut a video into smaller segments based on specified time intervals.
    
    Args:
        video_file (str): Path to the input video file
        output_folder (str): Directory to save the cut video segments
        phase (str): Surgical phase name to include in the output filename
        start_time (float): Start time in seconds for the segment
        end_time (float): End time in seconds for the segment
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Extract case name from the video file path
    case_name = video_file.split("/")[-1].split(".")[0]
    interval = 5  # Fixed interval size in seconds
    
    # Calculate the center of the first interval
    current_center = ((start_time // interval) + 1) * interval
    current_end = current_center + interval / 2

    # Process the video in intervals until reaching the end time
    while current_end != end_time:
        # Determine the start and end of the current interval
        current_start = current_center - interval / 2
        if current_start // interval == start_time // interval:
            current_start = start_time
        current_end = current_center + interval / 2
        if current_end // interval >= end_time // interval:
            current_end = end_time

        # Move to the next center point
        current_center += interval
        
        # Create a sanitized phase name for the filename
        sanitized_phase = sanitize_filename(phase)
        
        # Generate output filename with case, phase, and time information
        output_filename = "{}/{}_{}_{:.2f}_{:.2f}.mp4".format(
            output_folder, case_name, sanitized_phase, current_start, current_end
        )
        
        # Use ffmpeg to cut the video segment
        ffmpeg.input(video_file, ss=current_start, t=current_end-current_start).output(
            output_filename
        ).run(overwrite_output=True)

if __name__ == "__main__":
    # Define paths for data processing
    data_folder = "datasets/cataract1k/annotations/"
    video_folder = "surgery Cataract-1K/Phase_recognition_dataset/"
    output_folder = "datasets/cataract1k/videos/"
    annotations_dir = os.path.join(data_folder, "phase_annotations")
    
    # Get list of case directories
    cases = [path for path in os.listdir(annotations_dir) 
             if os.path.isdir(os.path.join(annotations_dir, path))]
    
    # Create paths to annotation files
    annotations = [os.path.join(annotations_dir, case, f"{case}_annotations_phases.csv") 
                  for case in cases]

    ## Process surgical phases
    for case, annotation in zip(cases, annotations):
        # Load annotation data
        ann_df = pd.read_csv(annotation)
        video_file = os.path.join(video_folder, "videos_224", f"{case}.mp4")
        
        # Process each phase annotation
        for i in range(len(ann_df)):
            phase = ann_df.loc[i, "comment"]
            start_frame = ann_df.loc[i, "frame"]
            start_time = ann_df.loc[i, "sec"]
            end_time = ann_df.loc[i, "endSec"]
            cut_video(video_file, output_folder, phase, start_time, end_time)

    ## Code for processing idle states (commented out)
    ## for case, annotation in zip(cases, annotations):
    ##     ann_df = pd.read_csv(annotation)
    ##     video_file = os.path.join(video_folder, "videos_224", f"{case}.mp4")
    ##     for i in range(len(ann_df)):
    ##         phase = "idle"
    ##         if i == 0:
    ##             if ann_df.loc[i, "sec"] != 0:
    ##                 start_time = 0
    ##                 end_time = ann_df.loc[i, "sec"]
    ##                 cut_video(video_file, output_folder, phase, start_time, end_time)
    ##         else:
    ##             start_time = ann_df.loc[i-1, "endSec"]
    ##             end_time = ann_df.loc[i, "sec"]
    ##             cut_video(video_file, output_folder, phase, start_time, end_time)