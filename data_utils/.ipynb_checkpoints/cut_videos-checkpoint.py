import ffmpeg
import pandas as pd
import os
import re

def sanitize_filename(filename):
    # Define the regex pattern to match unwanted characters
    pattern = r'[<>:"/\\|?*.\s]+'
    
    # Replace unwanted characters with an underscore
    sanitized = re.sub(pattern, '_', filename)
    
    # Remove leading and trailing underscores
    sanitized = sanitized.strip('_')
    
    return sanitized

def cut_video(video_file, output_folder, phase, start_time, end_time):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    case_name = video_file.split("/")[-1].split(".")[0]
    interval = 5
    current_center = ((start_time // interval) + 1) * interval
    current_end = current_center + interval / 2

    while current_end != end_time:
        # Determine the start and end of the current interval
        current_start = current_center - interval / 2
        if current_start // interval == start_time // interval:
            current_start = start_time
        current_end = current_center + interval / 2
        if current_end // interval >= end_time // interval:
            current_end = end_time

        current_center += interval
        sanitized_phase = sanitize_filename(phase)
        output_filename = "{}/{}_{}_{:.2f}_{:.2f}.mp4".format(output_folder,case_name, sanitized_phase, current_start, current_end)
        ffmpeg.input(video_file, ss=current_start, t=current_end-current_start).output(output_filename).run(overwrite_output=True)

if __name__ == "__main__":
    data_folder = "datasets/cataract1k/annotations/"
    video_folder = "cataract1k/surgery Cataract-1K/Phase_recognition_dataset/"
    output_folder = "datasets/cataract1k/videos/"
    annotations_dir = os.path.join(data_folder, "phase_annotations")
    cases = [path for path in os.listdir(annotations_dir) if os.path.isdir(os.path.join(annotations_dir,path))]
    annotations = [os.path.join(annotations_dir, case, f"{case}_annotations_phases.csv") for case in cases]

    ## phases
    for case, annotation in zip(cases, annotations):
        ann_df = pd.read_csv(annotation)
        video_file = os.path.join(video_folder, "videos_224", f"{case}.mp4")
        for i in range(len(ann_df)):
            phase = ann_df.loc[i, "comment"]
            start_frame = ann_df.loc[i, "frame"]
            start_time = ann_df.loc[i, "sec"]
            end_time = ann_df.loc[i, "endSec"]
            cut_video(video_file, output_folder, phase, start_time, end_time)

    ## idle states
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