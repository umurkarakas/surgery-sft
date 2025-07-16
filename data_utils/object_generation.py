import os
from glob import glob
import json
import pandas as pd

def select_video_from_timestamp(case_videos, timestamp):
    """
    Find the video that contains a specific timestamp.
    
    Args:
        case_videos (list): List of video filenames for a case
        timestamp (float): Timestamp in seconds to search for
        
    Returns:
        str or None: Filename of the video containing the timestamp, or None if not found
    """
    for video in case_videos:
        # Extract start and end times from the video filename
        split = video.split("_")
        start, end = float(split[-2]), float(split[-1])
        
        # Check if timestamp falls within this video's time range
        if timestamp >= start and timestamp <= end:
            return video
    
    # Return None if no matching video is found
    return None

def main():
    """
    Main function to generate object annotations for cataract surgery videos.
    
    This function:
    1. Identifies common cases between segment and phase annotations
    2. Maps objects and phases to specific timestamps
    3. Creates a JSON file with case, timestamp, phase, and object information
    """
    # Define paths to annotation directories
    segment_case_dir = "datasets/cataract1k/annotations/segment_annotations/cases/"
    segment_cases = os.listdir(segment_case_dir)
    phase_case_dir = "datasets/cataract1k/annotations/phase_annotations/"
    phase_cases = os.listdir(phase_case_dir)

    # Find cases that have both segment and phase annotations
    common_cases = list(set(segment_cases).intersection(set(phase_cases)))

    # Get list of available videos
    videos_dir = "datasets/cataract1k/videos/"
    videos = os.listdir(videos_dir)

    # Dictionary to store case objects
    case_objects = {}
    
    # Process each case
    for case in common_cases:
        # Get videos for this case
        case_videos = [video[:-4] for video in videos if video.startswith(case)]
        
        # Load phase annotations
        phase_ann_df = pd.read_csv(
            f"datasets/cataract1k/annotations/phase_annotations/{case}/{case}_annotations_phases.csv"
        )
        
        # Load segment annotations (objects)
        with open(os.path.join(segment_case_dir, case, "annotations", "instances.json"), "r") as f:
            inst_json = json.load(f)
        
        # Create mappings for image IDs and category IDs
        image_id_to_name = {x["id"]:x["file_name"] for x in inst_json["images"]}
        category_id_to_name = {x["id"]:x["name"] for x in inst_json["categories"]}
        
        # Initialize case in dictionary if not present
        if case not in case_objects:
            case_objects[case] = {}
            
        # Process each annotation
        for ann in inst_json["annotations"]:
            # Get image name and calculate timestamp in seconds
            image_name = image_id_to_name[ann["image_id"]]
            if int(image_name.split("_")[-1].split(".")[0]) == 2:
                continue
            image_seconds = max(0, int(image_name.split("_")[-1].split(".")[0])-2) * 5
            
            # Initialize timestamp entry if not present
            if image_seconds not in case_objects[case]:
                case_objects[case][image_seconds] = {}
            
            # Find the phase for this timestamp
            phase = phase_ann_df[
                (phase_ann_df.sec <= image_seconds) & 
                (phase_ann_df.endSec >= image_seconds)
            ].comment
            
            # Set phase to "idle" if no phase is found
            if len(phase) == 0:
                phase = "idle"
            else:
                phase = phase.iloc[0]
                
            # Store phase information
            case_objects[case][image_seconds]["phase"] = phase
            
            # Initialize objects list if not present
            if "objects" not in case_objects[case][image_seconds]:
                case_objects[case][image_seconds]["objects"] = []
                
            # Add object to the list
            case_objects[case][image_seconds]["objects"].append(
                {"area": ann["area"], "bbox": ann["bbox"], "object_name": category_id_to_name[ann["category_id"]]}
            )
            
            # Find and store the video filename containing this timestamp
            case_objects[case][image_seconds]["video_filename"] = select_video_from_timestamp(
                case_videos, image_seconds
            )

    # Save the case objects to a JSON file
    with open(f'datasets/cataract1k/case_objects.json', 'w') as fp:
        json.dump(case_objects, fp)

if __name__ == "__main__":
    main() 