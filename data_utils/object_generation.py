import os
from glob import glob
import json
import pandas as pd

def select_video_from_timestamp(case_videos, timestamp):
    for video in case_videos:
        split = video.split("_")
        start, end = float(split[-2]), float(split[-1])
        if timestamp >= start and timestamp <= end:
            return video

def main():
    segment_case_dir = "datasets/cataract1k/annotations/segment_annotations/cases/"
    segment_cases = os.listdir(segment_case_dir)
    phase_case_dir = "datasets/cataract1k/annotations/phase_annotations/"
    phase_cases = os.listdir(phase_case_dir)

    common_cases = list(set(segment_cases).intersection(set(phase_cases)))

    videos_dir = "datasets/cataract1k/videos/"
    videos = os.listdir(videos_dir)

    case_objects = {}
    for case in common_cases:
        case_videos = [video[:-4] for video in videos if video.startswith(case)]
        phase_ann_df = pd.read_csv(f"datasets/cataract1k/annotations/phase_annotations/{case}/{case}_annotations_phases.csv")
        with open(os.path.join(segment_case_dir, case, "annotations", "instances.json"), "r") as f:
            inst_json = json.load(f)
        
        image_id_to_name = {x["id"]:x["file_name"] for x in inst_json["images"]}
        category_id_to_name = {x["id"]:x["name"] for x in inst_json["categories"]}
        
        if case not in case_objects:
            case_objects[case] = {}
            
        for ann in inst_json["annotations"]:
            image_name = image_id_to_name[ann["image_id"]]
            image_seconds = max(0,int(image_name.split("_")[-1].split(".")[0])-2) * 5
            if image_seconds not in case_objects[case]:
                case_objects[case][image_seconds] = {}
            phase = phase_ann_df[(phase_ann_df.sec <= image_seconds) & (phase_ann_df.endSec >= image_seconds)].comment
            if len(phase) == 0:
                phase = "idle"
            else:
                phase = phase.iloc[0]
            case_objects[case][image_seconds]["phase"] = phase
            if "objects" not in case_objects[case][image_seconds]:
                case_objects[case][image_seconds]["objects"] = []
            case_objects[case][image_seconds]["objects"].append(category_id_to_name[ann["category_id"]])
            case_objects[case][image_seconds]["video_filename"] = select_video_from_timestamp(case_videos, image_seconds)

    with open(f'datasets/cataract1k/case_objects.json', 'w') as fp:
        json.dump(case_objects, fp)

if __name__ == "__main__":
    main() 