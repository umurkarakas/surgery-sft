from vllm import LLM, SamplingParams
import ast
import json
import re
import os

def remove_digits(string):
    """
    Remove all digits from a string.
    
    Args:
        string (str): Input string containing digits
        
    Returns:
        str: String with all digits removed
    """
    return re.sub(r'\d+', '', string)

def main():
    """
    Main function to generate question-answer pairs for cataract surgery videos.
    
    This function:
    1. Initializes a large language model (Mistral-Small-24B)
    2. Loads case objects containing phase and object information
    3. Generates prompts for the LLM to create Q&A pairs
    4. Processes LLM responses into structured JSON format
    5. Saves the generated Q&A pairs to a JSON file
    """
    # Initialize the large language model with appropriate parameters
    llm = LLM("Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ",
             gpu_memory_utilization=0.9,
             max_num_batched_tokens=32768,
             max_num_seqs=512,
             max_model_len=32768*4,
             dtype="bfloat16",
             enforce_eager=False,
             enable_chunked_prefill=True,
             max_seq_len_to_capture=131072,
             enable_prefix_caching=True)

    # Get home path and load case objects
    home_path = os.getenv("HOME")
    with open("datasets/cataract1k/case_objects.json", "r") as f:
        case_objects = json.load(f)

    # Define system prompt for the LLM
    SYSTEM_PROMPT = """You are a vision language model specialized in extracting information from cataract surgery videos.
    Your task is to analyze provided videos and answer the questions based on the inputted cataract surgery videos."""
    videos_path = "datasets/cataract1k/videos/"

    # Generate messages for the LLM
    all_messages = []
    all_videos = []
    
    # Process each case and timestamp
    for k, v in case_objects.items():
        for k1, v1 in v.items():
            # Skip if no video filename is available
            if v1["video_filename"] != None:
                # Process phase name
                cur_phase = v1['phase'].lower().replace("_", " ")
                if cur_phase == "capsule pulishing":
                    cur_phase = "capsule polishing"
                # Skip idle phases
                if cur_phase == "idle":
                    continue
                    
                # Add video filename to list
                all_videos.append(v1["video_filename"])
                
                # Process segment information
                #cur_segments = ", ".join(segment for segment in set([remove_digits(segment).lower() for segment in v1['objects']]))
                cur_segments = json.dumps(v1["objects"])
                cur_segments = cur_segments.replace("irrigation-aspiration", "irrigation and aspiration handpieces")
                
                # Create prompt for the LLM
                all_messages.append([
                    {
                        "role": "system",
                        "content": """You are a helpful assistant that is expert in cataract surgery videos.
                        Assume that you are given a cataract surgery video that is in the given phase and contains the given anatomical structures and instruments with their corresponding areas and bounding boxes.
                        Your task is to write me question answer pairs. 
                        <rules>
                        The first question & answer pair is about which phase of the surgery are we currently at.
                        The second question & answer pair is about the names of the visible anatomical structures in the current video, using the input.
                        The third question & answer pair is about the names of the visible surgical instruments in the current video, using the input.
                        The fourth question & answer pair is the spatial relation between the objects in the video, which you should infer from area and bounding box values of the objects.
                        From visible segments, figure out what the surgery instruments and the anatomical structures are and classify them accordingly in the answer and give a structured full sentence answer, explaining which segment is an anatomical structure and which segment is an instrument.
                        Do not give single word answers in JSON.
                        Give me your response as a JSON, with the following format:
                        [{"question1": "",
                        "answer1": ""},
                        {"question2": "",
                        "answer2": ""}
                        {"question3": "",
                        "answer3": ""},
                        {"question4": "",
                        "answer4": ""}]
                        
                        Do not give me explanation or summary, I only need a single JSON.
                        </rules>
                        --------------------------------------
                        Example for anatomical structures are: 
                            * Iris
                            * Pupil
                            * Intraocular Lens
                            * Cornea
                        --------------------------------------
                        Example for instruments are: 
                            * Slit/Incision Knife
                            * Gauge 
                            * Spatula 
                            * Capsulorhexis Cystotome 
                            * Phacoemulsifier Tip 
                            * Irrigation-Aspiration handpiece
                            * Lens Injector
                            * Capsulorhexis Forceps 
                            * Katena Forceps.
                        --------------------------------------
                        """,
                    },
                    {
                        "role": "user",
                        "content": f"""Phase: {cur_phase}
                        Anatomical structures and instruments: {cur_segments}
                        """
                    },
                ])
    print("len(all_messages): ", len(all_messages))
    # Generate responses from the LLM
    outputs = llm.chat(all_messages, sampling_params=SamplingParams(max_tokens=2048, temperature=0.6, repetition_penalty=1.05))    
    responses = [output.outputs[0].text.split("</think>")[-1] for output in outputs]
    cleaned_responses = [json_string.strip('```json\n').strip('```').strip() for json_string in responses]

    # Process responses into a dictionary
    responses_dict = dict(zip(range(len(cleaned_responses)), cleaned_responses))
    final_jsons = {}
    
    # Process responses until all are valid JSON
    while len(responses_dict) != 0:
        cur_copy = responses_dict.copy()
        for i, resp in cur_copy.items():
            try:
                # Parse JSON and validate structure
                cur_json = json.loads(resp)
                assert len(cur_json) == 4
                assert "question1" in cur_json[0]
                assert "answer1" in cur_json[0]
                assert len(cur_json[0]) == 2
                assert "question2" in cur_json[1]
                assert "answer2" in cur_json[1]
                assert len(cur_json[1]) == 2
                assert "question3" in cur_json[2]
                assert "answer3" in cur_json[2]
                assert len(cur_json[1]) == 2
                assert "question4" in cur_json[3]
                assert "answer4" in cur_json[3]
                assert len(cur_json[1]) == 2
                
                # Add video filename to JSON
                cur_json.insert(0, {"video_filename": all_videos[i] + ".mp4"})
                final_jsons[i] = cur_json
                del responses_dict[i]
            except Exception as e:
                continue
        
        # Regenerate responses for invalid JSON
        if responses_dict:
            cur_messages = [all_messages[i] for i in range(len(all_messages)) if i in responses_dict]
            cur_outputs = llm.chat(cur_messages, sampling_params=SamplingParams(max_tokens=2048, temperature=0.6, repetition_penalty=1.05))    
            cur_responses = [output.outputs[0].text.split("</think>")[-1] for output in cur_outputs]
            cur_cleaned_responses = [json_string.strip('```json\n').strip('```').strip() for json_string in cur_responses]
            responses_dict = dict(zip(list(responses_dict.keys()), cur_cleaned_responses))

    # Save results to JSON file
    with open("datasets/cataract1k/qa_pairs_without_idle.json", "w") as f:
        json.dump(list(final_jsons.values()), f)

if __name__ == "__main__":
    main() 