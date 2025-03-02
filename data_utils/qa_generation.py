from vllm import LLM, SamplingParams
import ast
import json
import re
import os

def remove_digits(string):
    return re.sub(r'\d+', '', string)

def main():
    # Initialize LLM
    llm = LLM("dwetzel/Mistral-Small-24B-Instruct-2501-GPTQ-INT4",
             gpu_memory_utilization=0.8,
             max_num_batched_tokens=32768,
             max_num_seqs=512,
             max_model_len=32768,
             dtype="bfloat16",
             enforce_eager=False,
             enable_chunked_prefill=True,
             max_seq_len_to_capture=131072,
             enable_prefix_caching=True)

    home_path = os.getenv("HOME")
    with open("datasets/cataract1k/case_objects.json", "r") as f:
        case_objects = json.load(f)

    SYSTEM_PROMPT = """You are a vision language model specialized in extracting information from cataract surgery videos.
    Your task is to analyze provided videos and answer the questions based on the inputted cataract surgery videos."""
    videos_path = "datasets/cataract1k/videos/"

    # Generate messages
    all_messages = []
    all_videos = []
    for k, v in case_objects.items():
        for k1, v1 in v.items():
            if v1["video_filename"] != None:
                cur_phase = v1['phase'].lower().replace("_", " ")
                if cur_phase == "capsule pulishing":
                    cur_phase = "capsule polishing"
                if cur_phase == "idle":
                    continue
                all_videos.append(v1["video_filename"])
                cur_segments = ", ".join(segment for segment in set([remove_digits(segment).lower() for segment in v1['objects']]))
                cur_segments = cur_segments.replace("irrigation-aspiration", "irrigation and aspiration handpieces")
                all_messages.append([
                    {
                        "role": "system",
                        "content": """You are a helpful assistant that is expert in cataract surgery videos.
                        Assume that you are given a cataract surgery video that is in the given phase and contains the given anatomical structures and instruments.
                        Your task is to write me question answer pairs. 
                        <rules>
                        The first question & answer pair is about which phase of the surgery are we currently at.
                        The second question & answer pair is about the visible anatomical structures in the current video, using the input.
                        The third question & answer pair is about the visible surgical instruments in the current video, using the input.
                        From visible segments, figure out what the surgery instruments and the anatomical structures are and classify them accordingly in the answer and give a structured full sentence answer, explaining which segment is an anatomical structure and which segment is an instrument.
                        Do not give single word answers in JSON.
                        Give me your response as a JSON, with the following format:
                        [{"question1": "",
                        "answer1": ""},
                        {"question2": "",
                        "answer2": ""}
                        {"question3": "",
                        "answer3": ""},]
                        
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

    # Generate responses
    outputs = llm.chat(all_messages, sampling_params=SamplingParams(max_tokens=2048, temperature=0.6, repetition_penalty=1.05))    
    responses = [output.outputs[0].text.split("</think>")[-1] for output in outputs]
    cleaned_responses = [json_string.strip('```json\n').strip('```').strip() for json_string in responses]

    # Process responses
    responses_dict = dict(zip(range(len(cleaned_responses)), cleaned_responses))
    final_jsons = {}
    
    while len(responses_dict) != 0:
        cur_copy = responses_dict.copy()
        for i,resp in cur_copy.items():
            try:
                cur_json = json.loads(resp)
                assert len(cur_json) == 3
                assert "question1" in cur_json[0]
                assert "answer1" in cur_json[0]
                assert len(cur_json[0]) == 2
                assert "question2" in cur_json[1]
                assert "answer2" in cur_json[1]
                assert len(cur_json[1]) == 2
                assert "question3" in cur_json[2]
                assert "answer3" in cur_json[2]
                assert len(cur_json[1]) == 2
                cur_json.insert(0, {"video_filename": all_videos[i] + ".mp4"})
                final_jsons[i] = cur_json
                del responses_dict[i]
            except Exception as e:
                continue
                
        if responses_dict:
            cur_messages = [all_messages[i] for i in range(len(all_messages)) if i in responses_dict]
            cur_outputs = llm.chat(cur_messages, sampling_params=SamplingParams(max_tokens=2048, temperature=0.6, repetition_penalty=1.05))    
            cur_responses = [output.outputs[0].text.split("</think>")[-1] for output in cur_outputs]
            cur_cleaned_responses = [json_string.strip('```json\n').strip('```').strip() for json_string in cur_responses]
            responses_dict = dict(zip(list(responses_dict.keys()), cur_cleaned_responses))

    # Save results
    with open("datasets/cataract1k/qa_pairs_without_idle.json", "w") as f:
        json.dump(list(final_jsons.values()), f)

if __name__ == "__main__":
    main() 