import json
import random
from transformers import AutoTokenizer
from tqdm import tqdm


def preprocess():
    path = 'MeetingBank.json'
    with open(path, 'r') as f:
        data = json.load(f)
    
    segment_level = dict()
    meeting_level = dict()

    for meeting_id in data:
        meeting = data[meeting_id]
        whole_transcript = ''
        whole_summary = ''
        for item_id in meeting['itemInfo']:
            item = meeting['itemInfo'][item_id]
            summary = item['Summary']
            source = ''
            for transcript in item['transcripts']:
                source += f'speaker {transcript["speaker"]}: {transcript["text"]}\n'
            
            key = f'{meeting_id}_{item_id}'
            segment_level[key] = {
                'summary': summary,
                'source': source
            }

            whole_transcript += f'Item {item_id}: {source}\n'
            whole_summary += f'Item {item_id}: {summary}\n'
    
        meeting_level[meeting_id] = {
            'id': meeting_id,
            'summary': whole_summary,
            'source': whole_transcript
        }

    return segment_level, meeting_level

def split_segment(segment_level):
    groups = ['train', 'validation', 'test']
    for g in groups:
        with open(f'./Metadata/Splits/{g}.json', 'r') as f:
            original_split = [json.loads(line) for line in f]
        segment_split = []
        for instance in original_split:
            segment_split.append({
                'id': instance['id'],
                'summary': segment_level[instance['id']]['summary'],
                'source': segment_level[instance['id']]['source']
            })
        with open(f'./{g}_segment.json', 'w') as f:
            json.dump(segment_split, f, indent=4)
        print(f'{g}: {len(segment_split)}')

def split_meeting(meeting_level):
    meeting_ids = list(meeting_level.keys())
    total = len(meeting_ids)
    random.shuffle(meeting_ids)

    train_ids = meeting_ids[:int(total*0.8)]
    validation_ids = meeting_ids[int(total*0.8):int(total*0.9)]
    test_ids = meeting_ids[int(total*0.9):]

    train_split = [meeting_level[meeting_id] for meeting_id in train_ids]
    validation_split = [meeting_level[meeting_id] for meeting_id in validation_ids]
    test_split = [meeting_level[meeting_id] for meeting_id in test_ids]

    json.dump(train_split, open('./train_meeting.json', 'w'), indent=4)
    json.dump(validation_split, open('./validation_meeting.json', 'w'), indent=4)
    json.dump(test_split, open('./test_meeting.json', 'w'), indent=4)

def combine_transcript():
    meetings = dict()
    with open('MeetingBank/MeetingBank.json', 'r') as f:
        data = json.load(f)
    
    for meeting_id in data:
        meeting = data[meeting_id]
        meetings[meeting_id] = dict()
        for item_id in meeting['itemInfo']:
            item = meeting['itemInfo'][item_id]
            summary = item['Summary']
            source = ''
            for transcript in item['transcripts']:
                source += f'speaker {transcript["speaker"]}: {transcript["text"]}\n'
            meetings[meeting_id][item_id] = {
                'summary': summary,
                'source': source
            }
    
    json.dump(meetings, open('meetingbank.json', 'w'), indent=4)


def combine_segments(tokenizer, threshold=16000):
    meeting_text = dict()
    meeting_token = dict()

    with open('Metadata/MeetingBank.json', 'r') as f:
        data = json.load(f)

    for meeting_id in tqdm(data):
        meeting = data[meeting_id]
        count = 0

        combined_sources = ''
        combined_summaries = ''
        source_tokens = []
        summary_tokens = []
        curr_length = 0
        for item_id in meeting['itemInfo']:
            item = meeting['itemInfo'][item_id]
            summary = f"Item {item_id} summary: {item['Summary']}"
            source = f'Item {item_id} transcripts:\n'
            for transcript in item['transcripts']:
                source += f'speaker {transcript["speaker"]}: {transcript["text"]}\n'
            
            source_token = tokenizer.encode(source)
            summary_token = tokenizer.encode(summary)
            item_length = len(source_token) + len(summary_token)

            if item_length > threshold:
                print(f'{meeting_id}_{item_id} is too long: {item_length}')
                continue
            elif item_length + curr_length <= threshold:
                source_tokens += source_token
                summary_tokens += summary_token
                combined_sources += source
                combined_summaries += summary
                curr_length += item_length
            else:
                meeting_text[f'{meeting_id}_{count}'] = {
                    'source': combined_sources,
                    'summary': combined_summaries,
                    'length': curr_length,
                    'id': f'{meeting_id}_{count}'
                }
                meeting_token[f'{meeting_id}_{count}'] = {
                    'source': source_tokens,
                    'summary': summary_tokens,
                    'id': f'{meeting_id}_{count}'
                }
                
                combined_sources = source
                combined_summaries = summary
                source_tokens = source_token
                summary_tokens = summary_token
                curr_length = item_length
                count += 1
        
        if curr_length > 0:
            meeting_text[f'{meeting_id}_{count}'] = {
                'source': combined_sources,
                'summary': combined_summaries,
                'length': curr_length,
                'id': f'{meeting_id}_{count}'
            }
            meeting_token[f'{meeting_id}_{count}'] = {
                'source': source_tokens,
                'summary': summary_tokens,
                'id': f'{meeting_id}_{count}'
            }

    meeting_ids = list(meeting_text.keys())
    total = len(meeting_ids)
    random.shuffle(meeting_ids)
    train_ids = meeting_ids[:int(total*0.8)]
    validation_ids = meeting_ids[int(total*0.8):int(total*0.9)]
    test_ids = meeting_ids[int(total*0.9):]
    
    train_text = [meeting_text[meeting_id] for meeting_id in train_ids]
    validation_text = [meeting_text[meeting_id] for meeting_id in validation_ids]
    test_text = [meeting_text[meeting_id] for meeting_id in test_ids]
    train_token = [meeting_token[meeting_id] for meeting_id in train_ids]
    validation_token = [meeting_token[meeting_id] for meeting_id in validation_ids]
    test_token = [meeting_token[meeting_id] for meeting_id in test_ids]

    json.dump(train_text, open('./train_text.json', 'w'), indent=4)
    json.dump(validation_text, open('./validation_text.json', 'w'), indent=4)
    json.dump(test_text, open('./test_text.json', 'w'), indent=4)
    json.dump(train_token, open('./train_token.json', 'w'), indent=4)
    json.dump(validation_token, open('./validation_token.json', 'w'), indent=4)
    json.dump(test_token, open('./test_token.json', 'w'), indent=4)


if __name__ == '__main__':
    # segment_level, meeting_level = preprocess()
    # split_segment(segment_level)
    # split_meeting(meeting_level)
    # combine_transcript()
    
    tokenizer = AutoTokenizer.from_pretrained("Yukang/LongAlpaca-7B")
    combine_segments(tokenizer, threshold=16000)

