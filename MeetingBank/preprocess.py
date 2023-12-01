import json
import random

def preprocess():
    path = 'MeetingBank.nosync/Metadata/MeetingBank.json'
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
        with open(f'MeetingBank.nosync/Metadata/Splits/{g}.json', 'r') as f:
            original_split = [json.loads(line) for line in f]
        segment_split = []
        for instance in original_split:
            segment_split.append({
                'id': instance['id'],
                'summary': segment_level[instance['id']]['summary'],
                'source': segment_level[instance['id']]['source']
            })
        with open(f'MeetingBank/{g}_segment.json', 'w') as f:
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

    json.dump(train_split, open('MeetingBank/train_meeting.json', 'w'), indent=4)
    json.dump(validation_split, open('MeetingBank/validation_meeting.json', 'w'), indent=4)
    json.dump(test_split, open('MeetingBank/test_meeting.json', 'w'), indent=4)


if __name__ == '__main__':
    segment_level, meeting_level = preprocess()
    split_segment(segment_level)
    split_meeting(meeting_level)


