import json

def preprocess():
    path = 'MeetingBank.nosync/Metadata/MeetingBank.json'
    with open(path, 'r') as f:
        data = json.load(f)
    
    instances = dict()
    for meeting_id in data:
        meeting = data[meeting_id]
        for item_id in meeting['itemInfo']:
            item = meeting['itemInfo'][item_id]
            summary = item['Summary']
            source = ''
            for transcript in item['transcripts']:
                source += f'speaker {transcript["speaker"]}: {transcript["text"]}\n'
            
            key = f'{meeting_id}_{item_id}'
            instances[key] = {
                'summary': summary,
                'source': source
            }
    with open('meetingbank.json', 'w') as f:
        json.dump(instances, f, indent=4)

def split():
    with open('meetingbank.json', 'r') as f:
        data = json.load(f)
    
    groups = ['train', 'validation', 'test']
    for g in groups:
        with open(f'MeetingBank.nosync/Metadata/Splits/{g}.json', 'r') as f:
            original_split = [json.loads(line) for line in f]
        new_split = []
        for instance in original_split:
            new_split.append({
                'id': instance['id'],
                'summary': data[instance['id']]['summary'],
                'source': data[instance['id']]['source']
            })
        with open(f'{g}.json', 'w') as f:
            json.dump(new_split, f, indent=4)
    
        

if __name__ == '__main__':
    # preprocess()
    split()
