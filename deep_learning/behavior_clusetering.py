import os
import json
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pprint
import tqdm

def behavior_clustering(dataset_path):
    with open(os.path.join(dataset_path, 'agents', '0', 'gt_vehicle_bbox', 'data.jsonl'), 'r') as f:
        vehicle_bbox_data = [json.loads(l) for l in f.readlines()]
    vehicle_bbox_data = {x['frame']: x for x in vehicle_bbox_data}

    with open(os.path.join(dataset_path, 'agents', '0', 'gt_location', 'data.jsonl'), 'r') as f:
        ego_location_data = [json.loads(l) for l in f.readlines()]
    ego_location_data = {x['frame']: x for x in ego_location_data}
    
    frames = [x['frame'] for x in ego_location_data.values()]

    sensor_range = 100.0
    # find change lane events
    change_lane_events = {}
    prev_actions = {}

    #event_actions = ['ChangeLaneLeft', 'ChangeLaneRight', 'Straight', 'Left', 'Right']
    event_actions = ['ChangeLaneLeft', 'ChangeLaneRight', 'Straight', 'Left', 'Right', 'LaneFollow', 'Unknown']

    for action in event_actions:
        change_lane_events[action] = {}

    for frame, data in vehicle_bbox_data.items():
        for action in event_actions:
            change_lane_events[action][frame] = []

        for vehicle in data['vehicles']:
            vid = vehicle['id']
            action = vehicle['current_action']

            ego_location = ego_location_data[frame]['location']
            distance = np.linalg.norm(np.array(ego_location)-np.array(vehicle['location']))

            if vid in prev_actions and distance < sensor_range and action != prev_actions[vid] and action in event_actions:
                data = {
                    'frame': frame,
                    'vehicle_id': vid,
                    'event': action,
                    'location': vehicle['location'],
                    'rotation': vehicle['rotation'],
                }
                if 'velocity' in vehicle:
                    # nuScenes has no velocity record
                    data['velocity'] = vehicle['velocity']
                change_lane_events[action][frame].append(data)
            
            prev_actions[vid] = action
    print(change_lane_events)
    # Slidering window for detecting an event
    window_length = 20 * 2
    window_interval = 5 * 2
    frame_windows = [[x for x in frames[i:i+window_length]] for i in range(0, len(frames)-window_length, window_interval)]
    # print('Number of window:', len(frame_windows))

    # Cluster with DBSCAN
    dbscan_eps = 3.5
    min_samples = 1

    clusters = {}

    for action in change_lane_events.keys():
        clusters[action] = []
        for window in frame_windows:
            events = []
            features = []
            
            for frame in window:
                for event in change_lane_events[action][frame]:
                    events.append(event)
                    features.append(event['location'][0:2])
            if len(features) == 0:
                continue
            features = np.array(features)
            db = DBSCAN(eps=dbscan_eps, min_samples=min_samples).fit(features)
            labels = db.labels_
            cluster_ids = np.unique(labels)

            if len(cluster_ids) > 1:
                # found clusters
                for idx in cluster_ids[1:]:
                    cluster_events = [events[i] for i in range(len(events)) if labels[i] == idx]
                    locations = [e['location'] for e in cluster_events]
                    location = np.mean(locations, axis=0).tolist()

                    clusters[action].append({
                        'action': action,
                        'location': location,
                        'start_frame': window[0],
                        'end_frame': window[1],
                        'time_window': window,
                        'size': int(np.sum(labels==idx)),
                        'events': cluster_events,
                    })

            # plt.scatter(features[:, 0], features[:, 1], c=labels)
            # plt.scatter([ego_location_data[f]['location'][0] for f in window], [ego_location_data[f]['location'][1] for f in window], c='red')
            # for i, event in enumerate(events):
            #     #plt.annotate(event['event'][10:11], (features[i, 0], features[i, 1]))
            #     plt.annotate(event['frame'], (features[i, 0], features[i, 1]))
            # plt.show()

    print('Number of clusters:')
    for action in clusters.keys():
        print(f'{action}: {len(clusters[action])}')
    #pprint.pprint(clusters)

    with open(os.path.join(dataset_path, 'behavior_clusters.json'), 'w') as f:
        json.dump(clusters, f)


if __name__ == '__main__':
    #dataset_path = '../data/record'
    dataset_path = '../data/reasoning_nuscenes'
    for scene in tqdm.tqdm(os.listdir(dataset_path)):
        behavior_clustering(os.path.join(dataset_path, scene))
