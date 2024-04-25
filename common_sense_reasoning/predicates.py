import sys
import json
import csv
import numpy as np
import cv2
import math

vehicleInput = open('gt_vehicle_bbox/data.jsonl')
lightInput = open('gt_traffic_light/data.jsonl')
predLightInput = open('pred_traffic_light/data.jsonl')
clusterInput = open('behavior_clusters.json')
locationInput = open('gt_location/data.jsonl')
outFile = open("predicates.txt", "w")
json_list = list(vehicleInput)
light_list = list(lightInput)
pred_light_list = list(predLightInput)
location_list = list(locationInput)
outputData = []
driverLocation = {}

with open('gnss/data.csv') as csvfile: #Code to extract location of driver vehicle
    lanes_reader = csv.reader(csvfile)
    next(lanes_reader)
    for row in lanes_reader:
        frame = row[0]
        lat_rad = (np.deg2rad(float(row[2])) + np.pi) % (2 * np.pi) - np.pi
        lon_rad = (np.deg2rad(float(row[3])) + np.pi) % (2 * np.pi) - np.pi
        R = 6378135 # Aequatorradii
        x = R * np.sin(lon_rad) * np.cos(lat_rad) # iO
        y = R * np.sin(-lat_rad) # iO
        z = row[4]
        driverLocation[frame] = (x,y,z)
        driverText = "driverLocation(" + str(frame) + ", " + str(x) + ", " + str(y) + ").\n"
        outFile.write(driverText)


cluster_json = json.load(clusterInput)
outputText = ""
for cluster_action in cluster_json:
    for cluster in cluster_json[cluster_action]:
        if math.dist([cluster['location'][0], cluster['location'][1]], [driverLocation[str(cluster['start_frame'])][0], driverLocation[str(cluster['start_frame'])][1]]) < 100:
            #change_action_cluster(FrameStart, FrameEnd, Action, C1X, C1Y, C2X, C2Y).
            ego = 'false'
            if math.dist([cluster['location'][0], cluster['location'][1]], [driverLocation[str(cluster['start_frame'])][0], driverLocation[str(cluster['start_frame'])][1]]) < 50:
                ego = 'true'
            action = cluster['action'].lower()
            start = cluster['start_frame']
            end = cluster['time_window'][::len(cluster['time_window'])-1][1]
            C1X = cluster['location'][0]
            C1Y = cluster['location'][1]
            C2X = cluster['location'][0]+5
            C2Y = cluster['location'][1]+5
            outputText = outputText + "change_action_cluster(" + str(start) + ", " + str(end) + ", " + action + ", " + ego + ', ' + str(C1X) + ", " + str(C1Y) + ", " + str(C2X) + ", " + str(C2Y) 
            outputText = outputText + ").\n"
outFile.write(outputText)

for json_str in json_list:  #Generates predicates about vehicle bounding boxes
    current_json = json.loads(json_str)
    outputText = ""
    for bounding_box_point in current_json['vehicles']:
        frame = str(current_json['frame'])
        outputText = outputText + "property(vehicle, " + str(current_json['frame'])
        outputText = outputText + ", " + str(bounding_box_point['id'])
        outputText = outputText + ", " + str(bounding_box_point['current_action']).lower()
        outputText = outputText + ", " + str(bounding_box_point['velocity'][0])
        outputText = outputText + ", " + str(bounding_box_point['velocity'][1])
        outputText = outputText + ", " + str(bounding_box_point['rotation'][2])
        xLocation = bounding_box_point['bbox']['location'][0]
        yLocation = bounding_box_point['bbox']['location'][2]
        outputText = outputText + ", " + str(bounding_box_point['bbox']['location'][0]+driverLocation[frame][0])
        outputText = outputText + ", " + str(bounding_box_point['bbox']['location'][2]+driverLocation[frame][1])
        outputText = outputText + ", " + str(bounding_box_point['bbox']['location'][0]+bounding_box_point['bbox']['extent'][0]+driverLocation[frame][0])
        outputText = outputText + ", " + str(bounding_box_point['bbox']['location'][2]+bounding_box_point['bbox']['extent'][1]+driverLocation[frame][1])
        outputText = outputText + ").\n"
    outFile.write(outputText)

for light_str in light_list:  #Generates predicates about vehicle bounding boxes
    current_json = json.loads(light_str)
    outputText = ""
    if str(current_json[ 'is_at_traffic_light']) == "True":
        frame = str(current_json['frame'])
        outputText = outputText + "property(traffic_light, " + str(current_json['frame'])
        outputText = outputText + ", " + str(current_json['current_traffic_light']['id'])
        outputText = outputText + ", " + str(current_json['current_traffic_light']['first_vehicle_id'])
        outputText = outputText + ", " + str(current_json['current_traffic_light']['state']).lower()
        xLocation = current_json['current_traffic_light']['bbox']['location'][0]
        yLocation = current_json['current_traffic_light']['bbox']['location'][2]
        outputText = outputText + ", " + str(current_json['current_traffic_light']['bbox']['location'][0]+driverLocation[frame][0])
        outputText = outputText + ", " + str(current_json['current_traffic_light']['bbox']['location'][2]+driverLocation[frame][1])
        outputText = outputText + ", " + str(current_json['current_traffic_light']['bbox']['location'][0]+current_json['current_traffic_light']['bbox']['extent'][0]+driverLocation[frame][0])
        outputText = outputText + ", " + str(current_json['current_traffic_light']['bbox']['location'][2]+current_json['current_traffic_light']['bbox']['extent'][1]+driverLocation[frame][1])
        outputText = outputText + ").\n"
        outFile.write(outputText)

for pred_light in pred_light_list: 
    current_json = json.loads(pred_light)
    outputText = ""
    outputText = outputText + "pred_light("+ str(current_json['frame'])
    outputText = outputText + ", " + str(str(current_json['is_under_red_traffic_light'])=='1').lower()
    outputText = outputText + ").\n"
    outFile.write(outputText)

for location in location_list: 
    current_json = json.loads(location)
    outputText = ""
    outputText = outputText + "driver_rotation("+ str(current_json['frame'])
    outputText = outputText + ", " + str(current_json['rotation'][2])
    outputText = outputText + ").\n"
    outFile.write(outputText)


for light_str in light_list:  
    current_json = json.loads(light_str)
    outputText = ""
    outputText = outputText + "ground_light("+ str(current_json['frame'])
    if str(current_json['current_traffic_light']) == 'None':
        outputText = outputText + ", " + str(current_json['current_traffic_light']).lower()
    else:
        outputText = outputText + ", " + (str(current_json['current_traffic_light']['state'] == 'Red')).lower()
    outputText = outputText + ").\n"
    outFile.write(outputText)

frameList = "frames(["
for json_str in json_list:  #Generates predicates about frame list
    current_json = json.loads(json_str)
    frame = str(current_json['frame'])
    frameList = frameList + str(frame) + ", "
frameList = frameList[:-2]+"]).\n"
outFile.write(frameList)

junctionInput = open('gt_junctions/data.jsonl')
json_list = list(junctionInput)
upText = ""
downText = ""
leftText = ""
rightText = ""
for json_str in json_list:
    current_json = json.loads(json_str)
    for bounding_box_point in current_json['junctions']:
        junctionX = ((bounding_box_point['bounding_box']['location'][0]) + (bounding_box_point['bounding_box']['location'][0]+bounding_box_point['bounding_box']['extent'][0]))/2
        junctionY = ((bounding_box_point['bounding_box']['location'][1]) + (bounding_box_point['bounding_box']['location'][1]+bounding_box_point['bounding_box']['extent'][1]))/2
        if math.dist([junctionX, junctionY], [driverLocation[str(current_json['frame'])][0], driverLocation[str(current_json['frame'])][1]]) < 50:
            outputText = "property(intersection, " + str(current_json['frame'])
            outputText = outputText + ", " + str(bounding_box_point['id'])
            outputText = outputText + ", " + str(bounding_box_point['bounding_box']['location'][0])
            outputText = outputText + ", " + str(bounding_box_point['bounding_box']['location'][1])
            outputText = outputText + ", " + str(bounding_box_point['bounding_box']['location'][0]+bounding_box_point['bounding_box']['extent'][0])
            outputText = outputText + ", " + str(bounding_box_point['bounding_box']['location'][1]+bounding_box_point['bounding_box']['extent'][1])
            outputText = outputText + ").\n"
            outFile.write(outputText)
            #property(intersection_side_up, Object_id, IS1X, IS1Y, W, H),
            upText = upText + "property(intersection_side_up, " + str(current_json['frame']) + ", " + str(bounding_box_point['id'])
            upText = upText + ", " + str(bounding_box_point['bounding_box']['location'][0])
            upText = upText + ", " + str(bounding_box_point['bounding_box']['location'][1]+bounding_box_point['bounding_box']['extent'][1]/2)
            upText = upText + ", " + str(bounding_box_point['bounding_box']['extent'][0])
            upText = upText + ", " + str(bounding_box_point['bounding_box']['extent'][1]/2) + ").\n"
            downText = downText + "property(intersection_side_down, " + str(current_json['frame']) + ", " + str(bounding_box_point['id'])
            downText = downText + ", " + str(bounding_box_point['bounding_box']['location'][0])
            downText = downText + ", " + str(bounding_box_point['bounding_box']['location'][1])
            downText = downText + ", " + str(bounding_box_point['bounding_box']['extent'][0])
            downText = downText + ", " + str(bounding_box_point['bounding_box']['extent'][1]/2) + ").\n"
            lefText = leftText + "property(intersection_side_left, " + str(current_json['frame']) + ", " + str(bounding_box_point['id'])
            lefText = lefText + ", " + str(bounding_box_point['bounding_box']['location'][0])
            lefText = lefText + ", " + str(bounding_box_point['bounding_box']['location'][1])
            lefText = lefText + ", " + str(bounding_box_point['bounding_box']['extent'][0]/2)
            lefText = lefText + ", " + str(bounding_box_point['bounding_box']['extent'][1]) + ").\n"
            rightText = rightText + "property(intersection_side_right, " + str(current_json['frame']) + ", " + str(bounding_box_point['id'])
            rightText = rightText + ", " + str(bounding_box_point['bounding_box']['location'][0]+bounding_box_point['bounding_box']['extent'][0]/2)
            rightText = rightText + ", " + str(bounding_box_point['bounding_box']['location'][1])
            rightText = rightText + ", " + str(bounding_box_point['bounding_box']['extent'][0]/2)
            rightText = rightText + ", " + str(bounding_box_point['bounding_box']['extent'][1]) + ").\n"
outFile.write(upText)
outFile.write(downText)
outFile.write(leftText)
outFile.write(rightText)
outFile.close()

with open('gnss/data.csv') as csvfile:
    lanes_reader = csv.reader(csvfile)
    next(lanes_reader)
    for row in lanes_reader:
        frame = row[0]
        lat_rad = (np.deg2rad(float(row[2])) + np.pi) % (2 * np.pi) - np.pi
        lon_rad = (np.deg2rad(float(row[3])) + np.pi) % (2 * np.pi) - np.pi
        R = 6378135 # Aequatorradii
        x = R * np.sin(lon_rad) * np.cos(lat_rad) # iO
        y = R * np.sin(-lat_rad) # iO
        z = row[4]





