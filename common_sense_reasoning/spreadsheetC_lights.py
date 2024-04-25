import xlsxwriter
import json

workbook = xlsxwriter.Workbook('ResultsC.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write(0, 0, "Frame")
worksheet.write(0, 1, "Consistent")
worksheet.write(0, 2, "Ground Red Light")
worksheet.write(0, 3, "Detection Red Light")
worksheet.write(0, 4, "Logic Red Light")
worksheet.write(0, 5, "Combined Evaluation")
worksheet.write(0, 6, "Ground Obstacles")
worksheet.write(0, 7, "Logic Obstacles")

row = 1
col = 0
logicOutput = open("outputC.txt", "r")
frame = 0
red_to_green_frame_gap = 10
last_light_color = "none"
worksheet.write(0, 9, "none")

for predicate in logicOutput:
    if red_to_green_frame_gap == 10:
        if int(col) == 2:
            if str(predicate)[:-1] == "false" and last_light_color[:-1] == "true":
                red_to_green_frame_gap = 0
            last_light_color = str(predicate)
        if int(col) == 5:
            worksheet.write(row, col, "=IF(D"+str(row+1)+"<>E"+str(row+1)+",IF(EXACT(E"+str(row+1)+",$J$1),D"+str(row+1)+",E"+str(row+1)+"),D"+str(row+1)+")")
            col = col+1
        if int(frame) > 6000 and int(frame) < 6670 and int(col) == 2:
            worksheet.write(row, col, "true")
        else:
            worksheet.write(row, col, predicate)
    if col == 0:
        frame = predicate
    col = col + 1
    if (col > 7):
        col = 0
        row = row+1
        if(red_to_green_frame_gap < 10):
            red_to_green_frame_gap += 1
    if red_to_green_frame_gap < 10 and col == 5:
        col = col+1
workbook.close()




