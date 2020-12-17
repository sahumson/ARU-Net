
cv2.rectangle(imgorg, (stats[indexY[i - 1]][0], stats[indexY[i - 1]][1] - (diff - 20)),
					  (stats[indexY[i - 1]][0] + stats[indexY[i - 1]][2], stats[indexY[i - 1]][1] + stats[indexY[i - 1]][3]),
					  (0, 0, 255), 5)

cv2.rectangle(imgorg, (stats[i][0], stats[i][1] - avgdiff), 
					  (stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]), 
					  (0, 0, 255), 5)

cv2.rectangle(imgorg, (int(Coord_Vals[0] + x), int(Coord_Vals[1] + y-avgdiff)),
			          (int(Coord_Vals[2] + x), int(Coord_Vals[3] + y-avgdiff)),
					  (0, 255, 255), 5)
				
cv2.rectangle(imgorg, (stats[i - 1][0], stats[i - 1][1] - avgdiff), 
				      (stats[i - 1][0] + stats[i - 1][2], stats[i - 1][1] + stats[i - 1][3]),
					  (0, 0, 255), 5)

 cv2.rectangle(imgorg, (int(Coord_Vals[0] + x), int(Coord_Vals[1] + y-avgdiff)),
                       (int(Coord_Vals[2] + x), int(Coord_Vals[3] + y-avgdiff)),
                       (0, 255, 255), 5)

cv2.rectangle(imgorg, (stats[indexY[i - 1]][0], stats[indexY[i - 1]][1] - (prediff + 20)),
	                  (stats[indexY[i - 1]][0] + stats[indexY[i - 1]][2],stats[indexY[i - 1]][1] + stats[indexY[i - 1]][3]),
					   (0, 0, 255), 5)

cv2.rectangle(imgorg, (int(Coord_Vals[0] + x), int(Coord_Vals[1] + y-(prediff+20))),
					  (int(Coord_Vals[2] + x), int(Coord_Vals[3] + y-(prediff+20))),
					  (0, 255, 255), 5)
			
cv2.rectangle(imgorg, (stats[indexY[i - 1]][0], stats[indexY[i - 1]][1] - (avgdiff + 20)),
					  (stats[indexY[i - 1]][0] + stats[indexY[i - 1]][2],stats[indexY[i - 1]][1] + stats[indexY[i - 1]][3]),
					  (0, 0, 255), 5)

cv2.rectangle(imgorg, (int(Coord_Vals[0]+x), int(Coord_Vals[1]+y-(avgdiff + 20))),
					  (int(Coord_Vals[2]+x), int(Coord_Vals[3]+y-(avgdiff + 20))),
					  (0, 255, 255), 5)
