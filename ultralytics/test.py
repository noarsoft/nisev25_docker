import csv
from ics import Calendar

import lib2

lib2.detect_qr_by_corner_pdp_fast("C:/yolo8_strip/ultralytics/output_k_fold_boost/0/training/images/11_0.jpg", "C:/yolo8_strip/ultralytics/test_crop")

# # โหลดไฟล์ ICS
# with open("C:\\yolo8_strip\\ultralytics\\calendar.ics", "r", encoding="utf-8") as file:
#     calendar = Calendar(file.read())
#
# # สร้าง CSV
# with open("C:\\yolo8_strip\\ultralytics\\calendar.csv", "w", newline="", encoding="utf-8") as csvfile:
#     fieldnames = ["Subject", "Start Date", "Start Time", "End Date", "End Time", "Description", "Location"]
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#
#     for event in calendar.events:
#         writer.writerow({
#             "Subject": event.name,
#             "Start Date": event.begin.date(),
#             "Start Time": event.begin.time(),
#             "End Date": event.end.date() if event.end else "",
#             "End Time": event.end.time() if event.end else "",
#             "Description": event.description,
#             "Location": event.location
#         })