# import cv2
# import json
# import numpy as np
#
# frame = cv2.imread('/home/id/Datasets/caltech/data/images/set00_V000_355.png')
#
# with open('/home/id/Datasets/caltech/data/annotations.json', 'r') as f:
#     j = json.load(f)
#
# try:
#     for key in j:
#         print(int(key[3:]))
#         print(key)
#         # pos = np.int0(np.around(key['pos']))
#         # if key['lbl'] == 'person':
#         #     cv2.rectangle(frame, (pos[0], pos[1]), (pos[0] + pos[2], pos[1] + pos[3]), (0, 255, 0), thickness=1)
#         # elif key['lbl'] == 'people':
#         #     cv2.rectangle(frame, (pos[0], pos[1]), (pos[0] + pos[2], pos[1] + pos[3]), (255, 0, 0), thickness=1)
# except KeyError:
#     print('nimei')
#
#     # altered log nFrame logLen frames maxObj
#
# cv2.imshow('Video', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # print(j['set00']['V000']['frames']['344'])


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

print(float(2)/3)
